# %%
import torch
from torch import nn
from transformers import AutoTokenizer, Trainer, EvalPrediction, set_seed, HfArgumentParser, TrainingArguments, LlamaForCausalLM, AutoModelForCausalLM, default_data_collator
from datasets import load_dataset, Dataset
from transformer_lens import HookedTransformer
import os
from dataclasses import dataclass
# %%
from typing import List, Sequence, Tuple
from wandb_callback import WandbPredictionProgressCallback
from torch import Tensor


class Multiplexer(nn.Module):
    def __init__(self, n_streams, d_model):
        super().__init__()
        self.n_streams = n_streams
        self.phis = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_streams)])

    def forward(self, x: List[Tensor]):
        assert len(x) == self.n_streams
        # x = [xx.to(self.phis[0].weight.data.dtype) for xx in x]
        x = [phi(xi) for xi, phi in zip(x, self.phis)]
        x = torch.stack(x).mean(dim=0)
        return x

class Demultiplexer(nn.Module):
    def __init__(self, n_streams, d_model):
        super().__init__()
        self.n_streams = n_streams
        self.phis = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_streams)])

    def forward(self, x):
        # x = x.to(self.phis[0].weight.data.dtype)
        x = [phi(x) for phi in self.phis]
        return x

class MultiplexedModel(nn.Module):
    def __init__(self, n_streams, model: HookedTransformer):
        super().__init__()
        self.n_streams = n_streams
        self.model = model
        model.requires_grad_(False)
        self.multiplexer = Multiplexer(n_streams, model.cfg.d_model)
        self.demultiplexer = Demultiplexer(n_streams, model.cfg.d_model)
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, input_ids, attention_mask, labels, merge_layer=0, **kwargs):
        assert len(input_ids) == self.n_streams
        assert len(attention_mask) == self.n_streams

        resids = [self.model.forward(input_ids[i], attention_mask=attention_mask[i], stop_at_layer=merge_layer, **kwargs) for i in range(self.n_streams)]

        resid_comb = self.multiplexer(resids)
        attention_mask_comb = torch.stack(attention_mask).prod(dim=0)

        last_resid_comb = self.model.forward(resid_comb, attention_mask=attention_mask_comb, start_at_layer=merge_layer, stop_at_layer=self.model.cfg.n_layers, **kwargs)

        last_resid = self.demultiplexer(last_resid_comb)

        logits_and_losses = [self.model.forward(last_resid[i], tokens=labels[i], start_at_layer=self.model.cfg.n_layers, return_type='both') for i in range(self.n_streams)]
        logits = [logits_and_losses[i][0] for i in range(self.n_streams)]
        losses = [logits_and_losses[i][1] for i in range(self.n_streams)]
        loss = sum(losses)

        return (
            loss, 
            logits,
            labels
        )
@dataclass
class ScriptArguments:
    model_id: str = "meta-llama/Meta-Llama-3-8B"
    n_streams: int = 4
    block_size: int = 1024
# %%
# Check if a GPU is available and set the device
local_rank = int(os.environ.get("LOCAL_RANK", 0))
if local_rank != 0:
    import sys
    sys.stdout = open(os.devnull, 'w')

set_seed(42)

args, train_args= HfArgumentParser((ScriptArguments, TrainingArguments)).parse_args_into_dataclasses()

# Load the tokenizer from the Hugging Face library
tokenizer = AutoTokenizer.from_pretrained(args.model_id)

transformer_model = HookedTransformer.from_pretrained_no_processing(args.model_id, torch_dtype=torch.bfloat16, device=train_args.device)
# transformer_model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.bfloat16, return_dict=False, attn_implementation="eager")
block_size = transformer_model.cfg.n_ctx


# %%
model = MultiplexedModel(args.n_streams, transformer_model).to(torch.bfloat16).to(train_args.device)

# model = transformer_model
# print number of trainable parameters
print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# %%

datasets = (load_dataset('roneneldan/TinyStories', split='train[:20000]') for _ in range(args.n_streams))
eval_datasets = (load_dataset('roneneldan/TinyStories', split='validation[:160]') for _ in range(args.n_streams))

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

def tokenization(batch):
    batch = tokenizer(batch['text'], padding="max_length")
    return batch

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result['labels'] = result['input_ids'].copy()
    return result

def get_zipped_dataset_generator(datasets: Tuple[Dataset], shards: List[int]):
    columns = datasets[0].column_names
    for s in shards:
        datasets = [dataset.shard(num_shards=16, index=s, contiguous=True) for dataset in datasets]
        dataset_len = len(datasets[0])
        for ex_i in range(dataset_len):
            yield {
                f'{key}': [ds[ex_i][key] for ds_i, ds in enumerate(datasets)]
                for key in columns
            }

def zip_datasets(datasets: Tuple[Dataset]):
    datasets = tuple([
        dataset.map(tokenization, batched=True, num_proc=24, remove_columns=list(dataset.features))
        .map(group_texts, batched=True, num_proc=24)
        .shuffle(seed=42 + i)
    for i, dataset in enumerate(list(datasets))])
    dataset = Dataset.from_generator(get_zipped_dataset_generator, gen_kwargs={'datasets': datasets, 'shards': list(range(16))}, num_proc=24)
    dataset.set_format(type='torch', output_all_columns=True)
    # dataset = dataset.map(concat_shards, batched=False, num_proc=16)
    return dataset

dataset = zip_datasets(datasets)
eval_dataset = zip_datasets(eval_datasets)
# dataset = zip_datasets((dataset_1,))
# eval_dataset = zip_datasets((eval_dataset_1,))
# dataset = dataset_1.map(tokenization, batched=True, num_proc=16).shuffle(seed=42)
# eval_dataset = eval_dataset_1.map(tokenization, batched=True, num_proc=16).shuffle(seed=42)

# %%
# from evaluate import load
# perplexity = load("perplexity", module_type="metric")

def preprocess_logits_for_metrics(logits, labels):
    # breakpoint()
    return 

def compute_metrics(pred: EvalPrediction, compute_result=False):
    ground_truth = []
    prediction = []
    for s in range(args.n_streams):
        labels = tokenizer.batch_decode(pred.label_ids[s])
        # logits = pred.predictions[0].argmax(axis=-1)
        prediction_text = tokenizer.batch_decode(pred.predictions[1][s])
        ground_truth.append(labels)
        prediction.append(prediction_text)
    return {"ground_truth": ground_truth, "prediction": prediction}

def collate_fn(examples):
    columns = examples[0].keys()
    return {
        key: [
            torch.stack([ex[key][i] for ex in examples])
            for i in range(args.n_streams)
        ]
        for key in columns
    }

trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=dataset,
    eval_dataset=eval_dataset,
    data_collator=collate_fn,
    # data_collator=default_data_collator,
    # compute_metrics=compute_metrics,
    # preprocess_logits_for_metrics=preprocess_logits_for_metrics
)

# Instantiate the WandbPredictionProgressCallback
progress_callback = WandbPredictionProgressCallback(
    trainer=trainer,
    tokenizer=tokenizer,
    val_dataset=eval_dataset,
    num_samples=10,
    freq=1,
    args=args
)

trainer.add_callback(progress_callback)

trainer.train()
trainer.evaluate()
