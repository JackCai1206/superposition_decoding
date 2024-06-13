# %%
import torch
from torch import nn
import tqdm
from transformers import AutoTokenizer, Trainer, EvalPrediction, set_seed, HfArgumentParser, TrainingArguments, LlamaForCausalLM, AutoModelForCausalLM, default_data_collator, PreTrainedModel, GenerationMixin
from datasets import load_dataset, Dataset
from transformer_lens import HookedTransformer
import os
from dataclasses import dataclass
# %%
from typing import List, Literal, Optional, Sequence, Tuple, Union
from wandb_callback import WandbPredictionProgressCallback
from torch import Tensor
from jaxtyping import Float, Int
from transformer_lens.utilities import devices
from transformer_lens.past_key_value_caching import HookedTransformerKeyValueCache
import transformer_lens.utils as utils

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
        # model.requires_grad_(False)
        self.multiplexer = Multiplexer(n_streams, model.cfg.d_model)
        self.demultiplexer = Demultiplexer(n_streams, model.cfg.d_model)
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, input_ids, attention_mask=None, labels=None, merge_layer=0, **kwargs):
        assert len(input_ids) == self.n_streams
        # assert len(attention_mask) == self.n_streams

        device = devices.get_device_for_block_index(0, self.model.cfg)
        input_ids = [input_id.to(device) for input_id in input_ids]
        if attention_mask is not None:
            attention_mask = [mask.to(device) for mask in attention_mask]
        else:
            attention_mask = [torch.ones_like(input_id) for input_id in input_ids]
        resids = [self.model.forward(input_ids[i], attention_mask=attention_mask[i], stop_at_layer=merge_layer, **kwargs) for i in range(self.n_streams)]

        resid_comb = self.multiplexer(resids)
        attention_mask_comb = torch.stack(attention_mask).prod(dim=0)

        last_resid_comb = self.model.forward(resid_comb, attention_mask=attention_mask_comb, start_at_layer=merge_layer, stop_at_layer=self.model.cfg.n_layers, **kwargs)

        last_resid = self.demultiplexer(last_resid_comb)

        if labels is not None:
            logits_and_losses = [self.model.forward(last_resid[i], tokens=labels[i], start_at_layer=self.model.cfg.n_layers, return_type='both') for i in range(self.n_streams)]
            logits = [logits_and_losses[i][0] for i in range(self.n_streams)]
            losses = [logits_and_losses[i][1] for i in range(self.n_streams)]
            loss = sum(losses)
        else:
            logits = [self.model.forward(last_resid[i], start_at_layer=self.model.cfg.n_layers, return_type='logits') for i in range(self.n_streams)]
            loss = None

        return (
            loss, 
            logits,
            labels
        )
    
    @torch.inference_mode()
    def generate(
        self,
        input: Tuple[Float[torch.Tensor, "batch pos"]] = "",
        max_new_tokens: int = 10,
        stop_at_eos: bool = True,
        eos_token_id: Optional[int] = None,
        do_sample: bool = True,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: float = 1.0,
        freq_penalty: float = 0.0,
        use_past_kv_cache: bool = True,
        prepend_bos: Optional[bool] = True,
        padding_side: Optional[Literal["left", "right"]] = "right",
        verbose: bool = True,
    ) -> Tuple[Int[torch.Tensor, "batch pos_plus_new_tokens"]]:
        tokens = input
        if torch.is_tensor(tokens):
            assert tokens.ndim == 3 and tokens.shape[0] == self.n_streams
            tokens = tokens.unbind(0)
        else:
            assert len(tokens) == self.n_streams
        assert isinstance(tokens[0], torch.Tensor)
        batch_size, ctx_length = tokens[0].shape
        device = devices.get_device_for_block_index(0, self.model.cfg)
        tokens = [t.to(device) for t in tokens]
        if use_past_kv_cache:
            past_kv_cache = HookedTransformerKeyValueCache.init_cache(
                self.model.cfg, self.model.cfg.device, batch_size
            )
        else:
            past_kv_cache = None

        stop_tokens: List[int] = []
        eos_token_for_padding = 0
        assert self.model.tokenizer is not None
        if stop_at_eos:
            tokenizer_has_eos_token = (
                self.model.tokenizer is not None and self.model.tokenizer.eos_token_id is not None
            )
            if eos_token_id is None:
                assert (
                    tokenizer_has_eos_token
                ), "Must pass a eos_token_id if stop_at_eos is True and tokenizer is None or has no eos_token_id"

                eos_token_id = self.model.tokenizer.eos_token_id

            if isinstance(eos_token_id, int):
                stop_tokens = [eos_token_id]
                eos_token_for_padding = eos_token_id
            else:
                # eos_token_id is a Sequence (e.g. list or tuple)
                stop_tokens = eos_token_id
                eos_token_for_padding = (
                    self.model.tokenizer.eos_token_id if tokenizer_has_eos_token else eos_token_id[0]
                )

        # An array to track which sequences in the batch have finished.
        finished_sequences = [torch.zeros(batch_size, dtype=torch.bool, device=self.model.cfg.device) for _ in range(self.n_streams)]

        # Currently nothing in HookedTransformer changes with eval, but this is here in case
        # that changes in the future.
        self.eval()
        for index in tqdm.tqdm(range(max_new_tokens), disable=not verbose):
            # While generating, we keep generating logits, throw away all but the final logits,
            # and then use those logits to sample from the distribution We keep adding the
            # sampled tokens to the end of tokens.
            if use_past_kv_cache:
                # We just take the final tokens, as a [batch, 1] tensor
                if index > 0:
                    logits = self.forward(
                        [t[:, -1:] for t in tokens],
                        return_type="logits",
                        prepend_bos=prepend_bos,
                        padding_side=padding_side,
                        past_kv_cache=past_kv_cache,
                    )[1]
                else:
                    logits = self.forward(
                        tokens,
                        return_type="logits",
                        prepend_bos=prepend_bos,
                        padding_side=padding_side,
                        past_kv_cache=past_kv_cache,
                    )[1]
            else:
                # We input the entire sequence, as a [batch, pos] tensor, since we aren't using
                # the cache.
                logits = self.forward(
                    tokens,
                    return_type="logits",
                    prepend_bos=prepend_bos,
                    padding_side=padding_side,
                )[1]
            final_logits = [l[:, -1, :] for l in logits]

            if do_sample:
                sampled_tokens = [
                    utils.sample_logits(
                        fl,
                        top_k=top_k,
                        top_p=top_p,
                        temperature=temperature,
                        freq_penalty=freq_penalty,
                        tokens=tokens,
                    ).to(devices.get_device_for_block_index(0, self.model.cfg))
                for fl in final_logits]
            else:
                sampled_tokens = [
                    fl.argmax(-1).to(
                        devices.get_device_for_block_index(0, self.model.cfg)
                    )
                for fl in final_logits]

            if stop_at_eos:
                # For all unfinished sequences, add on the next token. If a sequence was
                # finished, throw away the generated token and add eos_token_for_padding
                # instead.
                for st, fs in zip(sampled_tokens, finished_sequences):
                    st[fs] = eos_token_for_padding
                for i, fs in enumerate(finished_sequences):
                    fs.logical_or_(
                        torch.isin(
                            sampled_tokens[i].to(self.model.cfg.device),
                            torch.tensor(stop_tokens).to(self.model.cfg.device),
                        )
                    )

            tokens = [torch.cat([tokens[i], sampled_tokens[i].unsqueeze(-1)], dim=-1) for i in range(self.n_streams)]

            if stop_at_eos and all([fs.all() for fs in finished_sequences]):
                break

        return tokens

    
@dataclass
class ScriptArguments:
    model_id: str = "meta-llama/Meta-Llama-3-8B"
    n_streams: int = 1
    block_size: int = None
    random_weights: bool = False
    num_proc: int = 24
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
if args.random_weights:
    transformer_model.init_weights()

# transformer_model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.bfloat16, return_dict=False, attn_implementation="eager")
if args.block_size is None:
    block_size = transformer_model.cfg.n_ctx


# %%
model = MultiplexedModel(args.n_streams, transformer_model).to(torch.bfloat16).to(train_args.device)

# model = transformer_model
# print number of trainable parameters
print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# %%

datasets = (load_dataset('roneneldan/TinyStories', split='train') for _ in range(args.n_streams))
eval_datasets = (load_dataset('roneneldan/TinyStories', split='validation[:200]') for _ in range(args.n_streams))

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

def tokenization(batch):
    batch = tokenizer(batch['text'])
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
        datasets = [dataset.shard(num_shards=args.num_proc, index=s, contiguous=True) for dataset in datasets]
        dataset_len = len(datasets[0])
        for ex_i in range(dataset_len):
            yield {
                f'{key}': [ds[ex_i][key] for ds_i, ds in enumerate(datasets)]
                for key in columns
            }

def zip_datasets(datasets: Tuple[Dataset]):
    datasets = tuple([
        dataset.map(tokenization, batched=True, num_proc=args.num_proc, remove_columns=list(dataset.features))
        .map(group_texts, batched=True, num_proc=args.num_proc)
        .shuffle(seed=42 + i)
    for i, dataset in enumerate(list(datasets))])
    dataset = Dataset.from_generator(get_zipped_dataset_generator, gen_kwargs={'datasets': datasets, 'shards': list(range(args.num_proc))}, num_proc=args.num_proc, cache_dir=f'huggingface_cache/n_streams-{args.n_streams}/')
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

# train_args.set_logging(first_step=True, report)

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
