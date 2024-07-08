import torch
from transformers import AutoTokenizer, Trainer, EvalPrediction, set_seed, HfArgumentParser, TrainingArguments, LlamaForCausalLM, LlamaConfig, AutoModelForCausalLM, AutoConfig, GPT2LMHeadModel
from peft import get_peft_model, LoraConfig, TaskType
import os
from wandb_callback import WandbPredictionProgressCallback
from typing import cast

from lib.config import ScriptArguments
# from lib.modeling_old import MultiplexedModel
from lib.modeling import MultiplexedModel, MultiplexedModelConfig
from lib.data import get_datasets

# %%
# Check if a GPU is available and set the device
local_rank = int(os.environ.get("LOCAL_RANK", 0))
if local_rank != 0:
    import sys
    sys.stdout = open(os.devnull, 'w')

set_seed(42)

args, train_args= HfArgumentParser((ScriptArguments, TrainingArguments)).parse_args_into_dataclasses()
args = cast(ScriptArguments, args)

# Load the tokenizer from the Hugging Face library
tokenizer = AutoTokenizer.from_pretrained(args.model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'
tokenizer.add_bos_token = True
tokenizer.add_eos_token = True

if args.from_pretrained:
    # transformer_model = HookedTransformer.from_pretrained_no_processing(args.model_id, torch_dtype=torch.bfloat16, device=train_args.device)
    if args.random_weights:
        model_cfg = AutoConfig.from_pretrained(args.model_id)
        transformer_model = AutoModelForCausalLM.from_config(model_cfg, torch_dtype=torch.bfloat16, return_dict=False, attn_implementation="eager")
    else:
        transformer_model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.bfloat16, return_dict=False, attn_implementation="eager")
    args.block_size = transformer_model.config.max_position_embeddings
    args.d_model = transformer_model.config.hidden_size
    args.n_layers = transformer_model.config.num_hidden_layers
    args.n_heads = transformer_model.config.num_attention_heads
else:
    # model_cfg = HookedTransformerConfig(
    #     d_model=384,
    #     d_head=64,
    #     n_layers=6,
    #     n_ctx=512,
    #     act_fn='silu',
    #     positional_embedding_type='rotary',
    #     normalization_type='RMS',
    #     device=None,
    #     dtype=torch.bfloat16,
    # )
    # transformer_model = HookedTransformer(model_cfg, tokenizer, move_to_device=False)
    model_cfg = LlamaConfig(
        hidden_size=args.d_model,
        num_hidden_layers=args.n_layers,
        num_attention_heads=args.n_heads,
        max_position_embeddings=args.block_size,
        vocab_size=tokenizer.vocab_size,
    )
    transformer_model = LlamaForCausalLM(model_cfg)

print(transformer_model)

if args.use_lora:
    lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM)
    transformer_model = get_peft_model(transformer_model, lora_config)
    transformer_model.print_trainable_parameters()

# Not pass in args directly because MultiplexModel subclasses from PreTrainedModel and that has to take in PretrainedConfig
config = MultiplexedModelConfig(
    merge_layer=args.merge_layer,
    n_streams=args.n_streams,
    freeze_transformer=args.freeze_transformer,
    hidden_size=args.d_model,
    model_attribute=args.model_attribute,
    layers_attribute=args.layers_attribute
)
model = MultiplexedModel(config, transformer_model)

# print number of trainable parameters
print(f"Number of trainable parameters in transformer: {sum(p.numel() for p in transformer_model.parameters() if p.requires_grad)}")
print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

dataset, eval_dataset = get_datasets(args, tokenizer)

def preprocess_logits_for_metrics(logits, labels):
    # breakpoint()
    return 

def compute_metrics(pred: EvalPrediction, compute_result=False):
    # ground_truth = []
    # prediction = []
    # for s in range(args.n_streams):
    #     labels = tokenizer.batch_decode(pred.label_ids[s])
    #     # logits = pred.predictions[0].argmax(axis=-1)
    #     prediction_text = tokenizer.batch_decode(pred.predictions[1][s])
    #     ground_truth.append(labels)
    #     prediction.append(prediction_text)
    # return {"ground_truth": ground_truth, "prediction": prediction}
    return {}

def collate_fn(examples):
    columns = examples[0].keys()
    return {
        key: torch.stack([ex[key] for ex in examples], dim=1) # first dimension is n_streams
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
