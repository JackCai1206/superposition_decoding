set -e

# torchrun --standalone --nnodes=1 --nproc_per_node=2 ft_multiplex_llama.py \
#     --output_dir='output' \
#     --overwrite_output_dir=True \
#     --num_train_epochs=1 \
#     --max_steps=1000 \
#     --report_to='wandb' \
#     --logging_steps=1 \
#     --bf16=True \
#     --use_cpu=False \
#     --per_device_train_batch_size=4 \
#     --per_device_eval_batch_size=4 \
#     --gradient_accumulation_steps=2 \
#     --ddp_find_unused_parameters=False \
#     --fsdp='full_shard offload auto_wrap' \
#     --fsdp_config='{"activation_checkpointing": false, "transformer_layer_cls_to_wrap": ["LlamaDecoderLayer"]}' \
#     --eval_strategy="steps" \
#     --eval_steps=10

# WANDB_MODE=offline
CUDA_VISIBLE_DEVICES=1 WANDB_PROJECT=superposition-decoding python ft_multiplex_llama.py \
    --n_streams=1 \
    --random_weights=False \
    --model_id='gpt2' \
    --output_dir='output' \
    --run_name='test' \
    --overwrite_output_dir=True \
    --num_train_epochs=1 \
    --max_steps=10000 \
    --report_to='none' \
    --logging_steps=10 \
    --bf16=True \
    --use_cpu=False \
    --per_device_train_batch_size=8 \
    --per_device_eval_batch_size=8 \
    --gradient_accumulation_steps=4 \
    --eval_strategy="steps" \
    --eval_steps=40 \
    --batch_eval_metrics=True
    # --torch_compile=True \

# python -m pdb -c c ft_multiplex_llama.py \
#     output_dir='output' \
#     overwrite_output_dir=True \
#     num_train_epochs=1 \
#     max_steps=1000 \
#     report_to='none' \
#     logging_steps=1 \
#     bf16=True \
#     use_cpu=True \
#     per_device_train_batch_size=8 \
#     per_device_eval_batch_size=8 \
#     eval_strategy="steps" \
#     eval_steps=10
