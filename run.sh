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

# WANDB_MODE=disabled
# CUDA_VISIBLE_DEVICES=0,1 

    # True True False False \
    # True False False False \
    # True False True False \

for from_pretrained freeze_transformer random_weights use_lora in \
    True False False True
; do
    for n_streams in 2; do
        for merge_layer in 10; do
            CUDA_VISIBLE_DEVICES=0 WANDB_PROJECT=superposition-decoding WANDB_MODE=disabled python -m pdb ft_multiplex_llama.py \
                --n_streams=$n_streams \
                --merge_layer=$merge_layer \
                --model_attribute='transformer' \
                --layers_attribute='h' \
                --freeze_transformer=$freeze_transformer \
                \
                --model_id='gpt2' \
                --from_pretrained=$from_pretrained \
                --random_weights=$random_weights \
                --use_lora=True \
                \
                --output_dir='output' \
                --run_name='n_streams='$n_streams'-merge_layer='$merge_layer'-from_pretrained='$from_pretrained'-freeze_transformer='$freeze_transformer'-random_weights='$random_weights \
                --overwrite_output_dir=True \
                --num_train_epochs=2 \
                --max_steps=5000 \
                --report_to='wandb' \
                --logging_steps=10 \
                --bf16=True \
                --tf32=True \
                --adam_beta2=0.99 \
                --use_cpu=False \
                --gradient_accumulation_steps=32 \
                --eval_strategy="steps" \
                --eval_steps=250 \
                --batch_eval_metrics=True\
                --learning_rate=5e-4 \
                --lr_scheduler_type='cosine' \
                --warmup_steps=200 \
                --torch_compile=False \
                --per_device_train_batch_size=8 \
                --per_device_eval_batch_size=8
        done
    done
done

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
