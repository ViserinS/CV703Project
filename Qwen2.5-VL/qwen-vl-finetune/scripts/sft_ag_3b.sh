#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
# Distributed training configuration
MASTER_ADDR="127.0.0.1"
MASTER_PORT=$(shuf -i 20001-29999 -n 1)
# NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)  # Auto-detect GPU count
NPROC_PER_NODE=6
# DeepSpeed configuration
deepspeed=./scripts/zero3.json

# Model configuration
# Change the model path to the full path
llm="/remote-home/peachilk/codebase/Qwen2.5-VL/qwen-vl-finetune/Qwen2.5-VL-3B-Instruct"

# Training hyperparameters
lr=5e-6
batch_size=4
grad_accum_steps=4

# Training entry point
entry_file=qwenvl/train/train_qwen.py

# Dataset configuration - use your custom dataset
datasets="cassava_disease_deepseek_v3%100"  # Use 100% of your dataset

# Output configuration
run_name="qwen2vl-cassava-disease"
output_dir=./output/cassava_disease_deepseek_v3_3epoch

# Training arguments
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path "${llm}" \
    --dataset_use ${datasets} \
    --data_flatten True \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 3 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 50176 \
    --min_pixels 784 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 3 \
    --learning_rate ${lr} \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name}"

# Launch training
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}

         
