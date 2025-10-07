#!/bin/bash
# 获取当前时间
current_time=$(date "+%Y-%m-%d_%H:%M:%S")
echo "当前时间: $current_time"
train_samples=512
train_batch_size=256
micro_batch_size=8
n_sft_steps=144

# 训练 SFT 模型

python cs336_alignment/my_train_sft.py --train_samples $train_samples --train_batch_size $train_batch_size --micro_batch_size $micro_batch_size --n_sft_steps $n_sft_steps > sft_s${train_samples}_b${train_batch_size}_lr_2e-5_${current_time}.log 2>&1
