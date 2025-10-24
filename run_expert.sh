#!/bin/bash
# 获取当前时间
current_time=$(date "+%Y-%m-%d_%H:%M:%S")
echo "当前时间: $current_time"

train_batch_size=256
micro_batch_size=8
sample_questions_per_ei_step=512
rollout_per_prompt=4
sft_train_epochs=32
sft_train_epochs=4
# 训练 SFT 模型

#python cs336_alignment/my_train_sft.py --train_samples $train_samples --train_batch_size $train_batch_size --micro_batch_size $micro_batch_size --n_sft_steps $n_sft_steps > sft_s${train_samples}_b${train_batch_size}_lr_2e-5_${current_time}.log 2>&1
echo "python cs336_alignment/my_async_train_expert_iter.py --use_correct $use_correct --train_samples $train_samples --train_batch_size $train_batch_size --micro_batch_size $micro_batch_size --n_sft_steps $n_sft_steps > sft_s${train_samples}_b${train_batch_size}_lr2e-5_${current_time}.log 2>&1"
python cs336_alignment/my_async_train_expert_iter.py  --sample_questions_per_ei_step $sample_questions_per_ei_step --rollout_per_prompt $rollout_per_prompt --sft_train_epochs $sft_train_epochs > ei_s${sample_questions_per_ei_step}_g${rollout_per_prompt}_e$sft_train_epochs_lr2e-5_${current_time}.log 2>&1 &