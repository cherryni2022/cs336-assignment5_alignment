#!/bin/bash
# 获取当前时间
current_time=$(date "+%Y-%m-%d_%H:%M:%S")
echo "当前时间: $current_time"
n_grpo_steps=200
learning_rate=1e-5
advantage_eps=1e-6
rollout_batch_size=256
group_size=8
epochs_per_rollout_batch=1 # On-policy
train_batch_size=256
micro_batch_size=8
use_std_normalization=True
#gradient_accumulation_steps = 128 # microbatch size is 2, will fit on H100 gpu_memory_utilization: float = 0.85
loss_type="reinforce_with_baseline"
sub_experiment="adj_lr"


# 训练 grpo 模型
# 1. 基于作业给的参数, 比较不同learning_rate的效果
python cs336_alignment/my_train_grpo.py --sub_experiment $sub_experiment --learning_rate $learning_rate --loss_type $loss_type
#python cs336_alignment/my_train_grpo.py --sub_experiment $sub_experiment --learning_rate $learning_rate --loss_type $loss_type > grpo_lr_${learning_rate}_${current_time}.log 2>&1
