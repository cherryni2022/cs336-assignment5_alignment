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
#loss_type="reinforce_with_baseline"
# grpo_learning_rate: 调最优lr
# grpo_baselines: 比较比较有基线 (reinforce_with_baseline) 和无基线 (no_baseline) 的策略梯度方法的性能差异
sub_experiment="grpo_learning_rate"

# 训练 grpo 模型
# 1. 基于作业给的参数, 比较不同learning_rate的效果
#learning_rate=e-5
#learning_rate=3e-5
learning_rate=2e-5
#echo "python cs336_alignment/my_async_train_grpo.py --sub_experiment $sub_experiment --learning_rate $learning_rate --loss_type $loss_type > grpo_${sub_experiment}_${learning_rate}_${current_time}.log 2>&1"
#python cs336_alignment/my_train_grpo.py --sub_experiment $sub_experiment --learning_rate $learning_rate --loss_type $loss_type
#python cs336_alignment/my_train_grpo.py --sub_experiment $sub_experiment --learning_rate $learning_rate --loss_type $loss_type > grpo_${sub_experiment}_${learning_rate}_${current_time}.log 2>&1
#python cs336_alignment/my_async_train_grpo.py --sub_experiment $sub_experiment --learning_rate $learning_rate --loss_type $loss_type > grpo_${sub_experiment}_${learning_rate}_${current_time}.log 2>&1

# 2. 比较loss_type reinforce_with_baseline、no_baseline的策略梯度方法的性能差异
loss_type="no_baseline"
sub_experiment="effect_of_baselining"
echo "python cs336_alignment/my_async_train_grpo.py --sub_experiment $sub_experiment --learning_rate $learning_rate --loss_type $loss_type > grpo_${sub_experiment}_${learning_rate}_${current_time}.log 2>&1"
python cs336_alignment/my_async_train_grpo.py --sub_experiment $sub_experiment --learning_rate $learning_rate --loss_type $loss_type > grpo_${sub_experiment}_${learning_rate}_${current_time}.log 2>&1


# 3. use_std_normalization=False
sub_experiment="effect_of_std_normalization"
use_std_normalization=False
python cs336_alignment/my_async_train_grpo.py --sub_experiment $sub_experiment --learning_rate $learning_rate --loss_type $loss_type --use_std_normalization $use_std_normalization > grpo_${sub_experiment}_${learning_rate}_nostd_${current_time}.log 2>&1


# 4.观察比较 off-policy / on-policy 效果
#sub_experiment="effect_of_on_policy_vs_off_policy"
#loss_type="grpo_clip"
#epochs_per_rollout_batch=3
#echo "python cs336_alignment/my_async_train_grpo.py --sub_experiment $sub_experiment --learning_rate $learning_rate --loss_type $loss_type > grpo_${sub_experiment}_${learning_rate}_${current_time}.log 2>&1"
#python cs336_alignment/my_async_train_grpo.py --sub_experiment $sub_experiment --learning_rate $learning_rate --loss_type $loss_type --epochs_per_rollout_batch $epochs_per_rollout_batch > grpo_${sub_experiment}_${learning_rate}_${current_time}.log 2>&1


