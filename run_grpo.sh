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
# grpo_learning_rate: 调最优lr
# grpo_baselines: 比较比较有基线 (reinforce_with_baseline) 和无基线 (no_baseline) 的策略梯度方法的性能差异


# 训练 grpo 模型
# 1. 基于作业给的参数, 比较不同learning_rate的效果
#learning_rate=e-5
#learning_rate=3e-5
sub_experiment="grpo_learning_rate"
learning_rate=2e-5
loss_type="reinforce_with_baseline"
#echo "python cs336_alignment/my_async_train_grpo.py --sub_experiment $sub_experiment --learning_rate $learning_rate --loss_type $loss_type > grpo_${sub_experiment}_${learning_rate}_${current_time}.log 2>&1"
#python cs336_alignment/my_train_grpo.py --sub_experiment $sub_experiment --learning_rate $learning_rate --loss_type $loss_type
#python cs336_alignment/my_train_grpo.py --sub_experiment $sub_experiment --learning_rate $learning_rate --loss_type $loss_type > grpo_${sub_experiment}_${learning_rate}_${current_time}.log 2>&1
#python cs336_alignment/my_async_train_grpo.py --sub_experiment $sub_experiment --learning_rate $learning_rate --loss_type $loss_type --use_std_normalization $use_std_normalization > grpo_${sub_experiment}_${learning_rate}_${current_time}.log 2>&1

# 2. 比较loss_type reinforce_with_baseline、no_baseline的策略梯度方法的性能差异
sub_experiment="effect_of_baselining"
loss_type="no_baseline"
#echo "python cs336_alignment/my_async_train_grpo.py --sub_experiment $sub_experiment --learning_rate $learning_rate --loss_type $loss_type > grpo_${sub_experiment}_${learning_rate}_${current_time}.log 2>&1"
#python cs336_alignment/my_async_train_grpo.py --sub_experiment $sub_experiment --learning_rate $learning_rate --loss_type $loss_type --use_std_normalization $use_std_normalization > grpo_${sub_experiment}_${learning_rate}_${current_time}.log 2>&1


# 3. grpo_length_normalization
sub_experiment="grpo_length_normalization"
loss_type="reinforce_with_baseline"
#masked_normalize/masked_mean
masked_mean_or_normalize="masked_normalize"
#Compare normalization with masked_mean and masked_normalize
#echo "python cs336_alignment/my_async_train_grpo.py --sub_experiment $sub_experiment --learning_rate $learning_rate --loss_type $loss_type --use_std_normalization $use_std_normalization --masked_mean_or_normalize $masked_mean_or_normalize > grpo_${sub_experiment}_${learning_rate}_${current_time}.log 2>&1"
#python cs336_alignment/my_async_train_grpo.py --sub_experiment $sub_experiment --learning_rate $learning_rate --loss_type $loss_type --use_std_normalization $use_std_normalization --masked_mean_or_normalize $masked_mean_or_normalize > grpo_${sub_experiment}_${learning_rate}_${current_time}.log 2>&1

# 4. use_std_normalization=False
sub_experiment="grpo_group_standard_deviation"
loss_type="reinforce_with_baseline"
use_std_normalization=False
masked_mean_or_normalize="masked_normalize"
# echo "python cs336_alignment/my_async_train_grpo.py --sub_experiment $sub_experiment --learning_rate $learning_rate --loss_type $loss_type --masked_mean_or_normalize $masked_mean_or_normalize > grpo_${sub_experiment}_${learning_rate}_nostd_${current_time}.log 2>&1"
# python cs336_alignment/my_async_train_grpo.py --sub_experiment $sub_experiment --learning_rate $learning_rate --loss_type $loss_type --masked_mean_or_normalize $masked_mean_or_normalize > grpo_${sub_experiment}_${learning_rate}_nostd_${current_time}.log 2>&1


# 5.Off-policy GRPO hyperparameter sweep
sub_experiment="grpo_off_policy"
use_std_normalization=False
masked_mean_or_normalize="masked_normalize"
loss_type="grpo_clip"
epochs_per_rollout_batch=3
echo "python cs336_alignment/my_async_train_grpo.py --sub_experiment $sub_experiment --eval_steps 2 --learning_rate $learning_rate --loss_type $loss_type --epochs_per_rollout_batch $epochs_per_rollout_batch --masked_mean_or_normalize $masked_mean_or_normalize > grpo_${sub_experiment}_${learning_rate}_${current_time}.log 2>&1"
python cs336_alignment/my_async_train_grpo.py --sub_experiment $sub_experiment --learning_rate $learning_rate --eval_steps 2 --loss_type $loss_type --epochs_per_rollout_batch $epochs_per_rollout_batch --masked_mean_or_normalize $masked_mean_or_normalize > grpo_${sub_experiment}_${learning_rate}_${current_time}.log 2>&1

# 6. Off-policy GRPO-Clip ablation(grpo_off_policy_clip_ablation)消融试验
# sub_experiment="grpo_off_policy_clip_ablation"
# use_std_normalization=False
# masked_mean_or_normalize="masked_normalize"
# loss_type="no_clip"
# epochs_per_rollout_batch=3
# echo "python cs336_alignment/my_async_train_grpo.py --sub_experiment $sub_experiment --learning_rate $learning_rate --loss_type $loss_type --epochs_per_rollout_batch $epochs_per_rollout_batch --masked_mean_or_normalize $masked_mean_or_normalize > grpo_${sub_experiment}_${learning_rate}_${current_time}.log 2>&1"
# python cs336_alignment/my_async_train_grpo.py --sub_experiment $sub_experiment --learning_rate $learning_rate --loss_type $loss_type --epochs_per_rollout_batch $epochs_per_rollout_batch --masked_mean_or_normalize $masked_mean_or_normalize > grpo_${sub_experiment}_${learning_rate}_${current_time}.log 2>&1

# 7. prompt 消融试验(grpo_prompt_ablation)
# sub_experiment="grpo_prompt_ablation"
# use_std_normalization=False
# masked_mean_or_normalize="masked_normalize"
# loss_type="reinforce_with_baseline"
# epochs_per_rollout_batch=3
# no_prompt=True
# echo "python cs336_alignment/my_async_train_grpo.py --sub_experiment $sub_experiment --learning_rate $learning_rate --loss_type $loss_type --epochs_per_rollout_batch $epochs_per_rollout_batch --masked_mean_or_normalize $masked_mean_or_normalize > grpo_${sub_experiment}_${learning_rate}_${current_time}.log 2>&1"
# python cs336_alignment/my_async_train_grpo.py --sub_experiment $sub_experiment --learning_rate $learning_rate --loss_type $loss_type --epochs_per_rollout_batch $epochs_per_rollout_batch --masked_mean_or_normalize $masked_mean_or_normalize > grpo_${sub_experiment}_${learning_rate}_${current_time}.log 2>&1