import logging
import os
from dataclasses import asdict, dataclass
from contextlib import nullcontext
from typing import Callable, List
from concurrent.futures import ThreadPoolExecutor, Future
import dotenv
import fire
import time
import torch
import torch.nn as nn
import wandb
from argparse import ArgumentParser
from torch.utils.data import DataLoader, Dataset
import multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
import gc
from dataclasses import asdict, dataclass, field
from cs336_alignment.data_utils import load_and_format_prompts
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.grpo import compute_group_normalized_rewards, grpo_microbatch_train_step, masked_mean
from cs336_alignment.sft_utils import get_response_log_probs, tokenize_prompt_and_output
from cs336_alignment.utils import (
    clear,
    cycle_dataloader,
    get_run_name,
    print_color,
    print_rich_dict,
    save_model_and_tokenizer,
)
from cs336_alignment.vllm_utils import init_vllm, load_model_into_vllm_instance
from cs336_alignment.my_train_sft import evaluate_sft_model, evaluate_vllm, log_generations, sft_collate_fn, to_float

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(
    format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

@dataclass
class TrainConfig:
    experiment_name_base: str = "experiments"
    experiment_name: str = "grpo-qwen2.5-math"
    sub_experiment_name: str = "grpo_learning_rate"
    model_name: str = "Qwen/Qwen2.5-Math-1.5B"
    local_model_path: str = os.path.join(PROJECT_DIR, "models/Qwen2.5-Math-1.5B-Base")
    data_path: str = os.path.join(PROJECT_DIR, "data/gsm8k/train.jsonl")
    prompt_path: str = os.path.join(PROJECT_DIR, "cs336_alignment/prompts/r1_zero.prompt")

    # assignment 试验给出的参数 ------------------
    #group_size: int = 8
    #epochs_per_rollout_batch: int = 1 # On-policy
    
    #---------------------------------------
    
    # GRPO
    # test n_grpo_steps: int = 10
    # grpo算法外循环
    n_grpo_steps: int = 200
    rollout_batch_size: int = 256 # 每个grpo迭代,抽样并rollout后构建用于train的总样本数
    # n_prompts_per_rollout_batch = rollout_batch_size // group_size
    n_prompts_per_rollout_batch: int = 32
    group_size: int = 8
    # epochs_per_rollout_batch > 1 off-policy
    epochs_per_rollout_batch: int = 1 # On-policy
    # grpo算法内循环:针对每个grop迭代 rollout_batch的样本, train的steps数
    # train_steps_per_rollout_batch = rollout_batch_size // train_batch_size
    train_steps_per_rollout_batch: int = 4

    train_batch_size: int = 256
    gradient_accumulation_steps: int = 32
    micro_train_batch_size: int = 8 # train_batch_size/gradient_accumulation_steps

    advantage_eps: float = 1e-6
    use_std_normalization: bool = True
    masked_mean_or_normalize: str = "masked_mean" # masked_mean/masked_normalize
    # type:no_baseline,reinforce_with_baseline,grpo_clip
    loss_type: str = "reinforce_with_baseline"
    mixed_precision_training: bool = True
    # Optimizer
    learning_rate: float = 1e-5
    betas: tuple[float, float] = (0.9, 0.95)

    eval_steps: int = 2
    
    rollout_device: str = "cuda:0"
    eval_device: str = "cuda:1"
    train_device: str = "cuda:0"
    rollout_gpu_mem_util: float = 0.45
    eval_gpu_mem_util: float = 0.45

    # For VLLM sampling
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 1024
    stop_tokens: list[str] = field(default_factory=lambda: ["</answer>"])
    include_stop_str_in_output: bool = True
    min_tokens: int = 4
    vllm_seed: int = 42
    #vllm_seed: int = 43

# 计算一次log_probs
def get_old_log_probs(
    model, input_ids, labels, 
    train_config
) -> tuple[list[list[float]], list[list[float]]]:
    logging.info(f"[get_old_log_probs] 当前 model.training={model.training}")
    is_training = model.training
    model.eval()
    logging.info(f"[get_old_log_probs] 当前 model.training={model.training}")
    logging.info(f"[get_old_log_probs] 计算 log_probs 开始")
    log_probs = []
    token_entropy = []

    input_ids = input_ids.to(train_config.train_device)
    labels = labels.to(train_config.train_device)
    with torch.no_grad():
        # group_size 一组question 计算
        for train_step in range(0, train_config.n_prompts_per_rollout_batch):
            start_index = train_step * train_config.group_size
            input_ids_part = input_ids[
                start_index : start_index + train_config.group_size,
                :,
            ]
            labels_part = labels[
                start_index : start_index + train_config.group_size,
                :,
            ]
            start_compute = time.time()
            out = get_response_log_probs(
                model=model,
                input_ids=input_ids_part,
                labels=labels_part,
                return_token_entropy=True,
            )

            # Accumulate tensors directly; avoid per-iteration GPU->CPU sync
            log_probs.append(out["log_probs"])  # tensor of shape (group_size, seq_len) or similar
            token_entropy.append(out["token_entropy"])  # tensor
            compute_time = time.time() - start_compute
            logging.info(f"[get_old_log_probs] 计算 train_step_{train_step} log_probs 耗时: {compute_time*1000:.2f}ms")

            # logging.info(f"[get_old_log_probs] 计算 train_step_{train_step} clear 耗时: {clear_time*1000:.2f}ms")

    clear()
    # After loop, concatenate tensors and convert to lists once
    log_probs = torch.cat(log_probs, dim=0).tolist()
    token_entropy = torch.cat(token_entropy, dim=0).tolist()
    
    assert len(log_probs) == input_ids.shape[0]
    assert len(token_entropy) == input_ids.shape[0]

    model.train(is_training)
    return log_probs, token_entropy


def test_log_probs():
    train_config = TrainConfig()
    train_sample_num = 256
    prompts, cot, answers = load_and_format_prompts(train_config.data_path, train_config.prompt_path)
    train_prompts = prompts[:train_sample_num]
    train_cot = cot[:train_sample_num]
    train_answers = answers[:train_sample_num]

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=train_config.local_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cpu",
    ).to(train_config.train_device)

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=train_config.local_model_path,
    )

    # Encode prompts and responses
    encoded = tokenize_prompt_and_output(train_prompts, train_cot, tokenizer)
    input_ids = encoded["input_ids"]
    labels = encoded["labels"]
    response_mask = encoded["response_mask"]
    start = time.time()
    old_log_probs, token_entropy = get_old_log_probs(
        model, input_ids, labels, train_config
    )
    elapsed = time.time() - start
    logging.info(f"[GRPORolloutDataset] generate Rollout Dataset Done, "
                 f"input_ids.len={len(input_ids)},"
                 f"labels.len={len(labels)},"
                 f"response_mask.len={len(response_mask)},"
                 f"old_log_probs.len={len(old_log_probs)},"
                 f"token_entropy.len={len(token_entropy)},"
                 f"耗时={elapsed*1000:.2f}ms")

def test_compute_group_normalized_rewards():
    train_config = TrainConfig()
    train_sample_num = 32
    prompts, cot, answers = load_and_format_prompts(train_config.data_path, train_config.prompt_path)
    sample_prompts = prompts[:train_sample_num]
    sample_cot = cot[:train_sample_num]
    sample_answers = answers[:train_sample_num]
    # rollout_vllm = init_vllm(model_id=train_config.local_model_path, 
    #                  device=train_config.rollout_device, 
    #                  seed=132,
    #                  gpu_memory_utilization=0.85)

    # grpo_sampling_params = SamplingParams(
    #     temperature=train_config.temperature,
    #     top_p=train_config.top_p,
    #     max_tokens=train_config.max_tokens,
    #     min_tokens=train_config.min_tokens,
    #     stop=train_config.stop_tokens,
    #     include_stop_str_in_output=train_config.include_stop_str_in_output,
    #     n=train_config.group_size,
    #     seed=train_config.vllm_seed,
    # )

    # all_gens = rollout_vllm.generate(sample_prompts, grpo_sampling_params)
    all_prompts = []
    all_responses = []
    all_answers = []
    for question, answer, cot in zip(sample_prompts, sample_answers, sample_cot):
        for x in range(train_config.group_size):
            all_prompts.append(question)
            all_responses.append(cot)
            all_answers.append(answer)

    advantages, raw_rewards, metadata = compute_group_normalized_rewards(
            r1_zero_reward_fn,
            rollout_responses=all_responses,
            repeated_ground_truths=all_answers,
            group_size=train_config.group_size,
            advantage_eps=train_config.advantage_eps,
            normalized_by_std=train_config.use_std_normalization,
    )
    print(f"[compute_group_normalized_rewards test] advantage.shape:, {advantages.shape},"
        f", raw_rewards.shape: {raw_rewards.shape}")

if __name__ == "__main__":
    #test_log_probs()
    test_compute_group_normalized_rewards()
