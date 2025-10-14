import logging
import os
from dataclasses import asdict, dataclass
from contextlib import nullcontext
from typing import Callable, List
import dotenv
import fire
import time
import torch
import torch.nn as nn
import wandb
from argparse import ArgumentParser
from torch.utils.data import DataLoader, Dataset
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
logging.getLogger("vllm").setLevel(logging.WARNING)
logging.basicConfig(
    format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


@dataclass
class TrainConfig:
    experiment_name_base: str = "experiments"
    experiment_name: str = "grpo-qwen2.5-math"
    sub_experiment_name: str = "adj_lr"
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
    # type:no_baseline,reinforce_with_baseline,grpo_clip
    loss_type: str = "reinforce_with_baseline"
    mixed_precision_training: bool = True
    # Optimizer
    learning_rate: float = 1e-5
    betas: tuple[float, float] = (0.9, 0.95)

    eval_steps: int = 2
    
    eval_device: str = "cuda:1"
    #eval_device: str = "cuda:1"
    train_device: str = "cuda:0"

    # For VLLM sampling
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 1024
    stop_tokens: list[str] = field(default_factory=lambda: ["</answer>"])
    include_stop_str_in_output: bool = True
    min_tokens: int = 4
    vllm_seed: int = 42

    def __post_init__(self):
        assert self.train_batch_size % self.micro_train_batch_size == 0, "train_batch_size must be divisible by micro_train_batch_size"
        self.gradient_accumulation_steps = self.train_batch_size // self.micro_train_batch_size
        assert self.rollout_batch_size % self.group_size == 0, "rollout_batch_size must be divisible by group_size"
        self.n_prompts_per_rollout_batch = self.rollout_batch_size // self.group_size
        assert self.train_batch_size >= self.group_size, "train_batch_size must be greater than or equal to group_size"
        
        self.n_microbatches_per_rollout_batch = self.rollout_batch_size // self.micro_train_batch_size

        total_train_data_per_grpo_step = (
            self.epochs_per_rollout_batch * self.rollout_batch_size
        )
        self.train_batch_size = self.micro_train_batch_size * self.gradient_accumulation_steps

        self.train_steps_per_rollout_batch = total_train_data_per_grpo_step // self.train_batch_size
        logging.info(f"[trainConfig] after init train_batch_size: {self.train_batch_size},"
                    f"micro_train_batch_size:{self.micro_train_batch_size},"
                    f"gradient_accumulation_steps: {self.gradient_accumulation_steps},"
                    f"rollout_batch_size:{self.rollout_batch_size}, group_size:{self.group_size},"
                    f"n_prompts_per_rollout_batch:{self.n_prompts_per_rollout_batch},"
                    f"epochs_per_rollout_batch:{self.epochs_per_rollout_batch},"
                    f"n_microbatches_per_rollout_batch:{self.n_microbatches_per_rollout_batch},"
                    f"train_steps_per_rollout_batch:{self.train_steps_per_rollout_batch}")

@dataclass
class EvaluateConfig:
    data_path: str = os.path.join(PROJECT_DIR, "data/gsm8k/test.jsonl")
    prompt_path: str = os.path.join(PROJECT_DIR, "cs336_alignment/prompts/r1_zero.prompt")
    temperature: float = 1.0
    top_p: float = 1.0
    #stop_tokens: list[str] = field(default_factory=lambda: ["</answer>"])
    stop_tokens: list[str] = field(default_factory=lambda: ["</answer>"])
    max_tokens: int = 1024
    include_stop_str_in_output: bool = True
    eval_result_dir: str = os.path.join(PROJECT_DIR, "evaluations/grpo")

#原train.jsonl 数据集+prompt_template =>(prompt,cot,true_answer)
class GRPODataset(Dataset):
    def __init__(self, train_prompts, train_cot, train_answers):
        self.train_prompts = train_prompts
        self.train_cot = train_cot
        self.train_answers = train_answers

    def __len__(self):
        return len(self.train_prompts)

    def __getitem__(self, idx: int) -> tuple[str, str, int]:
        prompt = self.train_prompts[idx]
        cot = self.train_cot[idx]
        answer = self.train_answers[idx].strip()
        return prompt, cot, answer

# 计算一次log_probs
def get_old_log_probs(
    model, input_ids, labels, train_config: TrainConfig
) -> tuple[list[list[float]], list[list[float]]]:
    is_training = model.training
    model.eval()

    log_probs = []
    token_entropy = []

    input_ids = input_ids.to(train_config.train_device)
    labels = labels.to(train_config.train_device)

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

        out = get_response_log_probs(
            model=model,
            input_ids=input_ids_part,
            labels=labels_part,
            return_token_entropy=True,
        )

        log_probs.extend(out["log_probs"].tolist())
        token_entropy.extend(out["token_entropy"].tolist())

        del out
        clear()

    assert len(log_probs) == input_ids.shape[0]
    assert len(token_entropy) == input_ids.shape[0]

    model.train(is_training)
    return log_probs, token_entropy

# 每个grpo迭代需要从原完整train_data(prompts,responses) 中采用rollout的样本
# 对于所有样本, 一次性计算其log_probs, token_entropy
class GRPORolloutDataset(Dataset):
    """The rollout dataset for GRPO training.
    input: (prompt, response, raw_reward, advantage)
    """

    def __init__(
        self, model, prompts, responses, raw_rewards, advantages, train_config: TrainConfig, tokenizer
    ):
        print_color("Generate Rollout Dataset...")
        self.raw_rewards = raw_rewards
        self.advantages = advantages

        # Encode prompts and responses
        encoded = tokenize_prompt_and_output(prompts, responses, tokenizer)
        input_ids = encoded["input_ids"]
        labels = encoded["labels"]
        response_mask = encoded["response_mask"]

        self.input_ids = input_ids
        self.labels = labels
        self.response_mask = response_mask

        # We need calculate the old log probs using old model,
        self.old_log_probs, self.token_entropy = get_old_log_probs(model, input_ids, labels, train_config)
        print_color("Generate Rollout Dataset Done.")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> tuple:
        input_ids = self.input_ids[idx]  # Tensor: [1, seq_len]
        labels = self.labels[idx]  # Tensor: [1, seq_len]
        response_mask = self.response_mask[idx]  # Tensor: [1, seq_len]
        raw_reward = self.raw_rewards[idx]
        advantage = self.advantages[idx]
        advantage = advantage.unsqueeze(-1)
        old_log_probs = self.old_log_probs[idx]

        return input_ids, labels, response_mask, raw_reward, advantage, torch.tensor(old_log_probs)

def update_policy(
    model,
    optimizer,
    train_config: TrainConfig,
    prompts,
    responses,
    raw_rewards,
    advantages,
    tokenizer,
    global_train_step,
    grpo_step,
):
    with torch.no_grad():
        dataset = GRPORolloutDataset(
            model,
            prompts,
            responses,
            raw_rewards,
            advantages,
            train_config,
            tokenizer=tokenizer,
        )
    # 每次load train_batch_size 效率更高
    dataloader = DataLoader(dataset=dataset, 
                batch_size=train_config.train_batch_size, 
                shuffle=True)
    cycled_dataloader = cycle_dataloader(dataloader)

    model.train()
    ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if train_config.mixed_precision_training
        else nullcontext()
    )

    global_step_ = global_train_step
    # train_steps_per_rollout_batch = epoches * rollout_batch_size/train_batch_size
    for train_step in range(train_config.train_steps_per_rollout_batch):
        # Fetch the next train_batch_size
        #train_batch = next(cycled_dataloader)
        input_ids, labels, response_mask, raw_rewards, advantages, old_log_probs = next(cycled_dataloader)
        input_ids = input_ids.to(train_config.train_device)
        labels = labels.to(train_config.train_device)
        response_mask = response_mask.to(train_config.train_device)
        old_log_probs = old_log_probs.to(train_config.train_device)
        #old_entropy = entropy.to(train_config.train_device)
        advantages = advantages.to(train_config.train_device)
        raw_rewards = raw_rewards.to(train_config.train_device)
        
        batch_mean_response_length = response_mask.sum(dim=-1).mean(dtype=torch.float32)
        batch_loss = 0
        accumulated_token_entropy = 0
        accumulated_clip_fraction = 0

        for train_microstep in range(train_config.gradient_accumulation_steps):
            start_index = train_microstep * train_config.micro_train_batch_size
            end_index = (train_microstep + 1) * train_config.micro_train_batch_size
            input_ids_micro = input_ids[start_index:end_index]
            labels_micro = labels[start_index:end_index]
            response_mask_micro = response_mask[start_index:end_index]
            advantages_micro = advantages[start_index:end_index]
            raw_rewards_micro = raw_rewards[start_index:end_index]
            old_log_probs_micro = old_log_probs[start_index:end_index]
            with ctx:
                log_probs_dict = get_response_log_probs(model=model, input_ids=input_ids_micro, labels=labels_micro, return_token_entropy=True)
                log_probs = log_probs_dict["log_probs"]
                entropy = log_probs_dict["token_entropy"]
                policy_log_probs = log_probs.to(train_config.train_device)
                loss, metadata = grpo_microbatch_train_step(
                    policy_log_probs,
                    response_mask_micro,
                    train_config.gradient_accumulation_steps,
                    loss_type=train_config.loss_type,
                    raw_rewards=raw_rewards_micro,
                    advantages=advantages_micro,
                    old_log_probs=old_log_probs_micro,
                    cliprange=0.2,
                )

                #mean_advantage = metadata["mean_advantage"]
                #clip_fraction = metadata["clip_fraction"]
                avg_token_entropy = masked_mean(entropy, response_mask_micro, dim=None)
                accumulated_token_entropy += avg_token_entropy.item()
                accumulated_clip_fraction += masked_mean(metadata["clip_fraction"], response_mask_micro, dim=None).item()
                batch_loss += loss.item()

            if train_microstep == train_config.gradient_accumulation_steps - 1:
                # gradient_accumulation_steps 训练完, 更新梯度
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

                logging.info(
                    f"[train grpo] Step result summary at grpo step_{grpo_step}, global train_step_{global_step_ + 1}: "
                    f" | accumulate_batch_loss: {batch_loss:.4f}"
                    f" | avg_loss: {batch_loss / train_config.gradient_accumulation_steps:.4f}"
                    f" | avg_response_entropy: {accumulated_token_entropy / train_config.gradient_accumulation_steps:.4f}"
                    f" | avg_clip_fraction: {accumulated_clip_fraction / train_config.gradient_accumulation_steps:.4f}"
                    f" | response_mask_entropy: {entropy[response_mask].mean().item():.6f}"
                    f" | Global Entropy: {entropy.mean().item():.6f}"
                    f" | Prompt Entropy: {entropy[~response_mask].mean().item():.6f}"
                )
        
                wandb.log({
                        "train/batch_loss": batch_loss,
                        "train/avg_loss": batch_loss / train_config.gradient_accumulation_steps,
                        "train/entropy": to_float(entropy.mean()),
                        "train/response_entropy": to_float(entropy[response_mask].mean()),
                        "train/avg_response_entropy": accumulated_token_entropy / train_config.gradient_accumulation_steps,
                        "train/prompt_entropy": to_float(entropy[~response_mask].mean()),
                        "train/avg_clip_fraction": accumulated_clip_fraction / train_config.gradient_accumulation_steps,
                        "train/grad_norm": grad_norm,
                        "train/mean_response_length": batch_mean_response_length,
                        "train_step": global_step_ + 1
                    }, step=global_step_)

        global_step_ += 1

    return global_step_

def train_grpo(
    train_config: TrainConfig,
    eval_config: EvaluateConfig,
    train_prompts,
    train_cot,
    train_answers,
    vllm: LLM,
):
    date_str = time.strftime("%m%d-%H%M%S")
    policy_type = "onpolicy" if train_config.epochs_per_rollout_batch == 1 else "offpolicy"
    wandb.init(
        entity=os.getenv("WANDB_ENTITY"),
        project="cs336-grpo",
        name=f"train_grpo_lr{train_config.learning_rate}_{train_config.loss_type}_{policy_type}_{date_str}",
        config={
            "n_grpo_steps": train_config.n_grpo_steps,
            "grpo_learning_rate": train_config.learning_rate,
            "grpo_rollout_batch_size": train_config.rollout_batch_size,
            "grpo_group_size": train_config.group_size,
            "grpo_epochs_per_rollout_batch": train_config.epochs_per_rollout_batch,
            "grpo_policy_type": policy_type,
            "grpo_use_std_normalization": train_config.use_std_normalization,
            "grpo_loss_type": train_config.loss_type,
            "grpo_train_batch_size": train_config.train_batch_size,
            "grpo_micro_batch_size": train_config.micro_train_batch_size,
        }
    )
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    grpo_sampling_params = SamplingParams(
        temperature=train_config.temperature,
        top_p=train_config.top_p,
        max_tokens=train_config.max_tokens,
        min_tokens=train_config.min_tokens,
        stop=train_config.stop_tokens,
        include_stop_str_in_output=train_config.include_stop_str_in_output,
        n=train_config.group_size,
        seed=train_config.vllm_seed,
    )
    eval_sp = SamplingParams(
        temperature=eval_config.temperature,
        top_p=eval_config.top_p,
        max_tokens=eval_config.max_tokens,
        stop=eval_config.stop_tokens,
        include_stop_str_in_output=eval_config.include_stop_str_in_output,
    )

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=train_config.local_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cpu",
    ).to(train_config.train_device)
    tokenizer = AutoTokenizer.from_pretrained(train_config.model_name)
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.0,lr=train_config.learning_rate, betas=train_config.betas)
    logging.info(f"[grpo train] Tokenizer {train_config.model_name} loaded from {train_config.local_model_path}")
    logging.info(f"[grpo train] Model {train_config.model_name} loaded on device: {train_config.train_device},"
                f" will eval on device:{train_config.eval_device}"
                f" from {train_config.local_model_path}")
    logging.info("[grpo train] Optimizer loaded")

    # This will return the batch data for grpo step
    base_ds = GRPODataset(train_prompts, train_cot, train_answers)
    base_dl = DataLoader(
        dataset=base_ds,
        batch_size=train_config.n_prompts_per_rollout_batch,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    cycled_dataloader = cycle_dataloader(base_dl)

    global_step = 0
    for grpo_step in range(train_config.n_grpo_steps):
        logging.info(f"[grpo train] start Grpo step_{grpo_step}")
        # Sample a batch of questions from dataset
        sample_batch = next(cycled_dataloader)
        # sample_prompts, sample_cots, sample_answers = zip(*sample_batch)
        sample_prompts, sample_cots, sample_answers = sample_batch
        sample_prompts = list(sample_prompts)
        sample_cots = list(sample_cots)
        sample_answers = list(sample_answers)

        logging.info(f"[grpo train] Grpo step_{grpo_step} sample batch: {len(sample_prompts)}"
                    f", sample_prompts[0]: {sample_prompts[0]}, "
                    f"sample_cots[0]: {sample_cots[0]}, "
                    f"sample_answers[0]: {sample_answers[0]}")

        # (4): Set the old policy
        load_model_into_vllm_instance(model, vllm)

        # (5): Sample G outputs per question.
        logging.info(f"Generating {train_config.group_size} outputs for each rollout samples {len(sample_prompts)}...")
        all_gens = vllm.generate(sample_prompts, grpo_sampling_params)
        all_prompts = []
        all_responses = []
        all_answers = []
        for question, answer, generate_outputs in zip(sample_prompts, sample_answers, all_gens):
            for _, output in enumerate(generate_outputs.outputs):
                all_prompts.append(question)
                all_responses.append(output.text)
                all_answers.append(answer)

        logging.info(f"[grpo train] Generated output for question, total rollout data {len(all_prompts)}:")
        print_rich_dict({"prompt": all_prompts[0], "responses": all_responses[0], "answers": all_answers[0]})

        # (6) / (7): Compute rewards for each sampled output
        advantages_train, raw_rewards_train, metadata = compute_group_normalized_rewards(
            r1_zero_reward_fn,
            rollout_responses=all_responses,
            repeated_ground_truths=all_answers,
            group_size=train_config.group_size,
            advantage_eps=train_config.advantage_eps,
            normalized_by_std=train_config.use_std_normalization,
        )


        global_step = update_policy(
            model=model,
            optimizer=optimizer,
            train_config=train_config,
            prompts=all_prompts,
            responses=all_responses,
            raw_rewards=raw_rewards_train,
            advantages=advantages_train,
            tokenizer=tokenizer,
            global_train_step=global_step,
            grpo_step=grpo_step,
        )

        # Evaluate
        if (grpo_step + 1) % train_config.eval_steps == 0:
            load_model_into_vllm_instance(model, vllm)
            # prompts, cot, answers = load_and_format_prompts(eval_config.data_path, eval_config.prompt_path)
            # results = evaluate_vllm(
            #     vllm_model=vllm,
            #     reward_fn=r1_zero_reward_fn,
            #     prompts=prompts,
            #     answers=answers,
            #     eval_sampling_params=eval_sp,
            # )
            evaluate_sft_model(eval_config, vllm, eval_step=grpo_step)
            
            # wandb.log({
            #     "eval/correct": results["correct"],
            #     "eval/count": results["count"],
            #     "eval/answer_wrong": results["answer_wrong"],
            #     "eval/format_wrong": results["format_wrong"],
            #     "eval/accuracy": results["accuracy"],
            #     "eval/format_wrong_rate": results["format_wrong_rate"],
            #     "eval/answer_wrong_rate": results["answer_wrong_rate"],
            #     "eval/avg_response_length": results["avg_response_length"],
            #     "eval/avg_correct_response_length": results["avg_correct_response_length"],
            #     "eval/avg_incorrect_response_length": results["avg_incorrect_response_length"],
            #     "eval_step": eval_step,}
            # )

            save_model_and_tokenizer(model, tokenizer, train_config, train_config.sub_experiment_name)

        save_model_and_tokenizer(model, tokenizer, train_config, train_config.sub_experiment_name)

    print("Grpo Training Finished")

def main(
    *,
    n_grpo_steps: int,
    rollout_batch_size: int,
    group_size: int,
    epochs_per_rollout_batch: int,
    learning_rate: float,
    use_std_normalization: bool,
    loss_type: str,
    train_batch_size:int ,
    micro_batch_size: int,
    sub_experiment_name: str,
    model_name: str = "Qwen/Qwen2.5-Math-1.5B",
    local_model_path: str = os.path.join(PROJECT_DIR, "models/Qwen2.5-Math-1.5B-Base"),
    data_path: str = os.path.join(PROJECT_DIR, "data/gsm8k/train.jsonl"),
    prompt_path: str = os.path.join(PROJECT_DIR, "cs336_alignment/prompts/r1_zero.prompt"),
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_tokens: int = 1024,
    seed: int = 123,
):
    dotenv.load_dotenv()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HOME"] = "~/autodl-tmp/hf"
    os.environ["TRANSFORMERS_CACHE"] = "~/autodl-tmp/hf/models"
    os.environ["HF_HUB_CACHE"] = "~/autodl-tmp/hf/hub"

    # Login Wandb
    api_key = os.getenv("WANDB_API_KEY")
    wandb.login(key=api_key)
    logging.info(f"init wandb.login with api_key:{api_key}")

    train_config = TrainConfig()
    eval_config = EvaluateConfig()

    # Set train config
    train_config.n_grpo_steps = n_grpo_steps
    train_config.rollout_batch_size = rollout_batch_size
    train_config.group_size = group_size
    train_config.n_prompts_per_rollout_batch = rollout_batch_size // group_size
    train_config.epochs_per_rollout_batch = epochs_per_rollout_batch
    assert train_config.train_batch_size >= train_config.group_size, "train_batch_size must be greater than or equal to group_size"
    # TODO: 某些参数需要计算
    train_config.use_std_normalization = use_std_normalization
    train_config.learning_rate = learning_rate
    train_config.loss_type = loss_type
    train_config.train_batch_size = train_batch_size
    train_config.micro_train_batch_size = micro_batch_size
    train_config.gradient_accumulation_steps = train_config.train_batch_size // train_config.micro_train_batch_size
    
    train_config.n_microbatches_per_rollout_batch = train_config.rollout_batch_size // train_config.micro_train_batch_size

    total_train_data_per_grpo_step = (
        train_config.epochs_per_rollout_batch * train_config.rollout_batch_size
    )
    train_config.train_steps_per_rollout_batch = total_train_data_per_grpo_step // train_config.train_batch_size

    print_rich_dict({"train config": asdict(train_config), 
                    "eval config": asdict(eval_config)})

    vllm = init_vllm(model_id=local_model_path, device=train_config.eval_device, seed=seed)
    prompts, cot, answers = load_and_format_prompts(train_config.data_path, train_config.prompt_path)

    train_grpo(
        train_config=train_config,
        eval_config=eval_config,
        train_prompts=prompts,
        train_cot=cot,
        train_answers=answers,
        vllm=vllm,
    )

    wandb.finish()

# 实验控制变量:
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--sub_experiment_name", type=str, default="adj_lr", help="sub_experiment_name")
    parser.add_argument("--n_grpo_steps", type=int, default=200, help="n_grpo_steps")
    parser.add_argument("--rollout_batch_size", type=int, default=256, help="total rollout samples per grpo step")
    parser.add_argument("--group_size", type=int, default=8, help="each grpo step sample group_size output per question")
    parser.add_argument("--epochs_per_rollout_batch", type=int, default=1, help="epochs_per_rollout_batch")
    parser.add_argument("--use_std_normalization", type=bool, default=True, help="use_std_normalization")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="learning_rate")
    # type:no_baseline,reinforce_with_baseline,grpo_clip
    parser.add_argument("--loss_type", type=str, default="reinforce_with_baseline", help="loss_type in no_baseline,grpo_clip,reinforce_with_baseline")
    parser.add_argument("--train_batch_size", type=int, default=256, help="train_batch_size")
    parser.add_argument("--micro_batch_size", type=int, default=8, help="micro_batch_size")
    
    args = parser.parse_args()
    logging.info(f"start train grpo with params: {args}")
    fire.Fire(main(n_grpo_steps=args.n_grpo_steps,
                    rollout_batch_size=args.rollout_batch_size,
                    group_size=args.group_size,
                    epochs_per_rollout_batch=args.epochs_per_rollout_batch,
                    learning_rate=args.learning_rate,
                    use_std_normalization=args.use_std_normalization,
                    loss_type=args.loss_type,
                    train_batch_size=args.train_batch_size,
                    micro_batch_size=args.micro_batch_size,
                    sub_experiment_name=args.sub_experiment_name,
                    ))