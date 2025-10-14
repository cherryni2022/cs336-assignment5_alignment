import logging
import math
import os
from contextlib import nullcontext
from dataclasses import asdict, dataclass
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
#from cs336_alignment.utils import print_color, print_rich_dict, save_model_and_tokenizer
from cs336_alignment.data_utils import extract_reference_answer, load_and_format_prompts
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.evaluate import get_vllm_response
from cs336_alignment.sft_utils import (
    get_response_log_probs,
    sft_microbatch_train_step,
    tokenize_prompt_and_output,
)
from cs336_alignment.utils import (
    get_run_name,
    print_rich_dict,
    print_color,
    save_model_and_tokenizer,
)
from cs336_alignment.vllm_utils import init_vllm, load_model_into_vllm_instance
from cs336_alignment.my_train_sft import  to_float, evaluate_sft_model, log_generations, sft_collate_fn
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
logging.getLogger("vllm").setLevel(logging.WARNING)
logging.basicConfig(
    format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# 实验要求: 
# 1. 改变每个问题的 rollout 数量 G 
# 2. 改变SFT 步骤中使用的 epoch 数量，并设置 n_ei_steps = 5
# 3. 在 {512, 1024, 2048} 中改变每个专家迭代步骤的批量大小（即 Db 的大小）

def clear():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

def cycle_dataloader(dataloader):
    """
    Creates a cycling iterator for a PyTorch DataLoader.
    """
    while True:
        for batch in dataloader:
            yield batch

def get_lr(it, max_lr, max_steps):
    min_lr = max_lr * 0.1
    if it >= max_steps:
        return min_lr
    # pure cosine decay
    decay_ratio = it / max_steps
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

@dataclass
class TrainEIConfig:
    experiment_name_base: str = "experiments"
    experiment_name: str = "sft-expert-iteration-qwen2.5-math"
    model_name: str = "Qwen/Qwen2.5-Math-1.5B"
    local_model_path: str = os.path.join(PROJECT_DIR, "models/Qwen2.5-Math-1.5B-Base")
    train_data_path: str = os.path.join(PROJECT_DIR, "data/gsm8k/train.jsonl")
    prompt_path: str = os.path.join(PROJECT_DIR, "cs336_alignment/prompts/r1_zero.prompt")

    # 控制每轮ei 的sft train 的参数变量
    sft_train_epochs: int = 8
    # sft_training_steps 每次基于sft_train_epochs计算
    # sft_training_steps: int = 8 
    sft_train_batch_size: int = 256
    gradient_accumulation_steps: int = 32
    micro_batch_size: int = 8
    mixed_precision_training: bool = True
    learning_rate: float = 1e-5 # 2e-5
    betas: tuple[float, float] = (0.9, 0.98)
    train_device: str = "cuda:0"

    # Expert Iteration hyper-parameters
    n_ei_steps: int = 5  # outer EI steps (replaces training_steps)
    # 每个专家迭代步骤的批量大小（即 Db 的大小）[512, 1024, 2048]
    sample_questions_per_ei_step: int = 512
    rollout_per_prompt: int = 4  # 每个问题的 rollout 数量 G in Algorithm 2

    # Logging / eval
    log_print_steps: int = 12

    eval_device: str = "cuda:1"
    eval_steps: int = 16

    # Learning Rate adjust
    lr_mode: str = "global"  # one of: "global", "per_outer", "constant"
    warmup_ratio: float = 0.03  # 0 = no warmup
    min_lr_factor: float = 0.1  # min_lr = lr * min_lr_factor
    scale_lr_by_pairs: bool = True  # scale lr by 1/sqrt(num_kept_pairs) per EI step

    # For Expert Iteration sampling
    ei_temperature: float = 1.0
    ei_top_p: float = 1.0
    ei_max_tokens: int = 1024
    ei_stop_tokens: list[str] = field(default_factory=lambda: ["</answer>"])
    ei_include_stop_str_in_output: bool = True
    ei_min_tokens: int = 4

@dataclass
class EvaluateConfig:
    data_path: str = os.path.join(PROJECT_DIR, "data/gsm8k/test.jsonl")
    prompt_path: str = os.path.join(PROJECT_DIR, "cs336_alignment/prompts/r1_zero.prompt")
    temperature: float = 1.0
    top_p: float = 1.0
    stop_tokens: list[str] = field(default_factory=lambda: ["</answer>"])
    max_tokens: int = 1024
    include_stop_str_in_output: bool = True
    eval_result_dir: str = os.path.join(PROJECT_DIR, "evaluations/expert")

# train.jsonl全量数据集,用于每轮专家迭代步骤中采样Db question
class ExpertTrainDataSet(Dataset):
    def __init__(self, data_path: str, prompt_path: str):
        self.data_path = data_path
        self.prompt_path = prompt_path
        self.train_prompts, self.train_cot, self.train_answers = load_and_format_prompts(data_path, prompt_path)
        logging.info(f"[ExpertTrainDataSet] Loaded {len(self.train_prompts)} samples from {data_path}")

    def __len__(self):
        return len(self.train_prompts)

    def __getitem__(self, index):
        prompt = self.train_prompts[index]
        cot = self.train_cot[index]
        answer = self.train_answers[index]
        return prompt, cot, answer

class SFTDataset(Dataset):
    def __init__(self, train_prompts, train_cot, train_answers):
        self.train_prompts = train_prompts
        self.train_cot = train_cot
        self.train_answers = train_answers

    def __len__(self):
        return len(self.train_prompts)

    def __getitem__(self, idx: int) -> tuple[str, str, float]:
        prompt = self.train_prompts[idx]
        cot = self.train_cot[idx]
        answer = to_float(self.train_answers[idx])

        return prompt, cot, answer

# 用于每次专家迭代步骤中，收集所有 reward==1 的 (prompt, output) 对
@torch.no_grad()
def ei_collect_correct_samples(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    answers: List[str],
    train_config: TrainEIConfig,
) -> tuple[list[str], list[str]]:
    """Return (correct_prompts, correct_outputs) where reward==1."""
    correct_prompts: list[str] = []
    correct_outputs: list[str] = []
    correct_answers: list[str] = []

    # Ensure we sample multiple outputs per prompt.
    ei_sampling_params = SamplingParams(
        temperature=train_config.ei_temperature,
        top_p=train_config.ei_top_p,
        max_tokens=train_config.ei_max_tokens,
        stop=train_config.ei_stop_tokens,
        include_stop_str_in_output=train_config.ei_include_stop_str_in_output,
        min_tokens=train_config.ei_min_tokens,
        n=train_config.rollout_per_prompt,
    )

    all_responses = vllm_model.generate(prompts, ei_sampling_params)
    for question, answer, responses in zip(prompts, answers, all_responses):
        for _, response in enumerate(responses.outputs):
            # compute reward
            result = reward_fn(response.text, str(answer))
            if result.get("reward", 0) == 1:
                correct_prompts.append(question)
                correct_outputs.append(response.text)
                correct_answers.append(str(answer))

    return correct_prompts, correct_outputs, correct_answers

# 针对超参数配置sft训练模型
def train_sft_model(
    model,
    optimizer,
    tokenizer,
    vllm,
    train_config: TrainEIConfig,
    train_prompts,
    train_cot,
    train_answers,
    global_sft_step: int = 0,  #全局sft训练的steps累积轮次
    curr_ei_steps: int = 0,  # ei训练的当前step轮次
    ):

    # step1: 准备训练数据
    train_dataset = SFTDataset(train_prompts, train_cot, train_answers)
    total_train_ds_samples = len(train_dataset)
    train_dataloader = DataLoader(
        dataset = train_dataset,
        batch_size = train_config.micro_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=lambda batch: sft_collate_fn(batch, tokenizer),
    )
    sft_data_loader = cycle_dataloader(train_dataloader)
    logging.info(f"[trainSFT | ei step_{curr_ei_steps}] Dataloader initialized with batch size {train_config.sft_train_batch_size},"
          f" micro batch size: {train_config.micro_batch_size}")
    # step3:准备模型训练的optimazer等
    # ---------------------
    # Mixed Precision Context
    # ---------------------
    ctx = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if train_config.mixed_precision_training
        else nullcontext()
    )

    # step4:训练模型
    total_loss = 0 # 累计每个sft step的loss
    n_sft_steps = total_train_ds_samples * train_config.sft_train_epochs // (train_config.gradient_accumulation_steps * train_config.micro_batch_size) + 1
    logging.info(f"[trainSFT | ei step_{curr_ei_steps}] Start sft training, "
                 f"from global sft step_{global_sft_step}, will run sft steps: {n_sft_steps},"
                 f"total_train_ds_samples: {total_train_ds_samples},"
                 f"sft_train_epochs: {train_config.sft_train_epochs},")
    for sft_step in range(n_sft_steps):
        batch_loss = 0
        #logging.info(f"[trainSFT | ei step_{curr_ei_steps}] sft global step_{global_sft_step}, sft local step_{sft_step}")
        for curr_grad_accum_step in range(train_config.gradient_accumulation_steps):
            # 从dataloader中获取一个micro batch的训练数据
            batch = next(iter(sft_data_loader))
            # sft训练中 ["answers"] 没用,eval时用
            input_ids = batch["input_ids"].to(train_config.train_device)
            labels = batch["labels"].to(train_config.train_device)
            response_mask = batch["response_mask"].to(train_config.train_device)

            with ctx:
                log_prob_response = get_response_log_probs(model=model, 
                                    input_ids=input_ids, labels=labels, 
                                    response_mask=response_mask,
                                    return_token_entropy=True)
                log_prob = log_prob_response["log_probs"]
                entropy = log_prob_response["token_entropy"]
                # 计算loss & backward()
                loss, _ = sft_microbatch_train_step(
                    log_prob, response_mask, train_config.gradient_accumulation_steps
                )
                logging.info(f"[train test log] ei step_{curr_ei_steps}] sft global step_{global_sft_step} |"
                        f"local sft step_{sft_step} | "
                        f"Gradient accumulation step_{curr_grad_accum_step} | "
                        f"response_mask_entropy: {entropy[response_mask].mean().item():.6f} |"
                        f"Global Entropy: {entropy.mean().item():.6f} |"
                        f"Prompt Entropy: {entropy[~response_mask].mean().item():.6f} |"
                        f"Loss: {loss.item():.4f}")

            batch_loss += loss

            # TODO 
            del input_ids
            del labels
            clear()

            # 累积梯度更新参数
            if curr_grad_accum_step == train_config.gradient_accumulation_steps - 1:
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                # TODO 是否考虑 ei_step, sft_step
                #adj_lr = get_lr(sft_step, train_config.learning_rate, train_config.training_steps)
                adj_lr = get_lr(curr_ei_steps, train_config.learning_rate, train_config.n_ei_steps)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = adj_lr
                # 更新参数
                optimizer.step()
                optimizer.zero_grad()
                avg_loss = batch_loss / train_config.gradient_accumulation_steps
                total_loss += avg_loss

                logging.info(
                    f"[trainSFT | ei step_{curr_ei_steps} sft_global step_{global_sft_step+1}]: "
                    f" | local sft step_{sft_step}"
                    f" | avg_loss: {avg_loss:.4f}"
                    f" | response_mask_entropy: {entropy[response_mask].mean().item():.6f}"
                    f" | Global Entropy: {entropy.mean().item():.6f}"
                    f" | Prompt Entropy: {entropy[~response_mask].mean().item():.6f}"
                    f" | adj_lr: {adj_lr:.6f}"
                )

                wandb.log(
                    {
                        "train_step": global_sft_step+1,
                        #"train/ei_step": curr_ei_steps,
                        "train/loss": avg_loss,
                        "train/response_entropy": to_float(entropy[response_mask].mean()),
                        "train/entropy": to_float(entropy.mean()),
                        "train/prompt_entropy": to_float(entropy[~response_mask].mean()),
                        "train/lr": adj_lr,
                    }
                )

        global_sft_step += 1

        if (global_sft_step % train_config.eval_steps == 0):
            load_model_into_vllm_instance(model, vllm)
            # evaluate model & wandb.log 信息
            evaluate_sft_model(eval_config, vllm, global_sft_step)
            save_model_and_tokenizer(model, tokenizer, train_config, f"ei_rollout_samples_{train_config.sample_questions_per_ei_step}")

        

    return total_loss / train_config.training_steps, global_sft_step

# 实验要求:
# 1.改变每个问题的 rollout 数量 G 和 SFT 步骤中使用的 epoch 数量:
#  交付:与不同 rollout 配置相关的验证准确率曲线。至少尝试 2 种不同的 rollout 数量和 epoch 数量
# 2.改变每个专家迭代步骤的批量大小（即 Db 的大小）: 512, 1024, 2048
# 控制变量:
# train_config.sample_questions_per_ei_step (每个专家迭代步骤的批量大小（即 Db 的大小))
# train_config.rollout_per_prompt (每个问题的 rollout 数量 G)
# train_config.training_steps (SFT 步骤中使用的 epoch 数量)
# 不变: rollout:4, training_steps=8 => sample_questions_per_ei_step [512, 1024, 2048]
# 不变: sample_questions_per_ei_step=512,train_steps=8 => rollout=4, rollout = 2, rollout=8
# 不变: sample_questions_per_ei_step=512, rollout=4 => training_steps=8, train_steps=16

def train_ei_model(
    train_config: TrainEIConfig,
    eval_config: EvaluateConfig,
    vllm: LLM,
):
    # step1: init wandb
    date_str = time.strftime("%m%d-%H%M%S")
    lr = train_config.learning_rate
    wandb.init(
        entity=os.getenv("WANDB_ENTITY"),
        project="cs336-alignment-ei",
        name=f"train_ei_{train_config.sample_questions_per_ei_step}_{train_config.rollout_per_prompt}_{train_config.sft_train_epochs}_{lr}_{date_str}",
        config = {
            "sample_questions_per_ei_step": train_config.sample_questions_per_ei_step,
            "rollout_per_prompt": train_config.rollout_per_prompt,
            "sft_train_epochs": train_config.sft_train_epochs, 
            "sft_train_batch_size": train_config.sft_train_batch_size,
            "micro_batch_size": train_config.micro_batch_size,
            "gradient_accumulation_steps": train_config.gradient_accumulation_steps,
        }
    )

    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    # step2: load model
    # ---------------------
    # Load Model and tokenizer
    # ---------------------
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=train_config.local_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cpu",
    ).to(train_config.train_device)
    tokenizer = AutoTokenizer.from_pretrained(train_config.local_model_path)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate, betas=train_config.betas)
    print(f"[EI train] Tokenizer {train_config.model_name} loaded from {train_config.local_model_path}")
    print(f"[EI train] Model {train_config.model_name} loaded on device: {train_config.train_device},"
            f" will eval on device:{train_config.eval_device}"
            f", from {train_config.local_model_path}")
    print(f"[EI train] Optimizer loaded, with learning rate: {train_config.learning_rate}")

    eval_sample_params = SamplingParams(
        temperature=eval_config.temperature,
        top_p=eval_config.top_p,
        max_tokens=eval_config.max_tokens,
        stop=eval_config.stop_tokens,
        include_stop_str_in_output=eval_config.include_stop_str_in_output,
    )

    # prepare data loader
    base_ds = ExpertTrainDataSet(train_config.train_data_path, train_config.prompt_path)
    base_data_loader = DataLoader(
        dataset=base_ds,
        batch_size=train_config.sample_questions_per_ei_step,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    ei_train_dataloader = cycle_dataloader(base_data_loader)
    global_step = 0
    # Expert_Iterator Training loop
    for ei_step in range(train_config.n_ei_steps):
        # Sample a batch of questions Db from D
        batch_samples = next(iter(ei_train_dataloader))
        batch_prompts = batch_samples[0]
        batch_answers = batch_samples[2]
        print(f"[EI train] Step)_{ei_step}: Sample {len(batch_prompts)} questions from D")

        # 每轮ei迭代训练完已经load model到vllm
        
        # Sample G outputs per question, compute rewards, filter to correct pairs
        correct_prompts, correct_outputs, correct_answers = ei_collect_correct_samples(
            vllm_model=vllm,
            reward_fn=r1_zero_reward_fn,
            prompts = batch_prompts,
            answers = batch_answers,
            train_config=train_config,
        )

        if len(correct_prompts) == 0:
            logging.info(f"[EI train] Step {ei_step}: no correct generations; skipping SFT update.")
            continue
        
        # print 采样的数据样例
        print_color(f"[EI train] Step_{ei_step}: Collect {len(correct_prompts)} correct sample output data")
        print_color(f"[EI train] Example correct sample data:"
                    f"correct_prompts: {correct_prompts[0]}"
                    f"correct_outputs: {correct_outputs[0]}"
                    f"correct_answers: {correct_answers[0]}")
        # sft train
        loss, global_step = train_sft_model(
            model = model,
            tokenizer = tokenizer,
            optimizer = optimizer,
            vllm = vllm,
            train_config = train_config,
            train_prompts = correct_prompts,
            train_cot = correct_outputs,
            train_answers = correct_answers,  # not used in SFT
            global_step = global_step,
            ei_steps = ei_step,
        )

        print_color(f"[EI train] Step_{ei_step} | Correct samples: {len(correct_prompts)} | Globel sft step: {global_step} Loss: {loss:.4f}", color="green")

        logging.info(f"Loaded weights to vllm at step_{ei_step}")
        # 一次sft训练结束, eval_model= model, 后一次ei_step 将利用eval_model对抽样数据采集outputs
        load_model_into_vllm_instance(model, vllm)
    
    
    save_model_and_tokenizer(model, tokenizer, train_config, f"ei_rollout_samples_{train_config.sample_questions_per_ei_step}")
    wandb.finish()

    return model

# 控制变量:
# train_config.sample_questions_per_ei_step (每个专家迭代步骤的批量大小（即 Db 的大小))
# train_config.rollout_per_prompt (每个问题的 rollout 数量 G)
# train_config.training_steps (SFT 步骤中使用的 epoch 数量)
# 不变: rollout:4, training_steps=8 => sample_questions_per_ei_step [512, 1024, 2048]
# 不变: sample_questions_per_ei_step=512,train_steps=8 => rollout=4, rollout = 2, rollout=8
# 不变: sample_questions_per_ei_step=512, rollout=4 => training_steps=8, train_steps=16
def main(*,
    sample_questions_per_ei_step_params: list[int] = [512],
    rollout_per_prompt: int = 4,
    sft_train_epochs: int = 8,
    sft_train_batch_size: int = 256,
    micro_batch_size: int = 8,
    model_name: str = "Qwen/Qwen2.5-Math-1.5B",
    local_model_path: str = os.path.join(PROJECT_DIR, "models/Qwen2.5-Math-1.5B-Base"),
    data_path: str = os.path.join(PROJECT_DIR, "data/gsm8k/train.jsonl"),
    prompt_path: str = os.path.join(PROJECT_DIR, "cs336_alignment/prompts/r1_zero.prompt"),
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_tokens: int = 1024,
    seed: int = 123,):

    dotenv.load_dotenv()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    os.environ["HF_HOME"] = "~/autodl-tmp/hf"
    os.environ["TRANSFORMERS_CACHE"] = "~/autodl-tmp/hf/models"
    os.environ["HF_HUB_CACHE"] = "~/autodl-tmp/hf/hub"

    train_config = TrainEIConfig()
    eval_config = EvaluateConfig()

    # Login Wandb
    api_key = os.getenv("WANDB_API_KEY")
    wandb.login(key=api_key)
    logging.info(f"[train] wandb login with key: {api_key}")
    
    # set train_config
    train_config.rollout_per_prompt = rollout_per_prompt
    train_config.sft_train_epochs = sft_train_epochs
    train_config.sft_train_batch_size = sft_train_batch_size
    train_config.micro_batch_size = micro_batch_size
    train_config.gradient_accumulation_steps = sft_train_batch_size // micro_batch_size
    logging.info(f"[train_config reset] TrainConfig local_model_path: {train_config.local_model_path},"
                    f" train_data_path: {train_config.train_data_path}, "
                    f" sft_train_batch_size: {train_config.sft_train_batch_size}, "
                    f" gradient_accumulation_steps: {train_config.gradient_accumulation_steps}, "
                    f" micro_batch_size: {train_config.micro_batch_size}, "
                    f" sft_train_epochs: {train_config.sft_train_epochs}")
    
    for sample_questions_per_ei_step in sample_questions_per_ei_step_params:
        # TODO: set config and log
        train_config.sample_questions_per_ei_step = sample_questions_per_ei_step
        logging.info(f"start train EI for sample_questions_per_ei_step={train_config.sample_questions_per_ei_step}")
        # 初始化vllm
        vllm = init_vllm(model_id=local_model_path, device=train_config.eval_device, seed=seed)
        logging.info(f"init_vllm with model_id: {local_model_path}, device: {train_config.eval_device}, seed: {seed}, gpu_memory_utilization: 0.9")
        
        train_ei_model(
            train_config,
            eval_config=eval_config,
            vllm=vllm,
        )

    wandb.finish()
    print(f"finish EI train")

# 控制变量:
# train_config.sample_questions_per_ei_step (每个专家迭代步骤的批量大小（即 Db 的大小))
# train_config.rollout_per_prompt (每个问题的 rollout 数量 G)
# train_config.training_steps (SFT 步骤中使用的 epoch 数量)
# 不变: rollout:4, training_steps=8 => sample_questions_per_ei_step [512, 1024, 2048]
# 不变: sample_questions_per_ei_step=512,train_steps=8 => rollout=4, rollout = 2, rollout=8
# 不变: sample_questions_per_ei_step=512, rollout=4 => training_steps=8, train_steps=16
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--sample_questions_per_ei_step", type=int,  nargs='+',
                    default=[512, 1024, 2048], help="sample_questions_per_ei_step")
    parser.add_argument("--rollout_per_prompt", type=int, default=4, help="rollout_per_prompt")
    parser.add_argument("--sft_train_epochs", type=int, default=10, help="sft train epochs")
    parser.add_argument("--sft_train_batch_size", type=int, default=256, help="sft train batch size")
    parser.add_argument("--micro_batch_size", type=int, default=8, help="micro batch size")
    args = parser.parse_args()
    test_sample_questions_per_ei_step = args.sample_questions_per_ei_step
    test_rollout_per_prompt = args.rollout_per_prompt
    test_sft_train_epochs = args.sft_train_epochs

    print(f"Start train EI for "
        f"sample_questions_per_ei_step={test_sample_questions_per_ei_step}, "
        f"rollout_per_prompt={test_rollout_per_prompt}, "
        f"sft_train_epochs={test_sft_train_epochs}",
        f"sft_train_batch_size={args.sft_train_batch_size}",
        f"micro_batch_size={args.micro_batch_size}")
    
    fire.Fire(main(sample_questions_per_ei_step_params=test_sample_questions_per_ei_step,
                 rollout_per_prompt=test_rollout_per_prompt,
                sft_train_epochs=test_sft_train_epochs,
                sft_train_batch_size=args.sft_train_batch_size,
                micro_batch_size=args.micro_batch_size))
    
    print(f"Finish train EI for "
        f"sample_questions_per_ei_step={test_sample_questions_per_ei_step}, "
        f"rollout_per_prompt={test_rollout_per_prompt}, "
        f"sft_train_epochs={test_sft_train_epochs}")
