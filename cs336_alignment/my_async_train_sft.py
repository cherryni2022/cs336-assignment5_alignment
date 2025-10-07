import logging
import math
import os
import random
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Callable, List
import time
import dotenv
import fire
import torch
import torch.nn as nn
from argparse import ArgumentParser
import wandb
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
import json

import threading
import queue
import copy
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Optional


from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.data_utils import extract_reference_answer, extract_gsm8k_answer, load_and_format_prompts
from cs336_alignment.sft_utils import (
    get_response_log_probs,
    sft_microbatch_train_step,
    tokenize_prompt_and_output
)
from cs336_alignment.evaluate import get_vllm_response
from cs336_alignment.utils import (
    get_run_name,
    print_rich_dict,
    save_model_and_tokenizer,
)
from cs336_alignment.vllm_utils import init_vllm, load_model_into_vllm_instance

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
logging.getLogger("vllm").setLevel(logging.WARNING)
logging.basicConfig(
    format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

SEED = 69
torch.manual_seed(SEED)
random.seed(SEED)

@dataclass
class TrainConfig:
    experiment_name_base: str = "experiments"
    experiment_name: str = "sft-qwen2.5-math"
    model_name: str = "Qwen/Qwen2.5-Math-1.5B"
    local_model_path: str = os.path.join(PROJECT_DIR, "models/Qwen2.5-Math-1.5B-Base")
    train_data_path: str = os.path.join(PROJECT_DIR, "data/gsm8k/train.jsonl")
    prompt_path: str = os.path.join(PROJECT_DIR, "cs336_alignment/prompts/r1_zero.prompt")

    # 控制train 的参数变量
    n_sft_steps: int = 256
    gradient_accumulation_steps: int = 32
    micro_batch_size: int = 8
    sft_train_batch_size: int = 256
    mixed_precision_training: bool = True
    learning_rate: float = 2e-5
    betas: tuple[float, float] = (0.9, 0.98)
    train_device: str = "cuda:0"

    # sft完整训练过程的训练数据 =>从数据集的train中抽取用于训练的数量
    # 作业要求: []
    num_train_samples: int = 128
    use_correct_samples: bool = False

    # Log print:
    log_print_steps = 16

    # For evaluation
    eval_device: str = "cuda:1"
    eval_interval_steps: int = 16

    def __post_init__(self):
        self.gradient_accumulation_steps = self.sft_train_batch_size // self.micro_batch_size
        logging.info(f"[train_config init] sft_train_batch_size: {self.sft_train_batch_size}, "
              f" gradient_accumulation_steps: {self.gradient_accumulation_steps}, "
              f" micro_batch_size: {self.micro_batch_size}")

@dataclass
class EvaluateConfig:
    data_path: str = os.path.join(PROJECT_DIR, "data/gsm8k/test.jsonl")
    prompt_path: str = os.path.join(PROJECT_DIR, "cs336_alignment/prompts/r1_zero.prompt")
    eval_result_dir: str = os.path.join(PROJECT_DIR, "evaluations")
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 1024

class AsyncEvaluator:
    def __init__(self, eval_config: EvaluateConfig, vllm: LLM):
        self.eval_config = eval_config
        self.vllm = vllm
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="eval")
        self.current_eval_future: Optional[Future] = None
        
    def start_async_evaluation(self, model: torch.nn.Module, eval_step: int) -> Future:
        """异步启动评估任务"""
        # 等待上一个评估完成
        if self.current_eval_future and not self.current_eval_future.done():
            logging.warning(f"Previous evaluation still running, waiting...")
            self.current_eval_future.result()  # 阻塞等待完成
            
        # 创建模型权重的深拷贝到CPU
        model.eval()
        model.tie_weights()
        cpu_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.train()
        
        # 提交异步评估任务
        self.current_eval_future = self.executor.submit(
            self._evaluate_async, cpu_state_dict, eval_step
        )
        return self.current_eval_future
        
    def _evaluate_async(self, cpu_state_dict: dict, eval_step: int):
        """异步评估的实际执行函数"""
        try:
            logging.info(f"[async_eval] Starting evaluation for step {eval_step}")
            
            # 加载权重到vLLM实例
            llm_model = self.vllm.llm_engine.model_executor.driver_worker.model_runner.model
            llm_model.load_weights(cpu_state_dict.items())
            torch.cuda.synchronize(torch.device("cuda:1"))
            
            # 执行评估
            evaluate_sft_model(self.eval_config, self.vllm, eval_step=eval_step)
            
            logging.info(f"[async_eval] Evaluation completed for step {eval_step}")
            
        except Exception as e:
            logging.error(f"[async_eval] Evaluation failed for step {eval_step}: {e}")
            raise
        finally:
            # 清理CPU内存
            del cpu_state_dict
            torch.cuda.empty_cache()
            
    def wait_for_completion(self):
        """等待当前评估完成"""
        if self.current_eval_future:
            self.current_eval_future.result()
            
    def shutdown(self):
        """关闭异步评估器"""
        self.wait_for_completion()
        self.executor.shutdown(wait=True)
# evaluate阶段log信息
# 给定的提示（例如，从验证集中采样）生成响应。为每个示例记录以下内容是个好主意：
# 1. 输入提示。
# 2. SFT/RL 模型生成的响应。
# 3. 标准答案。
# 4. 奖励信息，包括格式、答案和总奖励。
# 5. 响应的平均词元熵。
# 6. 平均响应长度、正确响应的平均响应长度和不正确响应的平均响应长度。
# TODO: eval阶段log结果, 不需要单独vllm load model to log
def log_generations(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    cot: List[str],
    answers: List[str],
    eval_sampling_params: SamplingParams,
    cur_step: int,
    sample_num: int = 2,
    ):
    random_indices = random.sample(range(len(prompts)), k=sample_num)
    sampled_prompts = [prompts[i] for i in random_indices]
    sampled_cot = [cot[i] for i in random_indices]
    sampled_answers = [answers[i] for i in random_indices]

    responses = get_vllm_response(vllm_model, sampled_prompts, eval_sampling_params)

    for i in range(sample_num):
        response = responses[i]
        answer = sampled_answers[i]
        prompt = sampled_prompts[i]
        extracted_answer = extract_reference_answer(response)
        true_label = sampled_cot[i]

        reward_dict = reward_fn(response, answer)

        info_dict = {
            "prompt": prompt,
            "true_label": true_label,
            "response": response,
            "true_answer": answer,
            "extracted_answer": extracted_answer,
            **reward_dict,
        }

        logging.info(f"======= Step: {cur_step}; Example {i} =======")
        # print_formatted_dict(info_dict)
        print_rich_dict(info_dict)
        logging.info("==============================================\n")

def get_lr(it, max_lr, max_steps):
    min_lr = max_lr * 0.1
    if it >= max_steps:
        return min_lr
    # pure cosine decay
    decay_ratio = it / max_steps
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

def get_cosine_lr(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """Cosine with warmup learning rate scheduler."""
    # First, we linearly warmup for warmup_iters steps.
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    # Then, if it > cosine_cycle_iters, we return min learning rate.
    if it > cosine_cycle_iters:
        return min_learning_rate
    # Else, we use cosine decay down to min learning rate.
    decay_ratio = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_learning_rate + coeff * (max_learning_rate - min_learning_rate)

def to_float(x):
    if isinstance(x, torch.Tensor):
        return x.float().item()
    elif isinstance(x, str):
        return float(x.strip())

    return float(x)

class SFTDataSet(Dataset):
    def __init__(self, train_prompts, train_cot, train_answers):
        self.train_prompts = train_prompts
        self.train_cot = train_cot
        self.train_answers = train_answers
        logging.info(f"[SFTDataSet] Loaded {len(self.train_prompts)} samples.")

    def __len__(self):
        return len(self.train_prompts)

    def __getitem__(self, index):
        prompt = self.train_prompts[index]
        cot = self.train_cot[index]
        answer = to_float(self.train_answers[index])
        return prompt, cot, answer

# data_loader 处理datasets => tokennizer 转化后的train数据
def sft_collate_fn(batch, tokenizer):
    prompts, cot, answers = zip(*batch)  # each is a tuple of strings
    prompts = list(prompts)
    cot = list(cot)

    batch_enc = tokenize_prompt_and_output(prompts, cot, tokenizer)

    return {**batch_enc, "answers": answers}

def evaluate_vllm(vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    answers: List[str],
    eval_sampling_params: SamplingParams,):
    logging.info(f"[eval] Evaluating {len(prompts)} samples.")
    responses = get_vllm_response(vllm_model, prompts, eval_sampling_params)
    total = len(prompts)
    total_reward = 0
    format_reward = 0
    answer_reward = 0
    response_length = 0
    correct_response_length = 0
    incorrect_response_length = 0
    log_samples = 3
    allinfo_dict_list = []
    for prompt, response, answer in zip(prompts, responses, answers):
        reward_dict = reward_fn(response, answer)
        total_reward += reward_dict["reward"]
        format_reward += reward_dict["format_reward"]
        answer_reward += reward_dict["answer_reward"]

        # 输出每条数据的判断结果
        extracted_answer = extract_reference_answer(response)
        info_dict: dict[str, Union[str, float]] = {
            "prompt": prompt,
            "response": response,
            "true_answer": answer,
            "extracted_answer": extracted_answer,
            **reward_dict,
        }
        allinfo_dict_list.append(info_dict)

        response_length += len(response.split())
        if reward_dict["reward"] == 1:
            correct_response_length += len(response.split())
        else:
            incorrect_response_length += len(response.split())
        
        if log_samples > 0:
            logging.info(f"[eval] sample prompt: {prompt}"
                        f", response: {response}"
                        f", answer: {answer}"
                        f", reward_result: {reward_dict}")
            log_samples -= 1

    avg_response_length = response_length // total
    avg_correct_response_length = correct_response_length // total_reward
    avg_incorrect_response_length = incorrect_response_length // (total - total_reward)

    format_wrong = total - format_reward
    answer_wrong = total - answer_reward
    results = {"correct": total_reward, 
                "format_wrong": format_wrong, 
                "answer_wrong": answer_wrong, 
                "count": total,
                "accuracy": total_reward / total,
                "format_wrong_rate": format_wrong / total,
                "answer_wrong_rate": answer_wrong / total,
                "avg_response_length": avg_response_length,
                "avg_correct_response_length": avg_correct_response_length,
                "avg_incorrect_response_length": avg_incorrect_response_length,
    }
    logging.info(f"[eval] statistic result: {results}")
    return results, allinfo_dict_list

def evaluate_sft_model(config: EvaluateConfig, vllm: LLM, eval_step: int):
    prompts, cot, answers = load_and_format_prompts(config.data_path, config.prompt_path)

    sampling_params = SamplingParams(
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    results, outinfo_dict_list = evaluate_vllm(vllm, r1_zero_reward_fn, prompts, answers, sampling_params)

    wandb.log(
        {
            "eval/correct": results["correct"],
            "eval/count": results["count"],
            "eval/answer_wrong": results["answer_wrong"],
            "eval/format_wrong": results["format_wrong"],
            "eval/accuracy": results["accuracy"],
            "eval/format_wrong_rate": results["format_wrong_rate"],
            "eval/answer_wrong_rate": results["answer_wrong_rate"],
            "eval/avg_response_length": results["avg_response_length"],
            "eval/avg_correct_response_length": results["avg_correct_response_length"],
            "eval/avg_incorrect_response_length": results["avg_incorrect_response_length"],
            "eval_step": eval_step,
        }
    )

    # 将eval test.jsonl数据的结果写文件
    eval_result_dir = config.eval_result_dir
    #eval_result_dir.mkdir(parents=True, exist_ok=True)
    out_file = os.path.join(eval_result_dir, f"evaluate_{eval_step}.jsonl")
    logging.info(f"[eval] Writing results to {out_file}")
    with open(out_file, "w", encoding="utf-8") as f:
        for result in outinfo_dict_list:
            if result["extracted_answer"] == result["true_answer"] or result["reward"] == 1:
                json.dump(result, f)
                f.write("\n")
    logging.info(f"[eval] Writing results to {out_file} finish ===========")

# 针对超参数配置sft训练模型
def train_sft_model(
    model,
    tokenizer,
    train_config: TrainConfig,
    eval_config: EvaluateConfig,
    vllm: LLM,
    train_prompts,
    train_cot,
    train_answers,
    evaluate: bool = True,):
    # step1: init wandb
    date_str = time.strftime("%m%d-%H%M%S")
    data_type = "correct" if train_config.use_correct_samples else "mix"
    lr = train_config.learning_rate
    wandb.init(
        entity=os.getenv("WANDB_ENTITY"),
        project="cs336-sft",
        name=f"train_sft_{data_type}_{train_config.num_train_samples}_b{train_config.sft_train_batch_size}_{lr}_{date_str}",
        config={
            "train_sample": train_config.num_train_samples,
            "learning_rate": train_config.learning_rate,
            "micro_batch_size": train_config.micro_batch_size,
            "batch_size": train_config.sft_train_batch_size,
        },
        reinit=True,)
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    # step2: 准备训练数据 & tokenizer
    train_dataset = SFTDataSet(train_prompts, train_cot, train_answers)
    train_dataloader = DataLoader(
        dataset = train_dataset,
        #batch_size = train_config.sft_train_batch_size,
        batch_size = train_config.micro_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=lambda batch: sft_collate_fn(batch, tokenizer),
    )
    logging.info(f"[train] Dataloader initialized with batch size {train_config.micro_batch_size} finish")

    # step3:准备模型训练的optimazer等
    # ---------------------
    # Mixed Precision Context
    # ---------------------
    ctx = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if train_config.mixed_precision_training
        else nullcontext()
    )
    # ---------------------
    # Optimizer
    # ---------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate, betas=train_config.betas)
    print("[train] Optimizer initialized")
    
     # 创建异步评估器
    async_evaluator = AsyncEvaluator(eval_config, vllm) if evaluate else None
    # step4:训练模型
    try:
        for sft_step in range(train_config.n_sft_steps):
            batch_loss = 0
            for curr_grad_accum_step in range(train_config.gradient_accumulation_steps):
                # 从dataloader中获取一个micro_batch的训练数据
                batch = next(iter(train_dataloader))
                input_ids = batch["input_ids"].to(train_config.train_device)
                labels = batch["labels"].to(train_config.train_device)
                response_mask = batch["response_mask"].to(train_config.train_device)

                with ctx:
                    log_prob_response = get_response_log_probs(model=model, 
                                        input_ids=input_ids, labels=labels, 
                                        return_token_entropy=True)
                    log_prob = log_prob_response["log_probs"]
                    entropy = log_prob_response["token_entropy"]
                    # 计算loss & backward()
                    loss, _ = sft_microbatch_train_step(
                        log_prob, response_mask, train_config.gradient_accumulation_steps
                    )
                    logging.info(f"[train test log] Step {sft_step} | "
                            f"Gradient accumulation step {curr_grad_accum_step} | "
                            f" | response_mask_entropy: {entropy[response_mask].mean().item():.6f}"
                            f" | Global Entropy: {entropy.mean().item():.6f}"
                            f" | Prompt Entropy: {entropy[~response_mask].mean().item():.6f}"
                            f" | Loss: {loss.item():.4f}")

                batch_loss += loss

                # 累积梯度更新参数
                if curr_grad_accum_step == train_config.gradient_accumulation_steps - 1:
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    adj_lr = get_lr(sft_step, train_config.learning_rate, train_config.n_sft_steps)
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = adj_lr
                    # 更新参数
                    optimizer.step()
                    optimizer.zero_grad()

                    logging.info(
                        f"[train] Step result summary at step_{sft_step}: "
                        f" | accumulate_batch_loss: {batch_loss:.4f}"
                        f" | avg_loss: {batch_loss / train_config.gradient_accumulation_steps:.4f}"
                        f" | response_mask_entropy: {entropy[response_mask].mean().item():.6f}"
                        f" | Global Entropy: {entropy.mean().item():.6f}"
                        f" | Prompt Entropy: {entropy[~response_mask].mean().item():.6f}"
                        f" | adj_lr: {adj_lr:.6f}"
                    )

                    wandb.log(
                        {
                            "train_step": sft_step + 1,
                            "train/accumulate_batch_loss": to_float(batch_loss),
                            "train/avg_loss": to_float(batch_loss / train_config.gradient_accumulation_steps),
                            "train/entropy": to_float(entropy.mean()),
                            "train/response entropy": to_float(entropy[response_mask].mean()),
                            "train/prompt entropy": to_float(entropy[~response_mask].mean()),
                            "train/lr": adj_lr,
                        }
                    )

            # if (sft_step + 1) % train_config.log_print_steps == 0 and evaluate:
            #     load_model_into_vllm_instance(model, vllm)
            #     log_generations(
            #         vllm,
            #         reward_fn=r1_zero_reward_fn,
            #         prompts=train_dataset.train_prompts,
            #         cot=train_dataset.train_cot,
            #         answers=train_dataset.train_answers,
            #         eval_sampling_params=SamplingParams(
            #             temperature=eval_config.temperature,
            #             top_p=eval_config.top_p,
            #             max_tokens=eval_config.max_tokens,
            #             stop=["</answer>"],
            #             include_stop_str_in_output=True,
            #         ),
            #         cur_step=sft_step,
            #         num_example=3,
            #     )

            #迭代eval_interval_steps次后, vllm评估模型
            if (sft_step + 1) % train_config.eval_interval_steps == 0 and evaluate:
                logging.info(
                    f"[train] Step_{sft_step}: saving model at {train_config.experiment_name}_{train_config.num_train_samples}"
                )
                save_model_and_tokenizer(model, tokenizer, train_config)

                # Run async evaluatoin
                logging.info(f"[eval] at step_{sft_step} start async evaluation==================")
                eval_future = async_evaluator.start_async_evaluation(model, sft_step)
                # load_model_into_vllm_instance(model, vllm)
                # evaluate_sft_model(eval_config, vllm, eval_step=sft_step)
                logging.info(f"[eval] Evaluation completed for step_{sft_step}=====================")
    finally:
        if async_evaluator:
            async_evaluator.shutdown()

def main(
    *,
    train_samples:list[int] = [7400, 256, 512, 1024, 128], 
    train_batch_size: int,
    micro_batch_size: int,
    n_sft_steps: int,
    use_correct: bool = False,
    model_name: str = "Qwen/Qwen2.5-Math-1.5B",
    local_model_path: str = os.path.join(PROJECT_DIR, "models/Qwen2.5-Math-1.5B-Base"),
    data_path: str = os.path.join(PROJECT_DIR, "data/gsm8k/train.jsonl"),
    prompt_path: str = os.path.join(PROJECT_DIR, "cs336_alignment/prompts/r1_zero.prompt"),
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_tokens: int = 1024,
    seed: int = 42,
):
    dotenv.load_dotenv()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    train_config = TrainConfig()
    eval_config = EvaluateConfig()
    # Login Wandb
    api_key = os.getenv("WANDB_API_KEY")
    wandb.login(key=api_key)
    logging.info(f"[train] wandb login with key: {api_key}")
    train_config.n_sft_steps = n_sft_steps
    train_config.micro_batch_size = micro_batch_size
    train_config.sft_train_batch_size = train_batch_size
    train_config.gradient_accumulation_steps = train_batch_size // micro_batch_size
    logging.info(f"[train_config reset] TrainConfig local_model_path: {train_config.local_model_path},"
                f" train_data_path: {train_config.train_data_path}, "
                f" sft_train_batch_size: {train_config.sft_train_batch_size}, "
                f" gradient_accumulation_steps: {train_config.gradient_accumulation_steps}, "
                f" micro_batch_size: {train_config.micro_batch_size}, "
                f" n_sft_step: {train_config.n_sft_steps}")
    
    # 基于本地已经下载好的模型路径加载vLLM模型
    vllm = init_vllm(model_id=local_model_path, device=train_config.eval_device, seed=seed, gpu_memory_utilization=0.9)
    logging.info(f"init_vllm with model_id: {local_model_path}, device: {train_config.eval_device}, seed: {seed}, gpu_memory_utilization: 0.9")

    prompts, cot, answers = load_and_format_prompts(train_config.train_data_path, train_config.prompt_path)
    logging.info(f"load train data from {train_config.train_data_path}, dataset size: {len(prompts)}")
    for train_sample_num in train_samples:
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
        logging.info(f"[train] Tokenizer {train_config.model_name} loaded from {train_config.local_model_path}")
        logging.info(f"[train] Model {train_config.model_name} loaded on device: {train_config.train_device},"
                f" will eval on device:{train_config.eval_device}"
                f" from {train_config.local_model_path}")

        train_config.num_train_samples = train_sample_num
        train_config.use_correct_samples = use_correct

        # 只使用指定数量的样本进行训练
        train_prompts = prompts[:train_sample_num]
        train_cot = cot[:train_sample_num]
        train_answers = answers[:train_sample_num]
        logging.info(f"[train sft] training for {train_sample_num} samples")

        train_sft_model(
            model,
            tokenizer,
            train_config,
            eval_config=eval_config,
            vllm=vllm,
            train_prompts=train_prompts,
            train_cot=train_cot,
            train_answers=train_answers,
        )

    wandb.finish()

train_samples = [7400, 256, 512, 1024, 128]
train_batch_size = 256
micro_batch_size = 8

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--use_correct", type=bool, default=False, help="use correct train data or not")
    parser.add_argument("--train_samples", type=int, nargs='+',
                    default=[256, 512, 1024, 7400, 128], help="train_samples num from datasets")
    parser.add_argument("--train_batch_size", type=int, default=256, help="train batch size")
    parser.add_argument("--micro_batch_size", type=int, default=8, help="micro batch size")
    parser.add_argument("--n_sft_steps", type=int, default=144, help="n_sft_steps")
    args = parser.parse_args()
    if args.train_samples:
        test_train_samples = args.train_samples
    else:
        test_train_samples = train_samples
    if args.train_batch_size:
        test_train_batch_size = args.train_batch_size
    if args.micro_batch_size:
        test_micro_batch_size = args.micro_batch_size
    
    logging.info(f"[Start] sft_train for {test_train_samples} samples,"
                 f"train_batch_size: {test_train_batch_size},"
                 f"micro_batch_size: {test_micro_batch_size},"
                 f"n_sft_steps: {args.n_sft_steps},"
                 f"use_correct: {args.use_correct}")
    fire.Fire(main(train_samples=test_train_samples, 
                train_batch_size=test_train_batch_size,
                micro_batch_size=test_micro_batch_size,
                n_sft_steps=args.n_sft_steps,
                use_correct=args.use_correct))
