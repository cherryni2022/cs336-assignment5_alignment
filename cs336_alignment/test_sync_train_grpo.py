import os
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional
import torch
from transformers import AutoModelForCausalLM
from vllm import SamplingParams

from cs336_alignment.vllm_utils import init_vllm, load_model_into_vllm_instance
from cs336_alignment.my_train_sft import evaluate_vllm
from cs336_alignment.data_utils import load_and_format_prompts
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
logging.getLogger("vllm").setLevel(logging.WARNING)
logging.basicConfig(
    format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


@dataclass
class TrainConfig:
    model_name: str = "Qwen/Qwen2.5-Math-1.5B"
    local_model_path: str = os.path.join(PROJECT_DIR, "models/Qwen2.5-Math-1.5B-Base")
    vllm_device: str = "cuda:0"  # 单卡共享一个 vLLM 对象
    vllm_seed: int = 42
    vllm_gpu_mem_util: float = 0.9
    prompt_path: str = os.path.join(PROJECT_DIR, "cs336_alignment/prompts/r1_zero.prompt")
    train_data_path: str = os.path.join(PROJECT_DIR, "data/gsm8k/train.jsonl")

    # sampling for rollout
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 512
    stop_tokens: List[str] = field(default_factory=lambda: ["</answer>"])
    include_stop_str_in_output: bool = True
    group_size: int = 4


@dataclass
class EvaluateConfig:
    data_path: str = os.path.join(PROJECT_DIR, "data/gsm8k/test.jsonl")
    prompt_path: str = os.path.join(PROJECT_DIR, "cs336_alignment/prompts/r1_zero.prompt")
    eval_result_dir: str = os.path.join(PROJECT_DIR, "evaluations/grpo")
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 512
    stop_tokens: List[str] = field(default_factory=lambda: ["</answer>"])
    include_stop_str_in_output: bool = True


def main(sample_train_prompts: int = 8, do_eval_every: int = 1, num_steps: int = 3, vllm_device_override: Optional[str] = None):
    train_cfg = TrainConfig()
    eval_cfg = EvaluateConfig()

    # 可选覆盖设备（若通过 CLI 指定）
    if vllm_device_override:
        train_cfg.vllm_device = vllm_device_override

    t_total_start = time.time()
    logging.info("[main] loading base HF model on train device for simulation")
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=train_cfg.local_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cpu",
    )
    # 初始化单个 vLLM 实例（共享给 rollout 和 evaluate，串行执行）
    llm = init_vllm(
        model_id=train_cfg.local_model_path,
        device=train_cfg.vllm_device,
        seed=train_cfg.vllm_seed,
        gpu_memory_utilization=train_cfg.vllm_gpu_mem_util,
    )
    logging.info(f"[sync-main] vLLM initialized on {train_cfg.vllm_device} with mem_util={train_cfg.vllm_gpu_mem_util}")

    # 加载训练 prompts（用于 rollout 生成）
    train_prompts, train_cot, train_answers = load_and_format_prompts(
        train_cfg.train_data_path, train_cfg.prompt_path
    )
    sample_prompts = train_prompts[:sample_train_prompts]

    # 串行模拟训练过程：每步先 rollout，再按频率 evaluate（都用同一个 llm）
    for step in range(num_steps):
        logging.info(f"[sync-main] step {step}: rollout generation start")
        t_ro_start = time.time()
        sp_rollout = SamplingParams(
            temperature=train_cfg.temperature,
            top_p=train_cfg.top_p,
            max_tokens=train_cfg.max_tokens,
            stop=train_cfg.stop_tokens,
            include_stop_str_in_output=train_cfg.include_stop_str_in_output,
            n=train_cfg.group_size,
        )
        responses = llm.generate(sample_prompts, sp_rollout)
        all_prompts: List[str] = []
        generations: List[str] = []
        for prompt, response in zip(sample_prompts, responses):
            for _, output in enumerate(response.outputs):
                all_prompts.append(prompt)
                generations.append(output.text)
        ro_dur = time.time() - t_ro_start
        logging.info(
            f"[sync-main] rollout received {len(all_prompts)} pairs, each prompt with {train_cfg.group_size} candidates"
        )
        logging.info(f"[sync-main] rollout duration: {ro_dur:.3f}s")

        # 按周期执行评估（串行，使用同一个 llm）
        if (step + 1) % do_eval_every == 0:
            logging.info(f"[sync-main] step {step}: evaluation start")
            t_ev_start = time.time()
            eval_prompts, eval_cot, eval_answers = load_and_format_prompts(
                eval_cfg.data_path, eval_cfg.prompt_path
            )
            sp_eval = SamplingParams(
                temperature=eval_cfg.temperature,
                top_p=eval_cfg.top_p,
                max_tokens=eval_cfg.max_tokens,
                stop=eval_cfg.stop_tokens,
                include_stop_str_in_output=eval_cfg.include_stop_str_in_output,
            )
            results, info_list = evaluate_vllm(llm, r1_zero_reward_fn, eval_prompts, eval_answers, sp_eval)
            os.makedirs(eval_cfg.eval_result_dir, exist_ok=True)
            out_file = os.path.join(eval_cfg.eval_result_dir, f"sync_test_evaluate_{step}.jsonl")
            with open(out_file, "w", encoding="utf-8") as f:
                import json
                for it in info_list:
                    # 仅记录正确或奖励为1的样本
                    if it.get("extracted_answer") == it.get("true_answer") or it.get("reward") == 1:
                        json.dump(it, f)
                        f.write("\n")
            ev_dur = time.time() - t_ev_start
            logging.info(f"[sync-main] eval accuracy: {results.get('accuracy'):.4f}, saved: {out_file}")
            logging.info(f"[sync-main] evaluation duration: {ev_dur:.3f}s")

        # 训练侧可在此进行权重更新；本测试仅模拟等待
        time.sleep(1.0)
        logging.info(f"[main train] step_{step} save_model_state_dict start=========")
        load_model_into_vllm_instance(model, llm, train_cfg.vllm_device)
        logging.info(f"[main train] step_{step} save_model_state_dict end =========")
        

    total_dur = time.time() - t_total_start
    logging.info("[sync-main] test finished")
    logging.info(f"[sync-main] total duration: {total_dur:.3f}s")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_train_prompts", type=int, default=8, help="number of train prompts to rollout")
    parser.add_argument("--do_eval_every", type=int, default=1, help="run evaluation every N steps")
    parser.add_argument("--num_steps", type=int, default=3, help="total training steps to simulate")
    parser.add_argument("--vllm_device", type=str, default=None, help="device for shared vLLM, e.g., cuda:1")
    args = parser.parse_args()
    main(
        sample_train_prompts=args.sample_train_prompts,
        do_eval_every=args.do_eval_every,
        num_steps=args.num_steps,
        vllm_device_override=args.vllm_device,
    )