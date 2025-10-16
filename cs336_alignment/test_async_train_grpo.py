import os
import logging
import time
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM
from vllm import SamplingParams

from cs336_alignment.vllm_utils import init_vllm
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
    train_device: str = "cuda:0"
    rollout_device: str = "cuda:0"
    eval_device: str = "cuda:0"
    vllm_seed: int = 42
    rollout_gpu_mem_util: float = 0.45
    eval_gpu_mem_util: float = 0.45
    prompt_path: str = os.path.join(PROJECT_DIR, "cs336_alignment/prompts/r1_zero.prompt")
    train_data_path: str = os.path.join(PROJECT_DIR, "data/gsm8k/train.jsonl")

    # sampling
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


def _load_weights_from_state_dict(llm, cpu_state_dict: dict, device: str):
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(cpu_state_dict.items())
    torch.cuda.synchronize(torch.device(device))


def rollout_worker(cmd_q: mp.Queue, res_q: mp.Queue, local_model_path: str, device: str, seed: int, gpu_mem_util: float,
                   temperature: float, top_p: float, max_tokens: int, stop_tokens: List[str], include_stop: bool):
    logging.basicConfig(format="%(asctime)s - rollout - %(levelname)s - %(message)s", level=logging.INFO)
    llm = init_vllm(model_id=local_model_path, device=device, seed=seed, gpu_memory_utilization=gpu_mem_util)
    logging.info(f"[rollout-worker] vLLM initialized on {device} with mem_util={gpu_mem_util}")

    while True:
        cmd = cmd_q.get()
        if cmd is None:
            continue
        ctype = cmd.get("type")
        if ctype == "shutdown":
            logging.info("[rollout-worker] shutdown received")
            break
        elif ctype == "update_weights":
            logging.info("[rollout-worker] update_weights received")
            cpu_sd = cmd.get("state_dict")
            _load_weights_from_state_dict(llm, cpu_sd, device)
            res_q.put({"type": "ack", "op": "update_weights"})
            logging.info("[rollout-worker] update_weights finished")
        elif ctype == "rollout":
            prompts: List[str] = cmd.get("prompts", [])
            n: int = cmd.get("n", 1)
            logging.info(f"[rollout-worker] rollout received {len(prompts)} prompts, n_outputs={n}")
            sp = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stop=stop_tokens,
                include_stop_str_in_output=include_stop,
                n=n,
            )
            responses = llm.generate(prompts, sp)
            all_prompts = []
            generations = []
            for prompt, response in zip(prompts, responses):
                for _, output in enumerate(response.outputs):
                    all_prompts.append(prompt)
                    generations.append(output.text)
            res_q.put({"type": "rollout_result", "prompts": all_prompts, "generations": generations})
        else:
            logging.warning(f"[rollout-worker] unknown cmd: {ctype}")


def eval_worker(cmd_q: mp.Queue, res_q: mp.Queue, local_model_path: str, device: str, seed: int, gpu_mem_util: float,
                eval_cfg: EvaluateConfig):
    logging.basicConfig(format="%(asctime)s - eval - %(levelname)s - %(message)s", level=logging.INFO)
    llm = init_vllm(model_id=local_model_path, device=device, seed=seed, gpu_memory_utilization=gpu_mem_util)
    logging.info(f"[eval-worker] vLLM initialized on {device} with mem_util={gpu_mem_util}")

    while True:
        cmd = cmd_q.get()
        if cmd is None:
            continue
        ctype = cmd.get("type")
        if ctype == "shutdown":
            logging.info("[eval-worker] shutdown received")
            break
        elif ctype == "update_weights":
            cpu_sd = cmd.get("state_dict")
            _load_weights_from_state_dict(llm, cpu_sd, device)
            res_q.put({"type": "ack", "op": "update_weights"})
        elif ctype == "evaluate":
            step: int = cmd.get("step", 0)
            logging.info(f"[eval-worker] evaluate received step={step}")
            prompts, cot, answers = load_and_format_prompts(eval_cfg.data_path, eval_cfg.prompt_path)
            sp = SamplingParams(
                temperature=eval_cfg.temperature,
                top_p=eval_cfg.top_p,
                max_tokens=eval_cfg.max_tokens,
                stop=eval_cfg.stop_tokens,
                include_stop_str_in_output=eval_cfg.include_stop_str_in_output,
            )
            results, info_list = evaluate_vllm(llm, r1_zero_reward_fn, prompts, answers, sp)
            os.makedirs(eval_cfg.eval_result_dir, exist_ok=True)
            out_file = os.path.join(eval_cfg.eval_result_dir, f"async_test_evaluate_{step}.jsonl")
            with open(out_file, "w", encoding="utf-8") as f:
                for it in info_list:
                    # 仅记录正确或奖励为1的样本
                    if it.get("extracted_answer") == it.get("true_answer") or it.get("reward") == 1:
                        import json
                        json.dump(it, f)
                        f.write("\n")
            logging.info(f"[eval-worker] eval results saved to {out_file}")
            res_q.put({"type": "evaluate_result", "results": results, "file": out_file})
        else:
            logging.warning(f"[eval-worker] unknown cmd: {ctype}")

def save_model_state_dict(model: AutoModelForCausalLM, state_dict: dict[str, torch.Tensor]):
    #手动保存到cpu state_dict
    model.eval()
    model.tie_weights()
    cpu_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.train()

def main(sample_train_prompts: int = 8, do_eval_every: int = 1):
    # 采用 spawn 以避免 CUDA 在 fork 后的上下文问题
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    train_cfg = TrainConfig()
    eval_cfg = EvaluateConfig()

    logging.info("[main] loading base HF model on train device for simulation")
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=train_cfg.local_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cpu",
    )
    #.to(train_cfg.train_device)  # 只测试用一块卡rollout+eval, 不训练

    # 深拷贝权重到CPU，避免持有GPU张量
    model.eval()
    model.tie_weights()
    cpu_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.train()

    # 启动两个子进程（同卡 cuda:1），分别用于 rollout 与 evaluate
    rollout_cmd_q: mp.Queue = mp.Queue()
    rollout_res_q: mp.Queue = mp.Queue()
    eval_cmd_q: mp.Queue = mp.Queue()
    eval_res_q: mp.Queue = mp.Queue()

    rollout_p = mp.Process(
        target=rollout_worker,
        args=(rollout_cmd_q, rollout_res_q, train_cfg.local_model_path, train_cfg.rollout_device, train_cfg.vllm_seed,
              train_cfg.rollout_gpu_mem_util, train_cfg.temperature, train_cfg.top_p, train_cfg.max_tokens,
              train_cfg.stop_tokens, train_cfg.include_stop_str_in_output),
        daemon=True,
    )
    eval_p = mp.Process(
        target=eval_worker,
        args=(eval_cmd_q, eval_res_q, train_cfg.local_model_path, train_cfg.eval_device, train_cfg.vllm_seed,
              train_cfg.eval_gpu_mem_util, eval_cfg),
        daemon=True,
    )

    rollout_p.start()
    eval_p.start()
    logging.info("[main] rollout and eval workers started on cuda:1 with process isolation")

    # 初次加载训练权重到两个 vLLM 实例
    rollout_cmd_q.put({"type": "update_weights", "state_dict": cpu_state_dict})
    eval_cmd_q.put({"type": "update_weights", "state_dict": cpu_state_dict})

    # 等待权重加载完成的确认（可选）
    _ = rollout_res_q.get()
    _ = eval_res_q.get()
    logging.info("[main] weights loaded into both vLLM instances")

    # 构造少量训练 prompts，用于 rollout 生成
    train_prompts, train_cot, train_answers = load_and_format_prompts(train_cfg.train_data_path, train_cfg.prompt_path)
    sample_prompts = train_prompts[:sample_train_prompts]

    # 模拟若干轮训练，同时进行 rollout 与（周期性）evaluate
    for step in range(3):
        logging.info(f"[main] step {step}: rollout update_weights....")
        rollout_cmd_q.put({"type": "update_weights", "state_dict": cpu_state_dict})
        _ = rollout_res_q.get()
        logging.info(f"[main] step {step}: rollout update_weights finished")

        logging.info(f"[main] step {step}: dispatch rollout generation")
        rollout_cmd_q.put({"type": "rollout", "prompts": sample_prompts, "n": train_cfg.group_size})

        if (step + 1) % do_eval_every == 0:
            logging.info(f"[main] step {step}: evaluate update_weights....")
            eval_cmd_q.put({"type": "update_weights", "state_dict": cpu_state_dict})
            _ = eval_res_q.get()
            logging.info(f"[main] step {step}: evaluate update_weights finished....")

            logging.info(f"[main] step {step}: dispatch evaluation")
            eval_cmd_q.put({"type": "evaluate", "step": step})

        # 等待 rollout 结果
        ro_msg = rollout_res_q.get(timeout=600)
        prompts = ro_msg.get("prompts")
        gens = ro_msg.get("generations")
        logging.info(f"[main] rollout received {len(prompts)} prompts, {len(gens)} responses, each with {train_cfg.group_size} candidates")

        # 获取 eval 结果（如果有）
        ev_msg: Optional[dict] = None
        try:
            ev_msg = eval_res_q.get(timeout=600)
        except Exception:
            pass
        if ev_msg:
            results = ev_msg.get("results")
            out_file = ev_msg.get("file")
            logging.info(f"[main] eval accuracy: {results.get('accuracy'):.4f}, saved: {out_file}")

        # 训练侧可在此进行权重更新；本测试仅模拟等待
        time.sleep(1.0)
        logging.info(f"[main train] step_{step} save_model_state_dict start=========")
        save_model_state_dict(model, cpu_state_dict)
        logging.info(f"[main train] step_{step} save_model_state_dict end =========")


    # 结束子进程
    rollout_cmd_q.put({"type": "shutdown"})
    eval_cmd_q.put({"type": "shutdown"})
    rollout_p.join(timeout=60)
    eval_p.join(timeout=60)
    logging.info("[main] test finished, workers terminated")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_train_prompts", type=int, default=8, help="number of train prompts to rollout")
    parser.add_argument("--do_eval_every", type=int, default=1, help="run evaluation every N steps")
    args = parser.parse_args()
    main(sample_train_prompts=args.sample_train_prompts, do_eval_every=args.do_eval_every)