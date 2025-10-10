import json
import logging
from pathlib import Path
from typing import Callable, List, Union

import fire
from vllm import LLM, SamplingParams

from cs336_alignment.data_utils import extract_reference_answer, load_and_format_prompts
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.utils import print_color, safe_slug
import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

logging.getLogger("vllm").setLevel(logging.WARNING)


def get_vllm_response(vllm_model, prompts, sampling_params) -> List[str]:
    result = vllm_model.generate(prompts, sampling_params)
    outputs = [output.outputs[0].text.strip() for output in result]
    return outputs

#evaluate 并记录所有correct的数据
def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    cot: List[str],
    true_answers: List[str],
    original_data: List[str],
    eval_sampling_params: SamplingParams,
):
    responses = get_vllm_response(vllm_model, prompts, eval_sampling_params)
    allinfo_dict_list = []
    for response, true_answer, prompt, original_line in zip(responses, true_answers, prompts, original_data):
        extracted_answer = extract_reference_answer(response)
        reward_dict = reward_fn(response, true_answer)
        print(f"response: {response}, true_answer: {true_answer}, extracted_answer: {extracted_answer}, reward_dict: {reward_dict}")

        info_dict: dict[str, Union[str, float]] = {
            "prompt": prompt,
            "response": response,
            "true_answer": true_answer,
            "original_data": original_line,
            "extracted_answer": extracted_answer,
            **reward_dict,
        }

        allinfo_dict_list.append(info_dict)

    return allinfo_dict_list

temperature: float = 1.0
top_p: float = 1.0
max_tokens: int = 1024
stop_tokens: list[str] = field(default_factory=lambda: ["</answer>"])
include_stop_str_in_output: bool = True
min_tokens: int = 4

def main(
    *,
    model_name: str = "Qwen/Qwen2.5-Math-1.5B",
    local_model_path: str = os.path.join(PROJECT_DIR, "models/Qwen2.5-Math-1.5B-Base"),
    data_path: str = os.path.join(PROJECT_DIR, "data/gsm8k/train.jsonl"),
    prompt_path: str = os.path.join(PROJECT_DIR, "cs336_alignment/prompts/r1_zero.prompt"),
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_tokens: int = 1024,
):
    print_color(f"Evaluating {model_name} on {data_path}")

    # 基于本地已经下载好的模型路径加载vLLM模型
    vllm_model = LLM(local_model_path)

    # Ensure we sample multiple outputs per prompt.
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop=stop_tokens,
        include_stop_str_in_output=include_stop_str_in_output,
        min_tokens=min_tokens,
        n=1,
    )

    # 保留原有的question, answer + prompt,cot,true_answer
    prompts, cot, true_answers, original_data = load_and_format_prompts_with_original_data(data_path, prompt_path)

    results = evaluate_vllm(vllm_model, r1_zero_reward_fn, prompts, cot, true_answers, original_data, sampling_params)

    # Save the results
    model_tag = safe_slug(model_name)
    data_stem = Path(data_path).stem
    out_dir = Path("evaluations")
    out_dir.mkdir(parents=True, exist_ok=True)
    evaluate_out_file = out_dir / f"evaluate_{model_tag}_{data_stem}.jsonl"
    correct_out_file = os.path.join(PROJECT_DIR, "data/gsm8k/correct_train.jsonl")
    correct_count = 0
    format_rewards = 0
    answer_reward = 0
    reward = 0
    with open(evaluate_out_file, "w", encoding="utf-8") as f:
        for result in results:
            if result["extracted_answer"] == result["true_answer"]:
                correct_count += 1
            format_rewards += result["format_reward"]
            answer_reward += result["answer_reward"]
            reward += result["reward"]
            json.dump(result, f)
            f.write("\n")
    

    with open(correct_out_file, "w", encoding="utf-8") as f:
        for result in results:
            if result["reward"] == 1.0:
                original_data_line = result["original_data"]
                f.write(original_data_line)
                f.write("\n")

    print_color(f"Correct answers: {correct_count}/{len(results)}", "green")
    print_color(f"Format rewards: {format_rewards}/{len(results)}", "green")
    print_color(f"Answer rewards: {answer_reward}/{len(results)}", "green")
    print_color(f"Total rewards: {reward}/{len(results)}", "green")
    print(f"Wrote {out_file}")


if __name__ == "__main__":
    fire.Fire(main)
