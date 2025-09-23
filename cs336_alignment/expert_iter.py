from vllm import LLM, SamplingParams
from typing import Callable, List, Tuple
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
import re
import json
import os
import gc
from collections import Counter
import argparse
from contextlib import contextmanager
# 获取项目根目录
import os
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

QWEN_MATH_BASE_MODEL_PATH = os.path.join(PROJECT_ROOT, "models/Qwen2.5-Math-1.5B-Base")
PROMPT_PATH = os.path.join(CURRENT_DIR, "prompts/r1_zero.prompt")
MATH_DATA_PATH = os.path.join(PROJECT_ROOT, "data/gsm8k")

if __name__ == "__main__":
    pass