"""数据集构造：把 GSM8K 等数据处理成 TRL GRPOTrainer 需要的格式。

TRL 要求训练集至少有 ``prompt`` 列：
  * conversational 形式：``prompt = [{"role": "user", "content": "..."}]``，
    TRL 会用 tokenizer 的 chat template 渲染并加 generation prompt；
  * 额外列（如 ``answer``）会作为 kwargs 透传给 reward 函数。

本模块同时提供 ``dummy`` 数据集，便于完全离线地跑通训练管线。
"""

from __future__ import annotations

import re
from typing import Any

from datasets import Dataset, load_dataset

QUESTION_SUFFIX = "Provide a numerical answer without units, written inside \\boxed{}."

# 与仓库 01-grpo/03-grpo.py 一致的一发 few-shot 示例。
FEWSHOT_PREFIX: list[dict[str, str]] = [
    {"role": "user", "content": "How many r's are in strawberry?" + QUESTION_SUFFIX},
    {
        "role": "assistant",
        "content": (
            "<think>\n\n</think>\n\n"
            "Let's spell the word out and number all the letters: "
            "1) s 2) t 3) r 4) a 5) w 6) b 7) e 8) r 9) r 10) y. "
            "We have r's at positions 3, 8, and 9. "
            "There are three r's. \\boxed{3}"
        ),
    },
]


def _normalize_answer(text: str) -> str:
    return text.replace(",", "").strip().rstrip(".")


def _extract_gsm8k_answer(answer_text: str) -> str:
    """GSM8K 的标准答案在 ``####`` 之后。"""
    match = re.search(r"####\s*(.+)", answer_text)
    return _normalize_answer(match.group(1) if match else answer_text)


def build_gsm8k(split: str = "train", limit: int | None = None) -> Dataset:
    """加载 GSM8K，返回含 ``prompt`` / ``answer`` 的数据集。"""
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    def to_prompt(example: dict[str, Any]) -> dict[str, Any]:
        messages = [
            *FEWSHOT_PREFIX,
            {"role": "user", "content": example["question"] + QUESTION_SUFFIX},
        ]
        return {"prompt": messages, "answer": _extract_gsm8k_answer(example["answer"])}

    return dataset.map(to_prompt, remove_columns=dataset.column_names)


def build_dapo_math(split: str = "train", limit: int | None = None) -> Dataset:
    """加载 DAPO-Math-17k（05-retool / 06-dapo / 07-gspo 用的数据集）。

    该数据集列结构在不同版本间有差异，这里兼容 ``answer`` 与
    ``reward_model.ground_truth`` 两种写法。
    """
    dataset = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split=split)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    def to_prompt(example: dict[str, Any]) -> dict[str, Any]:
        prompt = example.get("prompt") or example.get("question") or ""
        if isinstance(prompt, list):
            messages = prompt
        else:
            messages = [{"role": "user", "content": prompt}]
        gold = example.get("answer")
        if gold is None:
            reward_model = example.get("reward_model") or {}
            gold = reward_model.get("ground_truth", "")
        return {"prompt": messages, "answer": str(gold)}

    return dataset.map(to_prompt, remove_columns=dataset.column_names)


def build_dummy(limit: int = 16) -> Dataset:
    """完全离线的小数据集，用于验证训练管线能跑通（reward 恒为 0/1）。"""
    rows = []
    for i in range(limit):
        a, b = 1 + i % 9, 2 + i % 7
        rows.append(
            {
                "prompt": [
                    {
                        "role": "user",
                        "content": f"What is {a} + {b}?{QUESTION_SUFFIX}",
                    }
                ],
                "answer": str(a + b),
            }
        )
    return Dataset.from_list(rows)


DATASET_REGISTRY = {
    "gsm8k": build_gsm8k,
    "dapo_math": build_dapo_math,
    "dummy": build_dummy,
}


def build_dataset(name: str, split: str = "train", limit: int | None = None) -> Dataset:
    if name not in DATASET_REGISTRY:
        raise ValueError(f"未知数据集：{name}，可选 {sorted(DATASET_REGISTRY)}")
    if name == "dummy":
        return build_dummy(limit=limit or 16)
    return DATASET_REGISTRY[name](split=split, limit=limit)
