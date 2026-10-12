"""奖励函数集合，复现 agentic-rl-lab 各章节的 reward 设计。

TRL reward 函数约定
--------------------
    def reward(completions, **kwargs) -> list[float]

* ``completions``：同一个 prompt 采样出的 G 条回答（G = num_generations）。
* ``**kwargs``：会带上数据集里的额外列（例如 ``answer``），以及 TRL 注入的
  ``trainer_state``、``log_extra``、``log_metric`` 等。
* 返回值长度必须与 ``completions`` 相同。

这里的实现刻意与仓库脚本保持一致的判分口径，方便对照：

* ``boxed_reward``      —— 01-grpo / 09-vision-grpo：最后一个 ``\\boxed{}`` 精确匹配，0/1。
* ``math_verify_reward``—— 05-retool / 06-dapo / 07-gspo：``math_verify`` 等价判定，±1。
* ``dapo_reward``       —— 06-dapo：±1 之上叠加 Soft Overlong Punishment（软超长惩罚）。
"""

from __future__ import annotations

import re
from typing import Any

from math_verify import parse as mv_parse
from math_verify import verify as mv_verify

_BOXED_RE = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")
_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _as_text(completion: Any) -> str:
    """把 TRL 传来的 completion 统一成字符串。

    conversational 数据集下 completion 是 ``[{"role": "assistant", "content": ...}]``，
    其他情况直接是字符串。
    """
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts = []
        for message in completion:
            if isinstance(message, dict):
                parts.append(str(message.get("content", "")))
            else:
                parts.append(str(message))
        return "\n".join(parts)
    return str(completion)


def extract_boxed(text: str) -> str | None:
    """取最后一个 ``\\boxed{...}`` 的内容（与仓库脚本一致）。"""
    matches = _BOXED_RE.findall(text)
    if not matches:
        return None
    return matches[-1].strip()


def normalize_answer(text: str) -> str:
    """轻量归一化：去掉千分位逗号和首尾空白/句点。"""
    return text.replace(",", "").strip().rstrip(".")


def _final_number(text: str) -> str | None:
    """退化用：没有 boxed 时取最后出现的数字。"""
    numbers = _NUMBER_RE.findall(text.replace(",", ""))
    return numbers[-1] if numbers else None


def boxed_reward(completions: list[Any], answer: list[str] | None = None, **_: Any) -> list[float]:
    """01-grpo / 09-vision-grpo 的 0/1 精确匹配奖励。"""
    if answer is None:
        raise ValueError("boxed_reward 需要数据集提供 `answer` 列")
    rewards: list[float] = []
    for completion, gold in zip(completions, answer, strict=True):
        predicted = extract_boxed(_as_text(completion))
        if predicted is None:
            rewards.append(0.0)
            continue
        rewards.append(
            1.0 if normalize_answer(predicted) == normalize_answer(str(gold)) else 0.0
        )
    return rewards


def math_verify_reward(
    completions: list[Any], answer: list[str] | None = None, **_: Any
) -> list[float]:
    """05-retool / 07-gspo 的 ``math_verify`` 判定：正确 +1，错误或未作答 -1。"""
    if answer is None:
        raise ValueError("math_verify_reward 需要数据集提供 `answer` 列")
    rewards: list[float] = []
    for completion, gold in zip(completions, answer, strict=True):
        text = _as_text(completion)
        predicted = extract_boxed(text) or _final_number(text)
        try:
            ok = predicted is not None and mv_verify(
                mv_parse(str(gold)), mv_parse(str(predicted))
            )
        except Exception:
            ok = False
        rewards.append(1.0 if ok else -1.0)
    return rewards


class DAPOReward:
    """06-dapo 奖励：``math_verify`` ±1 叠加 Soft Overlong Punishment。

    Soft Overlong Punishment（论文 DAPO 四项改进之一）
    ---------------------------------------------------
    设 ``L = 回答 token 长度``、``Lmax = max_completion_length``、
    ``cache = overlong_cache``，惩罚窗口起点 ``start = Lmax - cache``：

        penalty(L) = 0                                   , L <= start
                   = -(L - start) / cache                , start < L < Lmax
                   = -1                                  , L >= Lmax

    总奖励 ``r = base_reward + penalty``，取值范围从 [-1, 1] 平滑压到 [-2, 1]。
    这里用字符长度近似 token 长度（奖励函数拿不到 token id），
    够教学用；若要严格对齐请改用 tokenizer 计数。
    """

    def __init__(self, max_completion_length: int, overlong_cache: int = 1024) -> None:
        self.max_completion_length = max_completion_length
        self.overlong_cache = overlong_cache
        self.start = max_completion_length - overlong_cache

    def _length_penalty(self, length: int) -> float:
        if length <= self.start:
            return 0.0
        if length >= self.max_completion_length:
            return -1.0
        return -(length - self.start) / self.overlong_cache

    def __call__(
        self,
        completions: list[Any],
        answer: list[str] | None = None,
        **_: Any,
    ) -> list[float]:
        if answer is None:
            raise ValueError("DAPOReward 需要数据集提供 `answer` 列")
        base = math_verify_reward(completions, answer)
        rewards: list[float] = []
        for completion, base_reward in zip(completions, base, strict=True):
            length = len(_as_text(completion))
            rewards.append(base_reward + self._length_penalty(length))
        return rewards


REWARD_REGISTRY: dict[str, Any] = {
    "boxed": boxed_reward,
    "math_verify": math_verify_reward,
    "dapo": DAPOReward,
}
