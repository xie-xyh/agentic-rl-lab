"""算法预设：把 agentic-rl-lab 各章节的算法映射到 TRL ``GRPOConfig``。

这样做的意义是——**算法理论直接对应 TRL 的一个开关**，学习时只要盯住这几个字段：

    loss_type                 聚合/裁剪方式（GRPO / DAPO / CISPO / Dr.GRPO / BNPO ...）
    importance_sampling_level 重要性比率在 token 级还是 sequence 级（GSPO 用 sequence）
    scale_rewards             优势是否按组内标准差缩放（GRPO 缩放，Dr.GRPO 不缩放）
    epsilon / epsilon_high    重要性比率的裁剪区间 [1-eps_low, 1+eps_high]
    beta                      KL 正则系数（0 表示不做 KL 约束）
    mask_truncated_completions 是否丢掉被截断的 completion

TRL 1.14 的 ``loss_type`` 取值：
    'grpo', 'dr_grpo', 'dapo', 'bnpo', 'cispo', 'sapo', 'luspo', 'vespo'
GSPO 没有独立 loss_type，而是 ``importance_sampling_level='sequence'``。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AlgorithmPreset:
    """一个算法的完整定义：TRL 配置 + reward + 参考论文。"""

    name: str
    description: str
    paper: str
    reward: str
    grpo_kwargs: dict[str, Any] = field(default_factory=dict)


PRESETS: dict[str, AlgorithmPreset] = {
    "grpo": AlgorithmPreset(
        name="grpo",
        description="DeepSeekMath 的 GRPO：组内相对优势，token 级重要性比率，无 KL。",
        paper="arXiv:2402.03300",
        reward="boxed",
        grpo_kwargs={
            "loss_type": "grpo",
            "importance_sampling_level": "token",
            "scale_rewards": "group",
            "epsilon": 0.2,
            "epsilon_high": None,
            "beta": 0.0,
        },
    ),
    "dapo": AlgorithmPreset(
        name="dapo",
        description=(
            "DAPO：Clip-Higher（eps_high=0.28）+ token-level 归一化 + 软超长惩罚。"
            "动态采样由数据/采样层控制，见 README。"
        ),
        paper="arXiv:2503.14476",
        reward="dapo",
        grpo_kwargs={
            "loss_type": "dapo",
            "importance_sampling_level": "token",
            "scale_rewards": "group",
            "epsilon": 0.2,
            "epsilon_high": 0.28,
            "beta": 0.0,
            "mask_truncated_completions": True,
        },
    ),
    "gspo": AlgorithmPreset(
        name="gspo",
        description=(
            "GSPO：把重要性比率从 token 级提升到 sequence 级，"
            "用很小的裁剪区间（3e-4 / 4e-4）。"
        ),
        paper="arXiv:2507.18071",
        reward="math_verify",
        grpo_kwargs={
            "loss_type": "grpo",
            "importance_sampling_level": "sequence",
            "scale_rewards": "group",
            "epsilon": 3e-4,
            "epsilon_high": 4e-4,
            "beta": 0.0,
        },
    ),
    "cispo": AlgorithmPreset(
        name="cispo",
        description="CISPO：裁剪的重要性比率被 stop-gradient，只让 logprob 带梯度。",
        paper="MiniMax (CISPO)",
        reward="boxed",
        grpo_kwargs={
            "loss_type": "cispo",
            "importance_sampling_level": "token",
            "scale_rewards": "group",
            "epsilon": 0.2,
            "epsilon_high": None,
            "beta": 0.0,
        },
    ),
    "dr_grpo": AlgorithmPreset(
        name="dr_grpo",
        description="Dr. GRPO：去掉组内标准差缩放，并除以全局常数，消除长度/难度偏置。",
        paper="arXiv:2503.20783",
        reward="boxed",
        grpo_kwargs={
            "loss_type": "dr_grpo",
            "importance_sampling_level": "token",
            "scale_rewards": "none",
            "epsilon": 0.2,
            "epsilon_high": None,
            "beta": 0.0,
        },
    ),
    "bnpo": AlgorithmPreset(
        name="bnpo",
        description="BNPO：只在本地 batch 内对有效 token 归一化。",
        paper="TRL",
        reward="boxed",
        grpo_kwargs={
            "loss_type": "bnpo",
            "importance_sampling_level": "token",
            "scale_rewards": "group",
            "epsilon": 0.2,
            "epsilon_high": None,
            "beta": 0.0,
        },
    ),
}


def get_preset(name: str) -> AlgorithmPreset:
    if name not in PRESETS:
        raise ValueError(f"未知算法：{name}，可选 {sorted(PRESETS)}")
    return PRESETS[name]
