"""统一训练入口：用 TRL ``GRPOTrainer`` 在本地复现各章 RL 算法。

示例
----
    # 冒烟测试（离线数据，单卡）
    CUDA_VISIBLE_DEVICES=0 python train.py --algorithm grpo --dataset dummy \
        --model Qwen/Qwen3.5-4B --limit 8 --num-generations 4 \
        --batch-size 2 --max-completion-length 128 --max-steps 2

    # GSM8K 上的 GRPO
    python train.py --algorithm grpo --dataset gsm8k --limit 1000 \
        --model Qwen/Qwen3.5-4B --num-generations 8 --batch-size 4 \
        --max-completion-length 512 --max-steps 200

    # 换算法只改 --algorithm
    python train.py --algorithm dapo ...
    python train.py --algorithm gspo ...
    python train.py --algorithm cispo ...

两卡训练（可选）：``accelerate launch --num_processes 2 train.py ...``。
"""

from __future__ import annotations

import argparse
from typing import Any

import torch
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

from trl_lab.data import build_dataset
from trl_lab.presets import PRESETS, get_preset
from trl_lab.rewards import REWARD_REGISTRY, DAPOReward

# Qwen 系列 LoRA 常见目标模块（attention + MLP）。
QWEN_LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="用 TRL 在本地复现 agentic-rl-lab 算法")
    parser.add_argument("--algorithm", choices=sorted(PRESETS), default="grpo")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--dataset", choices=["gsm8k", "dapo_math", "dummy"], default="gsm8k")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=None, help="只用前 N 条数据（冒烟用）")

    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=4, help="每步 prompt 数")
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--num-generations", type=int, default=8, help="每 prompt 采样数 G")
    parser.add_argument("--max-completion-length", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=None, help="覆盖 preset 的 KL 系数")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-lora", action="store_true", help="全参训练（显存需求大）")
    parser.add_argument("--enable-thinking", action="store_true", help="开启 Qwen 思考模式")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> tuple[GRPOConfig, str, str]:
    preset = get_preset(args.algorithm)
    output_dir = args.output_dir or f"outputs/{args.algorithm}"

    kwargs: dict[str, Any] = dict(preset.grpo_kwargs)
    if args.beta is not None:
        kwargs["beta"] = args.beta

    config = GRPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size * args.num_generations,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        top_p=args.top_p,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        bf16=True,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to="none",
        seed=args.seed,
        log_completions=True,
        chat_template_kwargs={"enable_thinking": args.enable_thinking},
        model_init_kwargs={"dtype": torch.bfloat16},
        **kwargs,
    )
    return config, preset.reward, output_dir


def build_reward(name: str, args: argparse.Namespace) -> Any:
    if name == "dapo":
        return DAPOReward(max_completion_length=args.max_completion_length)
    return REWARD_REGISTRY[name]


def main() -> None:
    args = parse_args()
    config, reward_name, output_dir = build_config(args)
    dataset = build_dataset(args.dataset, split=args.split, limit=args.limit)

    peft_config = None
    if not args.no_lora:
        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank * 2,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=QWEN_LORA_TARGETS,
        )

    print(f"[trl-lab] 算法={args.algorithm}  reward={reward_name}  数据={args.dataset}  样本={len(dataset)}")
    print(f"[trl-lab] output_dir={output_dir}")

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=[build_reward(reward_name, args)],
        args=config,
        train_dataset=dataset,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(output_dir)
    print(f"[trl-lab] 完成，LoRA 权重保存到 {output_dir}")


if __name__ == "__main__":
    main()
