# trl-lab：用 HuggingFace TRL 在本地 2×3090 上学 RL 算法

本目录是 [`agentic-rl-lab`](../README.md) 的**本地训练版**：保留原仓库的算法理论与 reward 口径，
但把付费的 PyTRIO 远程训练替换成社区官方框架 **HuggingFace TRL**，
让算法理论直接对应到 TRL 的一个配置开关，方便学习与复现。

> 原仓库的 `pytrio` 只是远程 REST 客户端，训练发生在它自己的服务器上；
> 而各章真正的算法逻辑（数据、reward、group-relative advantage、loss 选择）都在脚本里。
> 本目录用 `trl.GRPOTrainer` 复刻这些算法。

---

## 1. 环境（复用已有 GPU torch，避免重复下载）

本目录**不**用 uv 建独立环境，而是复用你已有的 conda 环境 `haze_classify`
（里面是 `torch 2.5.1+cu121`，两张 3090 都能识别），只补装 TRL：

```bash
# 确认 GPU 可用
~/miniconda3/envs/haze_classify/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"

# 只补装 TRL 相关小包（不会升级 torch/transformers）
~/miniconda3/envs/haze_classify/bin/python -m pip install \
  --index-url https://repo.huaweicloud.com/repository/pypi/simple \
  "trl==1.14.0" "peft==0.21.0" "math-verify>=0.9.0"
```

---

## 2. 算法 ↔ TRL 映射（核心）

TRL 1.14 的 `GRPOConfig` 原生支持这些损失/变体，无需自己写训练循环：

| 本仓库章节 | 算法 | TRL 配置 |
| --- | --- | --- |
| 01-grpo | GRPO | `loss_type="grpo"`, `scale_rewards="group"` |
| 06-dapo | DAPO | `loss_type="dapo"`, `epsilon_high=0.28`, `mask_truncated_completions=True` |
| 07-gspo | GSPO | `loss_type="grpo"`, `importance_sampling_level="sequence"`, `epsilon=3e-4`, `epsilon_high=4e-4` |
| 00/01 | CISPO | `loss_type="cispo"` |
| DR.GRPO | 去标准差缩放 | `loss_type="dr_grpo"`, `scale_rewards="none"` |
| BNPO | 局部 batch 归一化 | `loss_type="bnpo"` |

### 对应的数学（与 `00-loss-function/readme.md` 一致）

设重要性比率 `ratio = exp(log π_θ - log π_old)`，优势 `Â`：

* **GRPO**（组内相对优势）：
  `Â_i = (r_i - mean(r)) / (std(r)+ε)`，损失 `-E[min(ratio·Â, clip(ratio,1-ε,1+ε)·Â)]`
* **DAPO** 四项改进 → TRL 开关：
  1. Clip-Higher → `epsilon_high=0.28`
  2. Dynamic Sampling → 数据/采样层控制（见第 5 节局限）
  3. Token-level Loss → `loss_type="dapo"`（按全局有效 token 归一化）
  4. Soft Overlong Punishment → `trl_lab/rewards.py` 的 `DAPOReward`
* **GSPO**（sequence-level ratio）：
  `s_i = exp( (1/|y_i|) Σ_t log(π_θ/π_old) )`，每个回答只有 **1 个** ratio / 1 次裁剪
  → `importance_sampling_level="sequence"`
* **CISPO**：`-E[ sg(clip(ratio)) · log π_θ · Â ]`，裁剪后的 ratio 被 stop-gradient
  → `loss_type="cispo"`

---

## 3. 目录结构

```
trl-lab/
├── train.py                 # 统一入口：--algorithm 选算法
├── pyproject.toml           # 依赖声明（仅文档作用，实际用 conda 环境）
├── trl_lab/
│   ├── data.py              # GSM8K / DAPO-Math / dummy 数据 + prompt 构造
│   ├── rewards.py           # boxed / math_verify / DAPO 软超长惩罚
│   └── presets.py           # 算法 → GRPOConfig 映射（本表的代码实现）
└── tools/
    └── ms_download.py       # ModelScope 多线程下载器（绕开单连接限速）
```

---

## 4. 快速开始

所有命令在 `trl-lab/` 目录下执行。

### 4.1 离线冒烟（无需下载模型，用 Qwen3-0.6B 验证管线）

```bash
~/miniconda3/envs/haze_classify/bin/python tools/ms_download.py \
    --repo Qwen/Qwen3-0.6B --out ~/models/Qwen3-0.6B

HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 ~/miniconda3/envs/haze_classify/bin/python train.py \
    --algorithm grpo --dataset dummy --model ~/models/Qwen3-0.6B \
    --limit 4 --num-generations 4 --batch-size 2 --max-completion-length 128 --max-steps 2
```

### 4.2 下载 Qwen3.5-4B（约 8.7GB，多线程约 20 分钟）

```bash
~/miniconda3/envs/haze_classify/bin/python tools/ms_download.py \
    --repo Qwen/Qwen3.5-4B --out ~/models/Qwen3.5-4B --workers 48
```

### 4.3 GSM8K 上跑 GRPO / DAPO / GSPO

```bash
# 首次加载 GSM8K 需要联网，用国内镜像
export HF_ENDPOINT=https://hf-mirror.com

# GRPO
CUDA_VISIBLE_DEVICES=0 ~/miniconda3/envs/haze_classify/bin/python train.py \
    --algorithm grpo --dataset gsm8k --model ~/models/Qwen3.5-4B \
    --limit 1000 --num-generations 8 --batch-size 4 \
    --max-completion-length 512 --max-steps 200 --learning-rate 1e-5

# 换算法只改 --algorithm
... --algorithm dapo ...
... --algorithm gspo ...
... --algorithm cispo ...
```

> 两卡训练（可选）：`accelerate launch --num_processes 2 train.py ...`。
> 单卡 4B + LoRA + bf16 约需 16–20GB 显存；显存紧张时调小
> `--batch-size` / `--max-completion-length`，或加 `--grad-accum 2`。

### 4.4 显存不够怎么办（本机实测经验）

本机两张 3090 常被其他任务占用（实测可能只剩 10–14GB/卡），此时 4B + 8×512
的默认配置会 OOM。可用的降显存手段：

```bash
# 1) 减少每步 prompt 数和采样数、缩短回答长度
--batch-size 1 --num-generations 2 --max-completion-length 128

# 2) 缓解显存碎片
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 3) 选空闲的那张卡（先看占用）
nvidia-smi
CUDA_VISIBLE_DEVICES=1 ...
```

> 注意：如果卡上在跑你自己的其他任务，训练抢占显存可能把对方搞崩，
> 建议等空闲或换 `--model` 用更小的模型。

---

## 5. 与 agentic-rl-lab 的差异与局限

| 主题 | 说明 |
| --- | --- |
| Loss | TRL 内建 `grpo/dapo/cispo/...`，等价于原仓库 `forward_backward` 的 loss_fn |
| 数据 | 原仓库 DAPO/GSPO 用 DAPO-Math-17k，这里默认 GSM8K（小、快），可切 `--dataset dapo_math` |
| Dynamic Sampling | TRL 未内建 DAPO 的动态采样，需要自定义 sampler；`--dataset` 层的加权采样可近似 |
| 多轮 Agentic | Search-R1 / ReTool / ALFWorld / TEMPO / AgentOPSD 涉及环境交互，需用 `GRPOTrainer(environment_factory=...)` 或 verl/SkyRL，不在本目录范围 |
| 多模态 | Vision GRPO / Spec-o3 需要 VLM trainer；Qwen3.5-4B 本身是多模态模型 |
| 蒸馏 | OPD / OPSD 是 reverse-KL 蒸馏，TRL 有 `GKDTrainer` 可对照，本目录暂未接入 |
| 日志 | 原仓库用 SwanLab；本目录用 `report_to="none"`（终端日志），可改成 wandb/trackio |

---

## 6. 后续可扩展

* 把 `GRPOTrainer` 换成 `GKDTrainer` 复现 02-opd / 04-opsd 的 reverse-KL 蒸馏。
* 用 `GRPOTrainer(environment_factory=...)` 接 ReTool 的代码沙箱 / Search-R1 的检索工具。
* 用 `trl.experimental` 的 VLM GRPO 复现 09-vision-grpo。
