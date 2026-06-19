#!/usr/bin/env python3
"""第 5 章配套脚本：检查 LoRA 参数量与训练配置。

用法：
  # 方式 1：展示预计算的 v22 实验数据（无需 GPU / 模型下载）
  python scripts/tutorial/ch5_inspect_training.py --sample

  # 方式 2：用真实模型计算（需要 transformers + peft）
  python scripts/tutorial/ch5_inspect_training.py --model Qwen/Qwen3-4B-Thinking-2507 --rank 8

本脚本展示 LoRA 的"省钱魔法"：4B 全模型冻结，只训练 ~17 MB 参数。
同时展示训练配置（batch_size、mini_batch_size、lr）的关键细节。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# v22 实验预计算数据（从真实模型 config 精确计算）
V22_STATS = {
    "model": "Qwen/Qwen3-4B-Thinking-2507",
    "lora_rank": 8,
    "total_params": 4_057_518_080,
    "trainable_params": 15_409_152,
    "trainable_ratio": 0.00380,
    "lora_layers": [
        # (name, A_shape, B_shape, params)
        ("model.layers.0.self_attn.q_proj", "2560×8", "8×2560", 40960),
        ("model.layers.0.self_attn.k_proj", "2560×8", "8×640", 25600),
        ("model.layers.0.self_attn.v_proj", "2560×8", "8×640", 25600),
        ("model.layers.0.self_attn.o_proj", "2560×8", "8×2560", 40960),
        ("model.layers.0.mlp.gate_proj", "2560×8", "8×9728", 98304),
        ("model.layers.0.mlp.up_proj", "2560×8", "8×9728", 98304),
        ("model.layers.0.mlp.down_proj", "9728×8", "8×2560", 98304),
    ],
    "num_layers": 36,
    "hidden_size": 2560,
    "intermediate_size": 9728,
    "num_attention_heads": 32,
    "num_kv_heads": 8,
    # Training config
    "training_config": {
        "batch_size": 64,
        "prompts_per_step": 8,
        "G (repeat_times)": 8,
        "mini_batch_size": 9999,
        "lr": 5e-6,
        "lr_scheduler": "constant",
        "ppo_clip_epsilon": 0.2,
        "kl_coef": 0.0,
        "total_steps": 19,
    },
    # Timeline
    "timeline": {
        "rollout_minutes": 17,
        "trainer_minutes": 10,
        "sync_minutes": 1,
        "total_per_step_minutes": 28,
        "total_19_steps_hours": 8.9,
    },
}


def print_sample():
    """展示预计算数据。"""
    s = V22_STATS

    print(f"\n{'='*60}")
    print(f"  LoRA Training Inspection (v22 实验数据)")
    print(f"{'='*60}\n")

    # Section 1: Model & LoRA
    print("  ┌─────────────────────────────────────────────────────────┐")
    print("  │ Section 1: LoRA 参数量                                  │")
    print("  └─────────────────────────────────────────────────────────┘")
    print()
    print(f"    Model: {s['model']}")
    print(f"    LoRA rank: {s['lora_rank']}")
    print(f"    Total parameters: {s['total_params']:,} ({s['total_params']/1e9:.1f}B)")
    print(f"    Trainable (LoRA): {s['trainable_params']:,} (~{s['trainable_params']*2/1e6:.0f} MB in fp16)")
    print(f"    Trainable ratio:  {s['trainable_ratio']*100:.3f}%")
    print()
    print(f"    → Base 4B 参数永远不动。每步 RL 更新的就是这 ~{s['trainable_params']*2/1e6:.0f} MB。")
    print()

    # Section 2: Per-layer breakdown
    print("  ┌─────────────────────────────────────────────────────────┐")
    print("  │ Section 2: Per-layer LoRA 结构（第 0 层示例）           │")
    print("  └─────────────────────────────────────────────────────────┘")
    print()
    print(f"    LoRA 原理: ΔW = A · B, A∈R^(d×r), B∈R^(r×d)")
    print(f"    rank={s['lora_rank']}, 远小于 hidden_size={s['hidden_size']}")
    print()
    for name, a_shape, b_shape, params in s["lora_layers"]:
        print(f"    {name}")
        print(f"      A=[{a_shape}], B=[{b_shape}] → {params:,} params")
    print(f"    ...")
    print(f"    (共 {s['num_layers']} 层，每层结构相同)")
    print()

    # Section 3: Training config
    print("  ┌─────────────────────────────────────────────────────────┐")
    print("  │ Section 3: 训练配置                                     │")
    print("  └─────────────────────────────────────────────────────────┘")
    print()
    cfg = s["training_config"]
    print(f"    batch_size:       {cfg['batch_size']} ({cfg['prompts_per_step']} prompts × G={cfg['G (repeat_times)']} trajectories)")
    print(f"    mini_batch_size:  {cfg['mini_batch_size']} (≥ 64 → 整 batch 单次 optim_step)")
    print(f"    lr:               {cfg['lr']} (constant scheduler)")
    print(f"    PPO clip epsilon: {cfg['ppo_clip_epsilon']}")
    print(f"    KL penalty:       {cfg['kl_coef']} (教学版关掉)")
    print(f"    total_steps:      {cfg['total_steps']}")
    print()
    print(f"    为什么 mini_batch_size=9999？")
    print(f"      → 确保走「整 batch 路径」：1 次 forward_backward + 1 次 optim_step")
    print(f"      → 避免 mini-batch SGD 的 intra-step off-policy drift（见教程 §5.3）")
    print()

    # Section 4: Timeline
    print("  ┌─────────────────────────────────────────────────────────┐")
    print("  │ Section 4: 单步训练时间线                               │")
    print("  └─────────────────────────────────────────────────────────┘")
    print()
    t = s["timeline"]
    total = t["total_per_step_minutes"]
    print(f"    t=0      explorer 开始 rollout 64 条 trajectory")
    print(f"              │ {t['rollout_minutes']} min")
    print(f"    t={t['rollout_minutes']}     rollout 完成 → 算 advantage → 推到 trainer")
    print(f"              │ {t['trainer_minutes']} min")
    print(f"    t={t['rollout_minutes']+t['trainer_minutes']}     trainer 完成 forward_backward + optim_step")
    print(f"              │ {t['sync_minutes']} min")
    print(f"    t={total}     weight sync 完成（仅传 17 MB）→ 下一步开始")
    print()
    print(f"    {cfg['total_steps']} steps × {total} min/step ≈ {t['total_19_steps_hours']:.1f} 小时")
    print()

    # Section 5: 健康度三件套
    print("  ┌─────────────────────────────────────────────────────────┐")
    print("  │ Section 5: 训练健康度监控                               │")
    print("  └─────────────────────────────────────────────────────────┘")
    print()
    print("    v22 实验全程健康值（trainer.log）：")
    print()
    print("    ┌───────────────┬──────────────┬──────────────┬─────────────────────┐")
    print("    │ 指标          │ 健康范围     │ v22 实测     │ 异常时怎么办        │")
    print("    ├───────────────┼──────────────┼──────────────┼─────────────────────┤")
    print("    │ ppo_kl        │ < 0.030      │ 0.005–0.011  │ > 0.03: 降 lr       │")
    print("    │ pg_clipfrac   │ < 0.050      │ 0.008–0.014  │ > 0.10: 降 lr       │")
    print("    │ reward_std/min│ > 0.050      │ 0.118–0.502  │ < 0.05: 增大 G      │")
    print("    └───────────────┴──────────────┴──────────────┴─────────────────────┘")
    print()


def print_live(model_path: str, rank: int):
    """用真实模型计算 LoRA 参数量。"""
    try:
        from transformers import AutoModelForCausalLM, AutoConfig
    except ImportError:
        print("错误: 需要安装 transformers。执行:")
        print("  pip install transformers")
        print("\n或者使用 --sample 查看预计算数据")
        return 1

    print(f"\n加载模型配置: {model_path} ...")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    hidden = config.hidden_size
    intermediate = getattr(config, "intermediate_size", hidden * 4)
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)

    # 估算 LoRA 参数量
    head_dim = hidden // num_heads
    q_dim = hidden
    k_dim = num_kv_heads * head_dim
    v_dim = k_dim
    o_dim = hidden

    # 每层 LoRA 参数: (in + out) * rank for each target
    targets = [
        ("q_proj", q_dim, hidden),
        ("k_proj", k_dim, hidden),
        ("v_proj", v_dim, hidden),
        ("o_proj", hidden, hidden),
        ("gate_proj", intermediate, hidden),
        ("up_proj", intermediate, hidden),
        ("down_proj", hidden, intermediate),
    ]

    lora_per_layer = sum((out_d + in_d) * rank for _, out_d, in_d in targets)
    total_lora = lora_per_layer * num_layers

    # 估算总参数量
    total_params = sum(p for p in [
        hidden * config.vocab_size,  # embedding
        num_layers * (
            q_dim * hidden + k_dim * hidden + v_dim * hidden + hidden * hidden +  # attn
            intermediate * hidden * 3  # mlp
        ),
    ])

    print(f"\n{'='*60}")
    print(f"  LoRA Training Inspection (live)")
    print(f"{'='*60}\n")
    print(f"  Model: {model_path}")
    print(f"  Hidden size: {hidden}, Layers: {num_layers}")
    print(f"  LoRA rank: {rank}")
    print(f"  Estimated total params: ~{total_params/1e9:.1f}B")
    print(f"  LoRA trainable params: {total_lora:,} (~{total_lora*2/1e6:.0f} MB in fp16)")
    print(f"  Trainable ratio: {total_lora/total_params*100:.3f}%")
    print()

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="第 5 章：检查 LoRA 参数量与训练配置"
    )
    parser.add_argument(
        "--sample", action="store_true",
        help="展示预计算的 v22 实验数据（无需模型下载）"
    )
    parser.add_argument(
        "--model", type=str, default="",
        help="模型路径（需要 transformers）"
    )
    parser.add_argument(
        "--rank", type=int, default=8,
        help="LoRA rank（默认: 8）"
    )
    args = parser.parse_args()

    if args.sample:
        print_sample()
    elif args.model:
        return print_live(args.model, args.rank)
    else:
        parser.print_help()
        print("\n提示: 使用 --sample 查看预计算数据（推荐），或 --model 用真实模型计算")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
