#!/bin/bash
# _start_v22.sh — 启动 V22 RL 训练（Agentic-RL 教程 baseline）
#
# 训练参数：
#   - mini_batch_size: 9999（trinity tinker_trainer 走整 batch + 1 次 optim_step 路径）
#     ⇒ TuFT FSDP backend 内部按 ModelConfig.micro_batch_size 切 micro-batch 累积梯度
#   - lr: 5e-6（square root scaling for grad-accum）
#   - LORA_RANK=8
#
# 启动: bash scripts/_start_v22.sh
# 监控: tail -f /tmp/trinity_v22_train.top.log
#
# 关键监控指标（前 10 step 必看）：
#   actor/ppo_kl                      — 期望 0.005-0.008；>0.03 立即 kill 降 lr 到 3e-6
#   actor/pg_clipfrac                 — 期望 <0.005
#   rollout/score/mean   step10       — 期望 ≥0.78
#   actor/num_micro_batches           — 期望 32 (FSDP backend 真在切 micro-batch 累积)
#                                       ≠32 说明 yaml mini_batch_size 没生效

set -e

HERE="$(cd "$(dirname "$0")" && pwd)"
TRINITY_ROOT="$(cd "$HERE/.." && pwd)"

# ---- 实验元数据 ----
export EXP_DATE=0608
export EXP_SOURCE=v22-grad-accum
export EXP_MODEL_NAME="Qwen/Qwen3-4B-Thinking-2507"
export EXP_NAME="Qwen-Qwen3-4B-Thinking-2507-rl-queries_simple-0608-v22-grad-accum"

# ---- 训练参数 ----
export BATCH_SIZE=8
export TRAIN_BATCH_SIZE=64
export TOTAL_STEPS=19
export TOTAL_EPOCHS=19
export EVAL_INTERVAL=999           # 取消独立 eval（train rollout 即 eval）
export SAVE_INTERVAL=1
export RUNNER_PER_MODEL=64
export ENGINE_NUM=1

# ---- 学习率（square root scaling for grad accumulation）----
export LR=5e-6
export LR_SCHEDULER_TYPE=constant
export LR_WARMUP_STEPS_RATIO=0.0

# ---- mini_batch_size: 9999（整 batch + 1 次 optim_step，让 TuFT FSDP backend 累积梯度）----
export MINI_BATCH_SIZE=9999

export SP_SIZE=1
export USE_TRAJECTORY_EXPERIENCE=1

# ---- LoRA rank ----
export LORA_RANK=8

# ---- prompt 限长补丁 ----
export MAX_STEP_TOKENS=31000

# ---- Wandb 监控 ----
export MONITOR_TYPE=wandb
export WANDB_API_KEY='<your-wandb-api-key>'
export WANDB_BASE_URL='<your-wandb-base-url>'
export WANDB_PROJECT=rl_tutorial

# ---- HF tokenizer 下载源 ----
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0

# ---- 训练日志 ----
export LOG_FILE="/tmp/trinity_v22_train.top.log"

# ---- Ray 集群配置 ----
export TRINITY_RAY_TMPDIR="/tmp/ray-trinity-train-v22"
export TRINITY_RAY_PORT=6394
export TRINITY_RAY_DASHBOARD_PORT=8272

# ---- 指向 V22 专属 yaml ----
export CONFIG_OVERRIDE="$TRINITY_ROOT/examples/copaw_rl/queries_simple.train.tinker.v22.yaml"

echo "[v22] EXP_DATE=$EXP_DATE  EXP_SOURCE=$EXP_SOURCE"
echo "[v22] EXP_NAME=$EXP_NAME"
echo "[v22] LR=$LR  MINI_BATCH_SIZE=$MINI_BATCH_SIZE  LORA_RANK=$LORA_RANK"
echo "[v22] CONFIG=$CONFIG_OVERRIDE"
echo "[v22] LOG_FILE=$LOG_FILE"
echo "[v22] RAY tmpdir=$TRINITY_RAY_TMPDIR port=$TRINITY_RAY_PORT dashboard=$TRINITY_RAY_DASHBOARD_PORT"

cd "$TRINITY_ROOT"
exec bash examples/copaw_rl/queries_simple/run_train.sh
