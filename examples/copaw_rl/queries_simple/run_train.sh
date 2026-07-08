#!/bin/bash
# run_train.sh — queries_simple GRPO RL 训练（基于 v20_top8 的 8 个 RL 候选 task）
#
# 数据集：
#   data/v20_top8/tasks.json  — v22 / v28 / v32 训练 yaml 默认 taskset_path
#   data/v28eval_top8x8ckpt/tasks.json  — v28eval 评估 yaml 默认 taskset_path
#
# 用法：
#   bash examples/copaw_rl/queries_simple/run_train.sh
# 监控：
#   tail -f /tmp/trinity_queries_simple_train.log
#   tensorboard --logdir checkpoints/CoPaw-Pro-RL/<group>/<name>/monitor

set -e

# ---- sandbox 外环境变量（与 run_bench.sh 同源）----
# OSS_* and DASHSCOPE_API_KEY are unused in the queries_simple tutorial path
# (run_simple_workflow does not depend on OSS; agent uses the RL model, not DashScope).
# Exported as empty defaults for backward compatibility with the legacy run.py path.
export OSS_ACCESS_KEY_ID="${OSS_ACCESS_KEY_ID:-}"
export OSS_ACCESS_KEY_SECRET="${OSS_ACCESS_KEY_SECRET:-}"
export OSS_REGION=cn-beijing
export OSS_ENDPOINT="https://oss-cn-beijing-internal.aliyuncs.com"
export OSS_BUCKET_NAME="copaw-dataset"
export OSS_PREFIX="data/0318_train_tasks_compressed/"
export DASHSCOPE_API_KEY="${DASHSCOPE_API_KEY:-}"
export E2B_DOMAIN="sandbox01.vpc.cn-hongkong.pai-eas.aliyuncs.com"
export E2B_TEMPLATE="agentscope-qwenpaw-0518"
export E2B_API_KEY="${E2B_API_KEY:?E2B_API_KEY must be set}"

# ---- TuFT base url（host VPC IP，让 sandbox 能访问）----
HERE="$(cd "$(dirname "$0")" && pwd)"
TRINITY_ROOT="$(cd "$HERE/../../.." && pwd)"
if [ -z "${TINKER_BASE_URL:-}" ]; then
  _IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
  export TINKER_BASE_URL="http://${_IP:-localhost}:10610"
fi
export TINKER_API_KEY="${TINKER_API_KEY:-tml-tuft-dev-key}"
echo "[train] TINKER_BASE_URL=$TINKER_BASE_URL"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# ---- 专用 Ray head（与 TuFT 隔离） ----
TRINITY_VENV="$TRINITY_ROOT/.venv"
TRINITY="$TRINITY_VENV/bin/trinity"
RAY_BIN="$TRINITY_VENV/bin/ray"
[ -x "$TRINITY" ]   || TRINITY=trinity
[ -x "$RAY_BIN" ]   || { echo "[train] ERROR: ray not found at $RAY_BIN"; exit 1; }

TRINITY_RAY_TMPDIR="${TRINITY_RAY_TMPDIR:-/tmp/ray-trinity-train}"
TRINITY_RAY_PORT="${TRINITY_RAY_PORT:-6392}"
TRINITY_RAY_DASHBOARD_PORT="${TRINITY_RAY_DASHBOARD_PORT:-8270}"
mkdir -p "$TRINITY_RAY_TMPDIR"
echo "[train] starting dedicated Ray head at 127.0.0.1:$TRINITY_RAY_PORT"
pkill -KILL -f "$TRINITY_RAY_TMPDIR" 2>/dev/null || true
sleep 1
"$RAY_BIN" start --head \
  --port="$TRINITY_RAY_PORT" \
  --dashboard-port="$TRINITY_RAY_DASHBOARD_PORT" \
  --temp-dir="$TRINITY_RAY_TMPDIR" \
  --num-gpus=0 \
  --include-dashboard=false \
  --disable-usage-stats >"${TRINITY_RAY_TMPDIR}/ray_head.log" 2>&1
export RAY_ADDRESS="127.0.0.1:${TRINITY_RAY_PORT}"
export RAY_TMPDIR="$TRINITY_RAY_TMPDIR"
echo "[train] RAY_ADDRESS=$RAY_ADDRESS"

# ---- 实验元数据 ----
export EXP_DATE=${EXP_DATE:-$(date +%m%d)}
export EXP_SOURCE=${EXP_SOURCE:-queries_simple_rl}
export EXP_MODEL_NAME="${EXP_MODEL_NAME:-Qwen/Qwen3-4B-Thinking-2507}"
export EXP_NAME=${EXP_NAME:-${EXP_MODEL_NAME//\//-}-rl-queries_simple-${EXP_DATE}}

# 默认训练参数（可被 env 覆盖）
export BATCH_SIZE=${BATCH_SIZE:-8}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
export TOTAL_STEPS=${TOTAL_STEPS:-40}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-40}
export EVAL_INTERVAL=${EVAL_INTERVAL:-3}
export SAVE_INTERVAL=${SAVE_INTERVAL:-1}  # V17: 每步保存 full checkpoint（旧 checkpoint 自动删除）
export RUNNER_PER_MODEL=${RUNNER_PER_MODEL:-64}
export ENGINE_NUM=${ENGINE_NUM:-1}
export LR=${LR:-1e-6}
export LR_SCHEDULER_TYPE=${LR_SCHEDULER_TYPE:-constant}
export LR_WARMUP_STEPS_RATIO=${LR_WARMUP_STEPS_RATIO:-0.0}
export SP_SIZE=${SP_SIZE:-1}
export MONITOR_TYPE=${MONITOR_TYPE:-tensorboard}

# ---- v10: trajectory-level Experience（一条 trajectory = 1 Datum）----
# 1 trajectory = 1 Experience。优势：
#   - 训练 token 总量 -14×（避免同一 trajectory 的历史被重复 forward）
#   - 每 explore step 产 64 经验 ≈ train_batch_size → staleness=1
# 设 USE_TRAJECTORY_EXPERIENCE=0 可回退到旧 step-level。
export USE_TRAJECTORY_EXPERIENCE=${USE_TRAJECTORY_EXPERIENCE:-1}
echo "[train] USE_TRAJECTORY_EXPERIENCE=$USE_TRAJECTORY_EXPERIENCE"

# ---- length / time penalty (workflow 侧) ----
# 单次 LLM forward 可安全处理的总 token 上限。单次超过会被截断；
# trajectory 中有任何 step 超过则该条轨迹被判 over_length，reward=0。
export MAX_STEP_TOKENS=${MAX_STEP_TOKENS:-32768}
# 沙箱内 agent 硬超时（signal.alarm），决定单条 trajectory 物理最长 wall time。
export MAX_AGENT_SECONDS=${MAX_AGENT_SECONDS:-1200}
# Trajectory 总 agent_call_seconds 上限。超过判 over_time，reward=0。
# 默认与沙箱硬超时 MAX_AGENT_SECONDS 对齐——agent_call_seconds 物理上不
# 会超过沙箱硬超时，所以 over_time 阈值天然取这个值；如要更严格惩罚长
# trajectory，可显式 export MAX_TRAJ_SECONDS 覆盖（如 600）。
export MAX_TRAJ_SECONDS=${MAX_TRAJ_SECONDS:-${MAX_AGENT_SECONDS}}
echo "[train] MAX_STEP_TOKENS=$MAX_STEP_TOKENS MAX_AGENT_SECONDS=$MAX_AGENT_SECONDS MAX_TRAJ_SECONDS=$MAX_TRAJ_SECONDS"

export TRINITY_QUERIES_SIMPLE_TASKSET_PATH="$TRINITY_ROOT/examples/copaw_rl/queries_simple/data"
export TRINITY_QUERIES_SIMPLE_TEST_CASES_DIR="$TRINITY_ROOT/examples/copaw_rl/queries_simple/test_cases"

# 1) sanity check: 验证 train tasks 路径可读（yaml 默认指向 v20_top8）
TRAIN_TASKS_PATH="${TRAIN_TASKS_PATH:-$HERE/data/v20_top8/tasks.json}"
if [ -f "$TRAIN_TASKS_PATH" ]; then
  echo "[train] using pre-built tasks ($(cat "$TRAIN_TASKS_PATH" | python3 -c 'import sys,json;print(len(json.load(sys.stdin)))') tasks at $TRAIN_TASKS_PATH)"
else
  echo "[train] note: $TRAIN_TASKS_PATH not found; relying on yaml path"
fi

CONFIG="${CONFIG_OVERRIDE:-$TRINITY_ROOT/examples/copaw_rl/queries_simple.train.tinker.ch1_repro.yaml}"
PLUGIN_DIR="$TRINITY_ROOT/examples/copaw_rl/workflows"
LOG_FILE="${LOG_FILE:-/tmp/trinity_queries_simple_train.log}"
: > "$LOG_FILE"

cleanup() {
  echo "[train] stopping dedicated Ray head"
  pkill -INT -f "$TRINITY_RAY_TMPDIR" 2>/dev/null || true
  sleep 2
  pkill -KILL -f "$TRINITY_RAY_TMPDIR" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[train] launching: $TRINITY run --config $CONFIG --plugin-dir $PLUGIN_DIR"
echo "[train] log -> $LOG_FILE"
"$TRINITY" run --config "$CONFIG" --plugin-dir "$PLUGIN_DIR" 2>&1 | tee "$LOG_FILE"
RC=${PIPESTATUS[0]}
echo "[train] trinity run exit code=$RC"
exit "$RC"
