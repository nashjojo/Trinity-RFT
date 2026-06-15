#!/bin/bash
# Tinker / TuFT mode launcher for CoPaw RL.
#
# Differences from rl_train.sh:
#   - inference is served by a local TuFT instance (Tinker-compatible) at
#     $TINKER_BASE_URL via the OpenAI-compatible /oai/api/v1 endpoints
#   - no vLLM is started locally; sandbox containers (qwenpaw) hit TuFT
#     directly through OpenAI protocol, with TuFT auto-loading LoRA via
#     `tinker://...` model paths
#   - Qwen3-4B-Thinking-2507 is text-only; multimodal pipeline is bypassed
#     by `text_only: true` in the yaml so vllm/HF AutoProcessor are skipped
#
# Sandbox 外环境变量：必须由调用者预先 export，否则 fail-fast。
export OSS_ACCESS_KEY_ID="${OSS_ACCESS_KEY_ID:?OSS_ACCESS_KEY_ID must be set}"
export OSS_ACCESS_KEY_SECRET="${OSS_ACCESS_KEY_SECRET:?OSS_ACCESS_KEY_SECRET must be set}"
export OSS_REGION=cn-beijing
export OSS_ENDPOINT="https://oss-cn-beijing-internal.aliyuncs.com"
export OSS_BUCKET_NAME="copaw-dataset"
export OSS_PREFIX="data/0318_train_tasks_compressed/"
export DASHSCOPE_API_KEY="${DASHSCOPE_API_KEY:?DASHSCOPE_API_KEY must be set}"
export E2B_DOMAIN="sandbox01.vpc.cn-hongkong.pai-eas.aliyuncs.com"
export E2B_TEMPLATE="agentscope-qwenpaw-0518"
export E2B_API_KEY="${E2B_API_KEY:?E2B_API_KEY must be set}"

# TuFT (Tinker-compatible) inference service.
# Use the host's internal IP (not localhost) so E2B sandboxes on the VPC
# network can reach TuFT via --provider-base-url.
if [ -z "${TINKER_BASE_URL:-}" ]; then
  _TINKER_HOST_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
  if [ -n "$_TINKER_HOST_IP" ]; then
    export TINKER_BASE_URL="http://${_TINKER_HOST_IP}:10610"
  else
    export TINKER_BASE_URL="http://localhost:10610"
  fi
  unset _TINKER_HOST_IP
fi
export TINKER_API_KEY="${TINKER_API_KEY:-tml-tuft-dev-key}"
echo "[tinker.sh] TINKER_BASE_URL=$TINKER_BASE_URL"

# Resolve trinity / ray binaries: use Trinity-RFT's own venv. We deliberately
# do NOT touch the TuFT venv any more — the two projects communicate purely
# over HTTP (Tinker SDK → TuFT's /oai/api/v1).
TRINITY_VENV="$(dirname "$0")/.venv"
if [ -x "$TRINITY_VENV/bin/trinity" ]; then
  TRINITY="$TRINITY_VENV/bin/trinity"
else
  TRINITY=trinity
fi
RAY_BIN="$TRINITY_VENV/bin/ray"
if [ ! -x "$RAY_BIN" ]; then
  echo "[tinker.sh] ERROR: ray binary not found at $RAY_BIN" >&2
  echo "[tinker.sh] Trinity-RFT venv may be missing. Try: cd $(dirname \"$0\") && uv sync" >&2
  exit 1
fi

# Force HF tokenizer/model resolution to use local cache (no internet in this
# environment). The model snapshot is already under ~/.cache/huggingface/hub.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# Both trinity debug processes (inference_model + workflow) must share a
# single Ray cluster so that workflow can look up the actor named
# ``explorer_rollout_model_0_0``. We start a *dedicated* head node here
# (separate temp dir / ports from any TuFT-managed Ray) and point both
# trinity processes at it via RAY_ADDRESS.
TRINITY_RAY_TMPDIR="${TRINITY_RAY_TMPDIR:-/tmp/ray-trinity-tinker}"
TRINITY_RAY_PORT="${TRINITY_RAY_PORT:-6390}"
TRINITY_RAY_DASHBOARD_PORT="${TRINITY_RAY_DASHBOARD_PORT:-8268}"

mkdir -p "$TRINITY_RAY_TMPDIR"
echo "[tinker.sh] starting dedicated Ray head at 127.0.0.1:$TRINITY_RAY_PORT (temp=$TRINITY_RAY_TMPDIR)"
# Kill any leftover processes from a previous run of *this* dedicated
# cluster (matched by the unique temp-dir path); never touch other Ray
# clusters on the host.
pkill -KILL -f "$TRINITY_RAY_TMPDIR" 2>/dev/null || true
sleep 1
# Do NOT call `ray stop` here — that would also tear down any other Ray
# cluster on this host (e.g. the TuFT-managed one).
"$RAY_BIN" start --head \
  --port="$TRINITY_RAY_PORT" \
  --dashboard-port="$TRINITY_RAY_DASHBOARD_PORT" \
  --temp-dir="$TRINITY_RAY_TMPDIR" \
  --num-gpus=0 \
  --include-dashboard=false \
  --disable-usage-stats >"${TRINITY_RAY_TMPDIR}/ray_head.log" 2>&1
export RAY_ADDRESS="127.0.0.1:${TRINITY_RAY_PORT}"
export RAY_TMPDIR="$TRINITY_RAY_TMPDIR"
echo "[tinker.sh] RAY_ADDRESS=$RAY_ADDRESS"

# Experiment metadata
export EXP_DATE=0419
export EXP_SOURCE=0318_train_tasks
export EXP_MODEL_NAME="Qwen/Qwen3-4B-Thinking-2507"  # TuFT-supported text-only model
export BATCH_SIZE=32
export TRAIN_BATCH_SIZE=1024
export EXP_NAME=${EXP_MODEL_NAME//\//-}-rl-${EXP_DATE}-tinker-v1
export SP_SIZE=1
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LR_SCHEDULER_TYPE=constant
export LR_WARMUP_STEPS_RATIO=0.0
export ENGINE_NUM=1

CONFIG=examples/copaw_rl/copaw_eval_workflow.tinker.yaml
PLUGIN_DIR=examples/copaw_rl/workflows
OUTPUT_DIR=/tmp/trinity_tinker_debug

# Trinity's `trinity debug --module workflow` *attaches* to an existing
# inference_model actor instead of creating one, so we have to spin one up
# first in the background, wait for it to be ready, then run the workflow
# debug in the foreground, and finally clean up.

IM_LOG="${TRINITY_TINKER_IM_LOG:-/tmp/trinity_tinker_im.log}"
WF_LOG="${TRINITY_TINKER_WF_LOG:-/tmp/trinity_tinker_wf.log}"

: > "$IM_LOG"
: > "$WF_LOG"

echo "[tinker.sh] starting inference_model debug in background, log=$IM_LOG"
"$TRINITY" debug --config "$CONFIG" --module inference_model >"$IM_LOG" 2>&1 &
IM_PID=$!

cleanup() {
  echo "[tinker.sh] cleaning up inference_model pid=$IM_PID"
  # Escalate: SIGINT first (graceful), then SIGKILL if it doesn't exit.
  # `trinity debug --module inference_model` does not always honor SIGINT,
  # so we never block forever on `wait`.
  kill -INT "$IM_PID" 2>/dev/null || true
  for _ in 1 2 3 4 5; do
    kill -0 "$IM_PID" 2>/dev/null || break
    sleep 1
  done
  if kill -0 "$IM_PID" 2>/dev/null; then
    echo "[tinker.sh] inference_model did not exit on SIGINT; sending SIGKILL"
    kill -KILL "$IM_PID" 2>/dev/null || true
  fi
  wait "$IM_PID" 2>/dev/null || true
  # Catch any orphan trinity-debug children (e.g. ray workers spawned by it)
  pkill -KILL -f 'trinity debug --config .*copaw_eval_workflow\.tinker\.yaml' 2>/dev/null || true
  echo "[tinker.sh] stopping dedicated Ray head (temp=$TRINITY_RAY_TMPDIR)"
  # Only kill processes belonging to *our* dedicated Ray cluster (matched
  # by the unique --temp-dir path), so any TuFT-managed cluster on the host
  # is left untouched.
  pkill -INT -f "$TRINITY_RAY_TMPDIR" 2>/dev/null || true
  sleep 2
  pkill -KILL -f "$TRINITY_RAY_TMPDIR" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Wait for the actor to be ready (at most 5 minutes).
WAIT_DEADLINE=$(( $(date +%s) + 300 ))
while true; do
  if grep -q 'Inference models started successfully for debugging' "$IM_LOG" 2>/dev/null; then
    echo "[tinker.sh] inference_model is ready"
    break
  fi
  if ! kill -0 "$IM_PID" 2>/dev/null; then
    echo "[tinker.sh] inference_model exited unexpectedly; tail of log:"
    tail -60 "$IM_LOG"
    exit 1
  fi
  if [ "$(date +%s)" -gt "$WAIT_DEADLINE" ]; then
    echo "[tinker.sh] timed out waiting for inference_model; tail of log:"
    tail -60 "$IM_LOG"
    exit 1
  fi
  sleep 3
done

echo "[tinker.sh] running workflow debug, log=$WF_LOG"
"$TRINITY" debug --config "$CONFIG" --module workflow \
  --plugin-dir "$PLUGIN_DIR" --output-dir "$OUTPUT_DIR" \
  --disable-overwrite 2>&1 | tee "$WF_LOG"
WF_RC=${PIPESTATUS[0]}
echo "[tinker.sh] workflow debug exit code=$WF_RC"
exit "$WF_RC"
