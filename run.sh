#!/bin/bash
# run.sh — one-shot entry for the Agentic-RL tutorial (docs/RL_tutorial).
#
# What it does:
#   1) install dependencies into a local .venv via `uv sync`
#      (Trinity-RFT is pulled in as a git dependency; see pyproject.toml)
#   2) launch the CH1 reproduction experiment (seed-fixed, 19 training steps)
#
# Usage:
#   # provide secrets via environment or a local ./.env (see .env.example)
#   bash run.sh
#
#   # quick smoke test (2 steps):
#   TOTAL_STEPS=2 bash run.sh
#
# Backend:
#   - default: self-hosted TuFT on this host  ->  http://<vpc-ip>:10610
#   - Tinker cloud: export TINKER_BASE_URL=https://api.tinker.thinkingmachines.ai
#                   export TINKER_API_KEY=<your tinker key>
set -e

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

# --- 1) optional local secrets file (gitignored) ---
if [ -f "$HERE/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$HERE/.env"
  set +a
fi

# --- 2) required secrets (fail-fast; never hardcoded in the repo) ---
: "${E2B_API_KEY:?set E2B_API_KEY (see .env.example)}"
# DASHSCOPE_API_KEY / OSS_* are unused in the queries_simple tutorial path
# (run_simple_workflow does not depend on OSS; agent uses the RL model, not DashScope).
# Export empty defaults so run_train.sh's optional references don't fail.
export DASHSCOPE_API_KEY="${DASHSCOPE_API_KEY:-}"
export OSS_ACCESS_KEY_ID="${OSS_ACCESS_KEY_ID:-}"
export OSS_ACCESS_KEY_SECRET="${OSS_ACCESS_KEY_SECRET:-}"

# --- 3) install deps into local .venv ---
echo "[run] uv sync ..."
uv sync

# --- 4) tutorial fixed params (all overridable via env) ---
export EXP_MODEL_NAME="${EXP_MODEL_NAME:-Qwen/Qwen3-4B-Thinking-2507}"
export EXP_DATE="${EXP_DATE:-$(date +%m%d)}"
export EXP_SOURCE="${EXP_SOURCE:-tutorial}"
export EXP_NAME="${EXP_NAME:-${EXP_MODEL_NAME//\//-}-rl-tutorial-${EXP_DATE}}"
export LR="${LR:-1e-6}"
export TOTAL_STEPS="${TOTAL_STEPS:-19}"
# 8 tasks with batch_size=8 => 1 rollout step per epoch, so training is
# actually capped by total_epochs. run_train.sh defaults TOTAL_EPOCHS=15
# (too low: stops at ~14 steps), so we override to match the yaml default
# (40) and let TOTAL_STEPS be the real limit -> 19 train / 20 rollout steps.
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-40}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-1}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-999}"
export EVAL_ON_STARTUP="${EVAL_ON_STARTUP:-false}"
export RUNNER_PER_MODEL="${RUNNER_PER_MODEL:-64}"

# reproducibility seed (sandbox LLM); vLLM/TuFT + data-shuffle seeds live in the yaml
export ROLLOUT_BASE_SEED="${ROLLOUT_BASE_SEED:-42}"

# DAPO soft length penalty
export OVERLONG_PENALTY_ENABLE="${OVERLONG_PENALTY_ENABLE:-1}"
export OVERLONG_MAX_TOKENS="${OVERLONG_MAX_TOKENS:-31000}"
export OVERLONG_BUFFER_TOKENS="${OVERLONG_BUFFER_TOKENS:-3000}"
export OVERLONG_PENALTY_FACTOR="${OVERLONG_PENALTY_FACTOR:-1.0}"

# --- 5) E2B sandbox (non-secret defaults; the key comes from env) ---
export E2B_DOMAIN="${E2B_DOMAIN:-sandbox01.vpc.cn-hongkong.pai-eas.aliyuncs.com}"
export E2B_TEMPLATE="${E2B_TEMPLATE:-agentscope-qwenpaw-0518}"

# --- 6) training backend base url ---
if [ -z "${TINKER_BASE_URL:-}" ]; then
  _IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
  export TINKER_BASE_URL="http://${_IP:-localhost}:10610"
fi
export TINKER_API_KEY="${TINKER_API_KEY:-tml-tuft-dev-key}"

# --- 7) make the `examples` namespace package importable by the plugin loader ---
#     (Trinity-RFT is installed non-editable, so the repo root is not on sys.path)
export PYTHONPATH="$HERE:${PYTHONPATH:-}"

# --- 8) Ray isolation: dedicated tmpdir + ports.
#     NEVER touch TuFT's default /tmp/ray. Cleanup only matches this tmpdir.
export TRINITY_RAY_TMPDIR="${TRINITY_RAY_TMPDIR:-/tmp/ray-trinity-train-tutorial}"
export TRINITY_RAY_PORT="${TRINITY_RAY_PORT:-6408}"
export TRINITY_RAY_DASHBOARD_PORT="${TRINITY_RAY_DASHBOARD_PORT:-8284}"

# --- 9) config + data paths ---
export CONFIG_OVERRIDE="$HERE/examples/copaw_rl/queries_simple.train.tinker.ch1_repro.yaml"
export TRINITY_QUERIES_SIMPLE_TASKSET_PATH="$HERE/examples/copaw_rl/queries_simple/data"
export TRINITY_QUERIES_SIMPLE_TEST_CASES_DIR="$HERE/examples/copaw_rl/queries_simple/test_cases"
export LOG_FILE="${LOG_FILE:-/tmp/trinity_tutorial_train.log}"

echo "[run] EXP_NAME=$EXP_NAME  TOTAL_STEPS=$TOTAL_STEPS"
echo "[run] TINKER_BASE_URL=$TINKER_BASE_URL"
echo "[run] launching training via run_train.sh ..."
exec bash "$HERE/examples/copaw_rl/queries_simple/run_train.sh"
