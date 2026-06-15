"""queries_simple_workflow.py — e2b qwenpaw sandbox 中跑 queries_simple
任务，并在容器销毁前注入 test_case_simple_XXX.py 做确定性
ground truth 校验。

bench 模式：reward 来自 sandbox 内 run_simple.py 输出的 result_simple.json，
workflow 返回 单个 Experience（空 tokens）仅供难度记账。

train (multi_step_grpo) 模式：从 task_dir/session.json 读
_model_trajectory（qwenpaw 在 sandbox 里记录的逐 step 调用），每 step
输出一个 Experience：
  - tokens         = prompt_token_ids + token_ids   (全序列)
  - prompt_length  = len(prompt_token_ids)
  - logprobs       = step.logprobs                  (仅 response 部分)
  - action_mask    = ones(len(token_ids))
  - reward         = final task reward (broadcast 由 multi_step_grpo 负责)
  - eid.step       = i
所有 step 共享同一份 reward；multi_step_grpo 会在同一 task 的
repeat_times 个 run 之间计算 advantage，再广播到该 run 的所有 step。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import torch

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows import WORKFLOWS
from trinity.common.workflows.workflow import MultiTurnWorkflow, Task

# 单次 LLM forward 在 FSDP-2 上可安全处理的总 token 上限。
# 超过则截断到 32K，并判 trajectory 为 over_length 失败（reward=0）。
MAX_STEP_TOKENS = int(os.environ.get("MAX_STEP_TOKENS", 32768))
# Trajectory 总耗时上限（agent_call_seconds），超过也判 over_time 失败。
MAX_TRAJ_SECONDS = float(os.environ.get("MAX_TRAJ_SECONDS", 300.0))

# ============================================================
# V26: DAPO-style soft length penalty 开关与参数
# 默认 OVERLONG_PENALTY_ENABLE=0 走原硬清零（reward=0）路径，
# 与历史实验 v22 / v23 / v17e / v25 行为完全一致。
# ============================================================
OVERLONG_PENALTY_ENABLE = int(os.environ.get("OVERLONG_PENALTY_ENABLE", "0"))
OVERLONG_MAX_TOKENS = int(os.environ.get("OVERLONG_MAX_TOKENS", "31000"))
OVERLONG_BUFFER_TOKENS = int(os.environ.get("OVERLONG_BUFFER_TOKENS", "3000"))
OVERLONG_PENALTY_FACTOR = float(os.environ.get("OVERLONG_PENALTY_FACTOR", "1.0"))

# 占位 Experience 用的 token id。
# 设计原则：action_mask=[0] 已保证不产生梯度，token 内容实际不参与训练。
# 选 0 是为了跨模型通用：
#   - 任何 tokenizer 的 vocab 都含 token id 0（不会越界）
#   - detokenize/decode 时不会报错
#   - 避免写死某个模型的 EOS/BOS（换模型时出错）
# 换模型时一般不需要改。如果将来想让 mask=1 让模型真学东西，
# 再考虑换成真实的 EOS token id。
PLACEHOLDER_PROMPT_TOKEN_ID = int(os.environ.get("PLACEHOLDER_PROMPT_TOKEN_ID", "0"))
PLACEHOLDER_RESPONSE_TOKEN_ID = int(os.environ.get("PLACEHOLDER_RESPONSE_TOKEN_ID", "0"))


def _make_placeholder_experience(
    reward: Optional[float],
    metrics: dict,
    info: dict,
) -> Experience:
    """构造最小占位 Experience（1 prompt token + 1 response token）。

    为什么不能用 tokens=[]：
        Trinity tinker trainer 的 to_tinker_input 在 prompt_length=0 时
        会调 torch.zeros(-1) 报 RuntimeError；response_length=0 时
        token_level_reward[-1] 报 IndexError。所以必须占 ≥1 个 token。

    为什么 token id 选 0 而不是 EOS：
        跨模型通用：任何 tokenizer vocab 都有 id=0，不越界不报错。
        既然 action_mask=[0] 已保证梯度=0，token 内容本身不重要。

    为什么 action_mask=[0]：
        这是 fallback 占位（session_missing / transient / 超时 hard kill），
        不是真实采样。mask=0 使 pg_loss / entropy_loss / kl_loss 都
        被 mask 抹零，实际不产生梯度，避免用假 logprobs 误导
        importance ratio。reward 仍会进入 GRPO group 的 mean/std
        统计，间接调节同组其他真实 trajectory 的 advantage。
    """
    return Experience(
        tokens=torch.tensor(
            [PLACEHOLDER_PROMPT_TOKEN_ID, PLACEHOLDER_RESPONSE_TOKEN_ID], dtype=torch.long
        ),
        prompt_length=1,
        logprobs=torch.tensor([0.0], dtype=torch.float),
        action_mask=torch.tensor([0], dtype=torch.bool),
        reward=float(reward) if reward is not None else 0.0,
        metrics=dict(metrics),
        info=dict(info),
    )


def _load_trajectory(task_dir: Optional[str]) -> list[dict]:
    """从 task_dir/session.json 读 _model_trajectory。任何异常返回空 list。"""
    if not task_dir:
        return []
    sp = Path(task_dir) / "session.json"
    if not sp.is_file():
        return []
    try:
        d = json.loads(sp.read_text(encoding="utf-8"))
        mt = d.get("agent", {}).get("_model_trajectory") or []
        return [s for s in mt if isinstance(s, dict)]
    except Exception:
        return []


def _build_step_experiences(
    trajectory: list[dict],
    reward: Optional[float],
    base_metrics: dict,
    base_info: dict,
    max_total_tokens: int = MAX_STEP_TOKENS,
) -> Tuple[List[Experience], int, int]:
    """把 trajectory 转成 multi-step Experience 列表。

    返回 (exps, num_truncated_steps, max_step_tokens_seen)。

    超长策略（不丢弃）：单条 step 总 token 超过 ``max_total_tokens``
    会被截断保留前 N 个 token：response 尾部先被切掉；若 prompt
    本身已超限则连 prompt 也从头保留前 N（这种极端 step 上下文被
    截断后训练价值低，但仍能作为 length penalty 信号供
    GRPO advantage 学习）。
    """
    exps: list[Experience] = []
    num_truncated = 0
    max_seen = 0
    for i, step in enumerate(trajectory):
        prompt_ids = list(step.get("prompt_token_ids") or [])
        resp_ids = list(step.get("token_ids") or [])
        logprobs = list(step.get("logprobs") or [])
        if not prompt_ids or not resp_ids or not logprobs:
            continue
        if len(resp_ids) != len(logprobs):
            continue
        total_len = len(prompt_ids) + len(resp_ids)
        max_seen = max(max_seen, total_len)
        truncated = False
        if total_len > max_total_tokens:
            truncated = True
            num_truncated += 1
            if len(prompt_ids) >= max_total_tokens:
                # prompt 已超阈：prompt 从头保留前 N，response 及 logprobs 丢弃。
                # 这种极端情形下该 step 无有效训练信号（action_mask 会为空），
                # 跳过生成 Experience，但 num_truncated 已计数。
                continue
            keep_resp = max_total_tokens - len(prompt_ids)
            resp_ids = resp_ids[:keep_resp]
            logprobs = logprobs[:keep_resp]
            if not resp_ids:
                continue

        full_tokens = torch.tensor(prompt_ids + resp_ids, dtype=torch.long)
        prompt_length = len(prompt_ids)
        action_mask = torch.ones(len(resp_ids), dtype=torch.bool)
        logprobs_t = torch.tensor(logprobs, dtype=torch.float)

        info = dict(base_info)
        info["trajectory_step"] = i
        info["trajectory_len"] = len(trajectory)
        info["step_total_tokens"] = total_len
        info["step_truncated"] = truncated
        exp = Experience(
            tokens=full_tokens,
            prompt_length=prompt_length,
            logprobs=logprobs_t,
            action_mask=action_mask,
            reward=reward,
            metrics=dict(base_metrics),
            info=info,
        )
        exp.eid.step = i
        exps.append(exp)
    return exps, num_truncated, max_seen


def _build_trajectory_experience(
    trajectory: list[dict],
    reward: Optional[float],
    base_metrics: dict,
    base_info: dict,
    max_total_tokens: int = MAX_STEP_TOKENS,
) -> Tuple[Optional[Experience], int, int]:
    """整条 trajectory 聚合成 1 个 Experience（multi-turn 多段 mask）。

    返回 (exp_or_None, truncated_count, total_tokens_seen)。

    tokens / mask 设计（利用 step.prompt_token_ids 累积含历史这个事实）：
      - tokens = last_step.prompt_token_ids + last_step.token_ids
        这就是完整 trajectory（first_prompt + R_1 + tool_1 + R_2 + ... + R_N）
      - prompt_length = len(first_step.prompt_token_ids)。“prompt” 只指系统
        提示以及首轮 user query，后续的 tool result 不计作 prompt（它们被
        action_mask=0 处理，本质也不造梯度）。
      - response_length = len(tokens) - prompt_length
      - action_mask：response 段中，每个 step.token_ids 区域 = 1（model 生成），
        tool result 区域 = 0。
      - logprobs：对齐 response 段，action_mask=1 处填 step.logprobs，
        =0 处填 0.0（不参与梯度）。
      - reward：trajectory-level final reward（已由 over_length/over_time 调过）。

    截断策略：trajectory 总长 > max_total_tokens 时保头部，尾部超出部分
    在 mask 上被截。这种情况下 truncated_count = 1。
    """
    # 过滤无效 step（缺 token / logprobs 不对齐）
    steps = [
        s
        for s in trajectory
        if s.get("prompt_token_ids")
        and s.get("token_ids")
        and s.get("logprobs")
        and len(s["logprobs"]) == len(s["token_ids"])
    ]
    if not steps:
        return None, 0, 0

    # 逐 step 拼接完整 trajectory tokens。
    # 为什么不用 last_step.prompt + last_step.token_ids：
    #   ReAct 多 turn 中，qwenpaw 在构造下一轮 prompt 时会对整个 chat history
    #   re-tokenize，BPE boundary 变化后 token id 甚至长度都能变。这样 last.prompt
    #   中嵌入的历史 R_k 在 token 维度上与 step_k.token_ids 不能严格对齐。
    #
    # 拼接逻辑：
    #   full = first.prompt + R_1 + tool_1 + R_2 + tool_2 + ... + R_N
    #   R_k = step_k.token_ids                              （model 生成，action_mask=1）
    #   tool_k = step_{k+1}.prompt[len(step_k.prompt)+len(step_k.token_ids):]
    #          = step_{k+1} 估价下“R_k 之后被插入”的 tool result + user turn。
    # 这样 R_k 区域在 full 中严格等于 step_k.token_ids，与 step.logprobs 严格对齐。
    first = steps[0]
    full_tokens: list[int] = list(first["prompt_token_ids"])
    prompt_length = len(full_tokens)
    r_k_positions: list[tuple[int, int]] = []  # (start, end) of each R_k in full
    for k, step in enumerate(steps):
        sr = list(step["token_ids"])
        rk_start = len(full_tokens)
        full_tokens.extend(sr)
        rk_end = len(full_tokens)
        r_k_positions.append((rk_start, rk_end))
        # 追加 tool_k（最后一 step 后面没有 tool）
        if k + 1 < len(steps):
            next_prompt = steps[k + 1]["prompt_token_ids"]
            tool_start = len(step["prompt_token_ids"]) + len(sr)
            if tool_start < len(next_prompt):
                full_tokens.extend(list(next_prompt[tool_start:]))

    total_tokens = len(full_tokens)
    max_seen = total_tokens

    truncated = 0
    if total_tokens > max_total_tokens:
        truncated = 1
        full_tokens = full_tokens[:max_total_tokens]
        total_tokens = max_total_tokens

    response_length = total_tokens - prompt_length
    if response_length <= 0:
        # first.prompt 本身已超 max_total_tokens，无法训练
        return None, truncated, max_seen

    action_mask = [0] * response_length
    logprobs_arr = [0.0] * response_length
    for (rk_start, rk_end), step in zip(r_k_positions, steps):
        if rk_start >= total_tokens:
            break  # 后续 step 都被截丢了
        rk_end_clipped = min(rk_end, total_tokens)
        rs = rk_start - prompt_length
        re_ = rk_end_clipped - prompt_length
        if rs < 0 or rs >= re_:
            continue
        sl = step["logprobs"]
        for k_idx in range(re_ - rs):
            action_mask[rs + k_idx] = 1
            if k_idx < len(sl):
                logprobs_arr[rs + k_idx] = float(sl[k_idx])

    info = dict(base_info)
    info["trajectory_len"] = len(steps)
    info["trajectory_total_tokens"] = max_seen
    info["trajectory_truncated"] = bool(truncated)

    exp = Experience(
        tokens=torch.tensor(full_tokens, dtype=torch.long),
        prompt_length=prompt_length,
        logprobs=torch.tensor(logprobs_arr, dtype=torch.float),
        action_mask=torch.tensor(action_mask, dtype=torch.bool),
        reward=reward,
        metrics=dict(base_metrics),
        info=info,
    )
    return exp, truncated, max_seen


@WORKFLOWS.register_module("queries_simple_workflow_v28eval")
class QueriesSimpleWorkflow(MultiTurnWorkflow):
    """单条 queries_simple 任务的 e2b rollout + ground-truth 校验。"""

    def __init__(
        self,
        *,
        task: Task,
        model: ModelWrapper,
        auxiliary_models: Optional[List[ModelWrapper]] = None,
    ):
        super().__init__(
            task=task,
            model=model,
            auxiliary_models=auxiliary_models,
        )

    def run(self) -> List[Experience]:
        # 延迟 import，避免 e2b SDK / numpy 等被无关代码路径加载
        from examples.copaw_rl.workflows.sandbox_utils import (
            get_or_create_sandbox,
            run_simple_workflow,
        )

        wa = self.task.workflow_args
        token = wa["token"]
        domain = wa["domain"]
        template = wa["template"]
        sandbox_id = wa.get("sandbox_id", None)
        dashscope_api_key = wa.get("dashscope_api_key", "") or None
        checkpoint_job_dir = wa["checkpoint_job_dir"]

        # test_cases 在 host 上的位置（同步进 sandbox 的 /root/test_cases/）
        test_cases_host_dir = Path(
            wa.get(
                "test_cases_host_dir",
                str(
                    Path(__file__).parent.parent
                    / "queries_simple"
                    / "test_cases"
                ),
            )
        )

        raw = self.task.raw_task
        task_id = str(raw["task_id"])
        # tasks.json 中 question 字段 = 已注入 fields 后的最终 query
        query = str(raw.get("question") or raw.get("query") or "")
        fields = dict(raw.get("fields") or {})
        if not fields:
            self.logger.warning(
                f"[Task {task_id}] tasks.json 缺少 fields，test_case 将无字段可用，必然 fail"
            )

        api_server_url = f"{self.model.api_address}/v1"
        provider_model_id = self.model.model_path or self.model.model_name
        # === V28 EVAL OVERRIDE ===
        # task.raw_task 中带 _eval_sampler_path 时，把它作为 model_path 透传给
        # qwenpaw（OpenAI request 的 model 字段）。TuFT OAI router 的 model_resolver
        # 看到 "tinker://..." 就会 dispatch 到对应 sampler-N 的 LoRA adapter。
        # 这样多个 sandbox 可以并行用不同 v28 历史 ckpt 做 inference，无需切换 active sampler。
        eval_sampler_path = raw.get("_eval_sampler_path")
        eval_ckpt_step = raw.get("_eval_ckpt_step")
        if eval_sampler_path:
            self.logger.info(
                f"[Task {task_id}] [v28eval] override sampler -> step {eval_ckpt_step}: {eval_sampler_path}"
            )
            provider_model_id = eval_sampler_path

        # per-task seed 透传：从 raw_task["_eval_seed"] 读取，设入环境变量
        # 让后续 run_simple_workflow 中的 _inject_llm_seed() 和固定 session_id 生效
        eval_seed = raw.get("_eval_seed")
        if eval_seed is not None:
            os.environ["LLM_SEED"] = str(eval_seed)
            self.logger.info(f"[Task {task_id}] [v28eval] per-task seed={eval_seed}")
        model_version = self.model.model_version
        # 落盘路径自带 train/eval 标签，让后续分析脚本不需要
        # 从 runner log 里反查 sandbox 归属。Trinity 为 eval taskset 中的 task
        # 自动设置 task.is_eval=True（见 trinity/common/workflows/workflow.py）。
        mode_tag = "eval" if getattr(self.task, "is_eval", False) else "rollout"
        model_label = f"step_{model_version}_{mode_tag}"

        sandbox, created = get_or_create_sandbox(
            sandbox_id, token, domain, template, self.logger
        )

        try:
            result = run_simple_workflow(
                sandbox=sandbox,
                task_id=task_id,
                query=query,
                fields=fields,
                api_server_url=api_server_url,
                model_path=provider_model_id,
                dashscope_api_key=dashscope_api_key,
                test_cases_host_dir=test_cases_host_dir,
                model_label=model_label,
                checkpoint_job_dir=checkpoint_job_dir,
                logger=self.logger,
            )
        except Exception as e:
            # 常见异常来源：e2b sandbox deadline_exceeded / process not found /
            # 脚手架超时。这不是 model 的“真实失败”，不应作为 reward=0
            # 训练信号；同 transient_failure 一起记账并 reward=None 让 GRPO 跳过。
            self.logger.error(
                f"[Task {task_id}] run_simple_workflow 异常: {e}", exc_info=True
            )
            result = {
                "task_id": task_id,
                "status": "sandbox_error",
                "reason": f"{type(e).__name__}: {e}",
                "passed": False,
                "score": 0.0,
                "agent_call_seconds": 0.0,
                "agent_llm_calls": 0,
                "task_dir": None,
            }
        finally:
            try:
                sandbox.kill()
            except Exception as e:
                self.logger.warning(f"sandbox.kill() 失败: {e}")

        # 把 result 中可量化的字段当 metrics（供 monitor / wandb）
        metrics = {
            k: float(v)
            for k, v in result.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        # 通过率字段：mode=bench 下用作"难度评测"的核心信号
        metrics["pass"] = 1.0 if result.get("passed") else 0.0

        # transient 故障（上游 LLM provider 重启 / 5xx 导致 0 LLM 调用，
        # 或者 sandbox e2b 依赖体报错）不应该贡献 reward。
        # 这类样本返回一个空-token Experience且 reward=0.0，避免 Trinity
        # workflow_runner 报 "An empty experience is generated" assert；
        # 同时由于 tokens=[]，action_mask=[]，实际不会产生梯度，
        # 仅占一个 GRPO group slot。
        is_transient = result.get("status") in {"no_llm_call", "sandbox_error", "call_agent_error", "qwenpaw_not_ready"}
        metrics["transient_failure"] = 1.0 if is_transient else 0.0

        if is_transient:
            self.logger.warning(
                f"[Task {task_id}] transient_failure status={result.get('status')} "
                f"reason={result.get('reason')} → 返回 dummy 占位 Experience reward=0"
            )
            return [_make_placeholder_experience(
                reward=0.0,
                metrics=metrics,
                info={
                    "task_id": task_id,
                    "model_label": model_label,
                    "status": result.get("status"),
                    "reason": result.get("reason"),
                    "transient": True,
                    "placeholder": True,
                },
            )]

        reward: Optional[float] = float(result.get("score") or 0.0) / 100.0  # score 是百分制 -> 归一化

        base_info = {
            "task_id": task_id,
            "model_label": model_label,
            "task_dir": result.get("task_dir"),
            "status": result.get("status"),
            "reason": result.get("reason"),
        }

        # 试图从 sandbox 的 session.json 提取逐步 trajectory，生成
        # multi-step Experience 列表。这是 train (multi_step_grpo) 的
        # 标准输入。如果 trajectory 缺失（老的 bench result 或 transient 故障）
        # 退化为单个空-token Experience，仅供 bench 记账。
        trajectory: list = []
        if not is_transient:
            trajectory = _load_trajectory(result.get("task_dir"))

        # ============================================================
        # over_length / over_time 判定 (训练侧 length / time penalty)
        # ============================================================
        # 思路：超长 / 超时的 trajectory 代价太高，即使 passed=True 也不应鼓励
        # 。设 reward = 0。GRPO 在同 task 的 repeat_times 个 run 内计 advantage
        # 时，这些样本相对“快且短”的成功样本会获得负 advantage，实现
        # length/time penalty。
        agent_call_seconds = float(result.get("agent_call_seconds") or 0.0)
        agent_timed_out = bool(result.get("agent_timed_out"))
        is_over_time = (not is_transient) and (
            agent_call_seconds >= MAX_TRAJ_SECONDS or agent_timed_out
        )
        traj_step_max_tokens = 0
        for s in trajectory:
            if isinstance(s, dict):
                p = s.get("prompt_token_ids") or []
                r = s.get("token_ids") or []
                lp = s.get("logprobs") or []
                if p and r and len(r) == len(lp):
                    traj_step_max_tokens = max(traj_step_max_tokens, len(p) + len(r))
        is_over_length_hard = (not is_transient) and traj_step_max_tokens > MAX_STEP_TOKENS

        # ---- time 维度：与 v22 完全一致的硬清零 ----
        if is_over_time and reward is not None:
            base_info["original_status"] = result.get("status")
            base_info["original_score"] = float(result.get("score") or 0.0)
            base_info["status"] = "over_time"
            base_info["agent_call_seconds"] = agent_call_seconds
            base_info["max_step_tokens"] = traj_step_max_tokens
            reward = 0.0
            metrics["pass"] = 0.0
        metrics["over_time"] = 1.0 if is_over_time else 0.0

        # ---- length 维度：根据 OVERLONG_PENALTY_ENABLE 选硬清零 / DAPO 软扣分 ----
        overlong_soft = 0.0
        is_over_length_soft_zone = False
        if (
            OVERLONG_PENALTY_ENABLE
            and (not is_transient)
            and (not is_over_time)
            and reward is not None
        ):
            expected = OVERLONG_MAX_TOKENS - OVERLONG_BUFFER_TOKENS
            exceed = traj_step_max_tokens - expected
            if exceed > 0:
                is_over_length_soft_zone = True
                overlong_soft = (
                    -exceed / float(OVERLONG_BUFFER_TOKENS) * OVERLONG_PENALTY_FACTOR
                )
                overlong_soft = max(overlong_soft, -OVERLONG_PENALTY_FACTOR * 1.5)
                reward = float(reward) + overlong_soft
                base_info["original_status"] = result.get("status")
                base_info["original_score"] = float(result.get("score") or 0.0)
                base_info["overlong_soft_penalty"] = overlong_soft
                base_info["max_step_tokens"] = traj_step_max_tokens
                base_info["status"] = "over_length_soft"
        elif is_over_length_hard and reward is not None and not is_over_time:
            base_info["original_status"] = result.get("status")
            base_info["original_score"] = float(result.get("score") or 0.0)
            base_info["status"] = "over_length"
            base_info["max_step_tokens"] = traj_step_max_tokens
            reward = 0.0
            metrics["pass"] = 0.0

        if OVERLONG_PENALTY_ENABLE:
            metrics["over_length"] = 1.0 if is_over_length_soft_zone else 0.0
        else:
            metrics["over_length"] = 1.0 if is_over_length_hard else 0.0
        metrics["over_length_soft_penalty"] = float(overlong_soft)
        metrics["max_step_tokens"] = float(traj_step_max_tokens)

        step_exps, num_truncated, max_seen = _build_step_experiences(
            trajectory, reward, metrics, base_info
        )
        metrics["trajectory_steps_truncated"] = float(num_truncated)
        metrics["trajectory_step_max_tokens"] = float(max_seen)

        # 切换开关：USE_TRAJECTORY_EXPERIENCE=1 走新路径（1 trajectory = 1 Experience）。
        # 默认 0 走旧路径以保证向后兼容。新路径需配 yaml algorithm_type=grpo。
        use_traj = os.environ.get("USE_TRAJECTORY_EXPERIENCE", "0") == "1"
        if use_traj:
            traj_exp, traj_truncated, traj_total_tokens = _build_trajectory_experience(
                trajectory, reward, metrics, base_info
            )
            metrics["trajectory_truncated"] = float(traj_truncated)
            metrics["trajectory_total_tokens"] = float(traj_total_tokens)
            if traj_exp is not None:
                self.logger.info(
                    f"[Task {task_id}] [traj-mode] passed={result.get('passed')} "
                    f"score={result.get('score')} reward={reward} "
                    f"status={base_info.get('status')} "
                    f"agent_seconds={agent_call_seconds:.1f} "
                    f"traj_total_tokens={traj_total_tokens} truncated={bool(traj_truncated)} "
                    f"resp_len={int(traj_exp.action_mask.sum().item())} "
                    f"sandbox={'created' if created else 'connected'}"
                )
                # === V28 EVAL 归因行（每条唯一不会被 Ray dedup）===
                if eval_ckpt_step is not None:
                    self.logger.info(
                        f"[v28eval-result] task={task_id} ckpt={eval_ckpt_step} "
                        f"score={result.get('score')} passed={result.get('passed')} "
                        f"reward={reward} status={base_info.get('status')} "
                        f"seed={eval_seed or ''} "
                        f"model_label={model_label}"
                    )
                return [traj_exp]
            # traj_exp 为 None（first prompt 已超限）走下面占位逻辑
        elif step_exps:
            self.logger.info(
                f"[Task {task_id}] passed={result.get('passed')} "
                f"score={result.get('score')} reward={reward} "
                f"status={base_info.get('status')} "
                f"agent_seconds={agent_call_seconds:.1f} "
                f"step_max_tokens={max_seen} truncated_steps={num_truncated} "
                f"trajectory_steps={len(step_exps)} sandbox={'created' if created else 'connected'}"
            )
            return step_exps

        # 没有 trajectory 数据（session_missing / agent 超时被 kill / 旧 bench result）。
        # 返回 EOS 占位 Experience，reward 保留（可能已被 over_length/over_time 置 0）。
        # Trinity workflow_runner 要求 len(exps)>0，不能返空 list。
        # tokens=[0, 0] + action_mask=[0] 使该 Experience 不产生梯度，只占一个 GRPO group slot。
        info = dict(base_info)
        info["placeholder"] = True
        self.logger.warning(
            f"[Task {task_id}] 无有效 trajectory（status={base_info.get('status')} "
            f"agent_seconds={agent_call_seconds:.1f} timed_out={agent_timed_out}）"
            f" → 返回 dummy 占位 Experience reward={reward}"
        )
        return [_make_placeholder_experience(reward=reward, metrics=metrics, info=info)]

        # 以下原 fallback 路径已被上面取代（dead code）
        exp = Experience(  # noqa: F841
            tokens=torch.tensor([]),
            logprobs=torch.tensor([]),
            prompt_length=0,
            action_mask=torch.tensor([]),
            reward=reward,
            metrics=metrics,
            info=base_info,
        )
        self.logger.info(
            f"[Task {task_id}] passed={result.get('passed')} "
            f"score={result.get('score')} reward={reward} "
            f"trajectory_steps=0 (bench-only) sandbox={'created' if created else 'connected'}"
        )
        return [exp]
