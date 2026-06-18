# 第 2 章：拆开黑盒 — 单次 rollout 到底发生了什么

> **本章模式：拆解**。你已经在第 1 章看到 score 涨了 10 pp，但说不清这 10 pp 来自哪里。这一章打开黑盒第一层：**单次 rollout 内部到底发生了什么**。

---

## 2.0 概念地图

先记住一个三层结构（后面几章都会反复回到它）：

```
┌─────────────────────────────────────────────────────────┐
│  Step（一次训练步） = 8 个 prompt × G=8 个 trajectory   │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Trajectory（一条 rollout）                       │   │
│  │  ┌────────────────────────────────────────────┐  │   │
│  │  │  ReAct 多步循环（这一章的主角）            │  │   │
│  │  │  message₀ → tool_call → tool_result → ...   │  │   │
│  │  └────────────────────────────────────────────┘  │   │
│  │  最后产出：score ∈ {0, 0.33, 0.67, 1.0}          │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

- 1 个 step（约 28 min）：trinity 取 8 个不同 prompt，每个采样 G=8 条 trajectory → **64 条 trajectory**。
- 1 条 trajectory（约 5 min）：模型在 sandbox 里跑一次完整的 ReAct 循环，最后被 verifier 打分。
- 1 步 ReAct（约 30 sec）：模型一次"思考 → 调工具 → 看结果"。

**这一章只看最里面那层**：单条 trajectory 内部，模型到底干了什么。

---

## 2.1 一个具体的 trajectory：simple_085 step 1（失败案例）

打开 `checkpoints/.../step_1_rollout/simple_085/<sandbox_id>/session.json`，里面是一条 trajectory 的完整记录：

```jsonc
{
  "agent": {
    "memory": {
      "content": [
        // [0] system message: 任务 prompt（asyncio TCP echo server 的描述）
        // [1] assistant: thinking + tool_use(execute_shell_command, "mkdir -p /opt/...")
        // [2] tool_result: "ok"
        // [3] assistant: thinking + tool_use(write_file, "/opt/server.py", "...asyncio.run(main())...")
        // [4] tool_result: "ok"
        // [5] assistant: tool_use(execute_shell_command, "python /opt/server.py &")
        // [6] tool_result: "RuntimeError: This event loop is already running"   ← 失败点
        // [7] assistant: tool_use(execute_shell_command, "curl 127.0.0.1:9085")
        // [8] tool_result: "Connection refused"
        // [9] assistant: 输出 【DONE】
      ]
    }
  }
}
```

→ 任务结束，verifier 跑 checklist，**3 个 check 过 1 个**，score = 0.33。

**关键观察**：模型确实做了一连串工具调用，但**没有从 tool_result 里学到反馈**。`RuntimeError` 已经报出来了，模型在第 [7] 步还是不管不顾地 curl。这是 base policy 的典型缺陷。

---

## 2.2 同一任务 step 19（成功案例）

```jsonc
{
  "agent": {
    "memory": {
      "content": [
        // [0] system message: 同上
        // [1] assistant: thinking + tool_use(write_file, "/opt/server.py",
        //      "loop = asyncio.new_event_loop(); loop.run_until_complete(serve())...")
        // [2] tool_result: "ok"
        // [3] assistant: tool_use(execute_shell_command, "python /opt/server.py &")
        // [4] tool_result: "Server listening on 127.0.0.1:9085"
        // [5] assistant: thinking + tool_use(execute_shell_command, "sleep 1; lsof -i:9085")
        //      ← 主动等待服务就绪
        // [6] tool_result: "python ... LISTEN"
        // [7] assistant: tool_use(execute_shell_command, "curl 127.0.0.1:9085")
        // [8] tool_result: "echoed: hello"
        // [9] assistant: write anchor file → 输出 【DONE】
      ]
    }
  }
}
```

→ 任务结束，**3 个 check 全过**，score = 1.0。

**对照表**（同一个任务，base policy vs step 19 的 policy）：

| 步骤 | base policy 的做法 | step 19 policy 的做法 | 关键差异 |
|---|---|---|---|
| 启动服务 | `asyncio.run(main())` | `loop = asyncio.new_event_loop(); loop.run_until_complete(...)` | 学会**避开 sandbox event loop 冲突** |
| curl 之前 | 立刻 curl | `sleep 1; lsof -i:PORT` 确认就绪 | 学会**等待服务 ready** |
| 看到错误后 | 忽略，继续走流程 | thinking 中明确分析错误 → 改写代码 | 学会**根据 tool_result 调整** |
| anchor 文件 | 偶尔忘写 | 稳定写入 | 学会**严格遵守任务约束** |

→ **这 4 行就是"+10 pp" 的微观解释**。RL 把这些散落在 trajectory 各处的小决策一点一点向"更可能成功"的方向推。

---

## 2.3 ReAct 循环的核心数据结构

每一条 message 长这样（OpenAI tool-calling 格式）：

```jsonc
// assistant 消息（模型产出）
{
  "role": "assistant",
  "content": [
    {"type": "text", "text": "<thinking>用 asyncio 实现 echo server，注意 sandbox 已有 event loop...</thinking>"},
    {"type": "tool_use", "id": "call_1", "name": "execute_shell_command",
     "input": {"command": "python -c '...'"}}
  ]
}

// system 消息（环境产出）
{
  "role": "system",
  "content": [
    {"type": "tool_result", "id": "call_1",
     "output": [{"type": "text", "text": "Server listening on 127.0.0.1:9085"}]}
  ]
}
```

记住一个事实：**模型每次 forward 看到的是从 [0] 到当前的所有历史**。也就是说：

- 第 [3] 步的 assistant 决策，依赖于 [0] 任务描述 + [1][2] 第一次 mkdir 结果 + [3] 自己当前的 thinking；
- 第 [9] 步的 assistant 决策，依赖于前面**所有** 8 条消息。

**这就是"多步状态变化"的字面含义**：每一步决策都依赖前面所有 tool_result 累积出的状态，而不是 RLVR 那种"一次输入一次输出"。

---

## 2.4 一条 trajectory 的关键数字

回到 step 1 / step 19 的对比，看看每条 trajectory 的"形状"指标：

| 指标 | step 1 平均 | step 19 平均 | 解读 |
|---|---:|---:|---|
| `agent_llm_calls` | 10.02 | 10.x | LLM 被调用的次数（≈ assistant turn 数） |
| `agent_call_seconds` | 339 s | 350 s | LLM 调用总时间 |
| `latency_seconds` | 372 s | 350 s | 一条 trajectory 总耗时（含 sandbox） |
| `over_length` 比例 | 7.8% | 4.7% | 因为输出过长被截断的比例 |
| `trajectory_steps_truncated` | 0.13 | 0.20 | 单 step 工具结果被截断的比例 |

**关键发现**：
- LLM 调用次数没显著变化（~10 次）→ trajectory 的"步数"没变
- over_length 反而下降 → 模型用更紧凑的方式做对了事
- 总耗时略减 → 不是靠"花更多时间想"赢的，是策略本身改进了

---

## 2.5 score 是按什么粒度给出的

注意 §2.1 / §2.2 看到的 score 不是 0 就是 1，而是 `0.33 / 0.67 / 1.0`——这就是**多 check 评分**的体现：

```python
# 伪代码：simple_085 的判分逻辑
checks = [
    server_listening_on_correct_port(),    # check 1
    curl_returns_correct_echo(),           # check 2
    anchor_file_correct(),                 # check 3
]
score = sum(checks) / len(checks)  # → {0, 0.33, 0.67, 1.0}
```

这意味着 reward 是**密集的中间信号**，不是稀疏的 0/1。

**为什么这很重要**：

- 如果只有 0/1 reward，RL 训练时 advantage 经常全为 0（要么全失败要么全成功），梯度信号弱；
- 多 check reward 让 advantage 永远有结构——做对 1 个 check 比 0 个好，做对 2 个比 1 个好。

第 3 章会展开讲 reward 的设计。

---

## 2.6 这一章你应该带走的

✅ **trajectory 的本质**：一段 ReAct 历史 [system, assistant, tool_result, assistant, ...]，每一步决策都看到前面所有上下文。
✅ **+10 pp 的来源**：不是模型变"聪明"了，而是在很多小决策点（启动方式、等待就绪、读懂错误）上变得更稳。
✅ **score 是密集的**：每个 check 给 1/N 分，比 0/1 reward 更利于 RL 学习。

❌ **你还不需要懂**：
- score 怎么从一组 check 聚合成 reward 给训练用（→ 第 3 章）
- 同一个 prompt 8 条 trajectory 之间怎么比较出 advantage（→ 第 4 章）
- 模型权重在看到这些 trajectory 后具体怎么变（→ 第 5 章）

**留个问题给自己**：你现在知道一条 trajectory 内部长什么样、它产出 0.33 / 0.67 / 1.0 这样的 score。但 RL 训练真正用的"reward"是不是就等于这个 score？还是还要做某种处理（normalize / clip / penalty）？第 3 章揭晓。

---

**上一章**：[第 1 章：5 分钟跑通](./ch1_5分钟跑通.md) ｜ **下一章**：[第 3 章：reward 怎么算](./ch3_reward怎么算.md)
