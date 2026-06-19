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

**任务简介**：在 E2B sandbox 里用 Python asyncio 写一个 TCP echo server，监听 18085 端口，然后本地连接发送 `ping CPW` 并把回包写入 output.txt。

<details><summary>点击展开：完整任务要求</summary>

- `SERVER_SCRIPT`: `/opt/cpw_simple_085/server.py`（asyncio TCP echo server）
- `SERVER_PORT`: 18085，监听 127.0.0.1
- `OUTPUT_FILE`: `/opt/cpw_simple_085/output.txt`，必须含 `ping CPW`
- `VERIFICATION_ANCHOR`: `CPW-SIMPLE-085`
- **3 个 check**：① anchor 文件存在 ② 端口监听 ③ output 含 ping CPW
- 全程在容器内完成，不使用外部网络

</details>

打开 `checkpoints/.../step_-1_rollout/simple_085/<sandbox_id>/session.json`，里面是一条 trajectory 的完整记录。下面是真实数据（你可以运行 `python scripts/tutorial/ch2_inspect_trajectory.py --sample` 自己看到完整版）：

```
[Step 1] Assistant:
         Tool call: execute_shell_command("mkdir -p /opt/cpw_simple_085")
           └→ Command executed successfully.
         Tool call: write_file("/opt/cpw_simple_085/server.py", <686 chars, 26 lines>)
           └→ Wrote 686 bytes.                        ← 写入了带 bug 的 asyncio 代码
[Step 2] Assistant:
         Tool call: execute_shell_command("python3 server.py &")
           └→ ⚠️ [stderr: SyntaxError: 'await' outside async function]
[Step 3] Assistant:
         Tool call: write_file("/opt/cpw_simple_085/server.py", <761 chars, 29 lines>)
           └→ Wrote 761 bytes.                        ← 重写了一版，仍有属性错误
[Step 4] Assistant:
         Tool call: execute_shell_command("python3 server.py &")
           └→ ⚠️ [stderr: AttributeError: 'coroutine' object has no attribute 'serve_forever']
[Step 5] Assistant (text only, no tool call):
         放弃了，输出一段文字但没有继续执行。
```

→ 任务结束，verifier 跑 checklist，**3 个 check 过 1 个**，score = 0.33。

**关键观察**：模型写了代码、跑了两次，**都从 stderr 拿到了明确错误**——但它没有真正理解错误、也没有验证服务是否启动，最后直接放弃了。这是 base policy 的典型缺陷：**看到错误但不会调整策略**。

---

## 2.2 同一任务 step 19（成功案例）

同样运行 `python scripts/tutorial/ch2_inspect_trajectory.py --sample --compare` 可以看到完整对比。下面是 step 19 的关键片段：

```
[Step 1~3] 初始尝试遇到各种错误（目录不存在、nc 未安装等）
           → 模型没有放弃，而是逐步修复
[Step 4]  Assistant:
           Tool call: execute_shell_command("mkdir -p /opt/cpw_simple_085")
             └→ Command executed successfully.
[Step 5]  Assistant:
           Tool call: write_file("/opt/cpw_simple_085/server.py", <583 chars>)
             └→ Wrote 583 bytes.              ← 正确的 async def handle_connection 写法
[Step 6~7] 启动 server 遇到端口冲突（OSError: [Errno 98] address already in use）
[Step 8]  Assistant:
           Tool call: execute_shell_command("ps -ef | grep LISTEN | grep 18085")
             └→ socat ... TCP4-LISTEN:18085    ← 主动检查谁占了端口
[Step 9]  Assistant:
           Tool call: execute_shell_command("kill -9 5790")
             └→ Command executed successfully.  ← 清除冲突进程
[Step 10~11] 重新启动 server + 再次 ps -ef 确认就绪
[Step 12] Assistant:
           Tool call: python3 -c "socket.connect(('127.0.0.1', 18085)); s.send(b'ping CPW')..."
             └→ Command executed successfully.  ← 连接成功
[Step 13] Assistant:
           Tool call: grep_search("ping CPW", "/opt/cpw_simple_085")
             └→ output.txt:1:> ping CPW         ← 验证结果正确
```

→ 任务结束，**3 个 check 全过**，score = 1.0。

**对照表**（同一个任务，base policy vs step 19 的 policy）：

| 步骤 | base policy 的做法 | step 19 policy 的做法 | 关键差异 |
|---|---|---|---|
| async 写法 | `SyntaxError` / `AttributeError` | 正确的 `async def` + `asyncio.run()` | 学会**写能跑的异步代码** |
| 连接前检查 | 不检查，直接放弃 | `ps -ef \| grep LISTEN` 确认就绪 | 学会**验证服务状态** |
| 看到错误后 | 忽略 stderr，放弃 | `kill` 冲突进程 → 重试 | 学会**根据 tool_result 调整** |
| 任务完成度 | 未完成（text only） | `grep` 验证 output 正确 | 学会**端到端走完全流程** |

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

## 2.6 动手试试

> 以下脚本让你亲手查看 trajectory 内部。即使不跑训练，也能直接用预置数据体验。

```bash
# 方式 1：用预置的 simple_085 样本（无需 API key）— 查看 base policy 的失败案例
python scripts/tutorial/ch2_inspect_trajectory.py --sample

# 方式 2：对比 step1（失败）和 step19（成功），看行为变化（推荐！）
python scripts/tutorial/ch2_inspect_trajectory.py --sample --compare

# 方式 3：用你自己 ch1 跑出的数据
python scripts/tutorial/ch2_inspect_trajectory.py \
  --session checkpoints/<你的实验>/step_-1_rollout/simple_085/<sandbox_id>/session.json

# 方式 4（在线版，需要 API key）：自己跑一条新的 trajectory
python scripts/tutorial/ch2_rollout_single.py --output ./my_trajectory/
```

> 方式 4 需要 `TINKER_API_KEY` 和 `E2B_API_KEY`，如果没有可以跳过——方式 1~3 用预置数据已经能完整体验本章内容。

---

## 2.7 这一章你应该带走的

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
