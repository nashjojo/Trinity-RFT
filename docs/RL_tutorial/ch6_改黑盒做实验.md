# 第 6 章：改黑盒 — 换任务、换模型、换 reward，看会发生什么

> **本章模式：开放实验**。前 5 章你已经从"看曲线"到"理解全链路"。这一章给你 6 个**单变量 ablation**，每个实验只改一两个字段，跑完和 baseline 对照。**这是真正学到 RL 直觉的方式——不是看公式，是看曲线变化**。

---

## 6.0 怎么用这一章

每个实验给你：

1. **Hypothesis**：你猜会发生什么；
2. **要改的字段**：精确到 yaml 的某一行；
3. **怎么对比**：把哪几条曲线叠在一起看；
4. **预期结果**（基于已跑过的实测数据）；
5. **延伸思考**：实验结果给你的启发。

前置：

- 第 1–5 章读完且至少跑通过一次 baseline；
- 有第 1 章 baseline 的 19-step 曲线（叫 `baseline.png`）；
- ~9 小时空闲时间跑一次新实验。

> **不要一次改两个变量！** 每个 ablation 必须**单变量**，否则区分不出"提升来自 A 还是 B"。这是 RL 实验最容易踩的坑。

---

## 6.1 实验 A：开启 KL penalty（防训练发散）

### Hypothesis

> baseline 19 步内训练健康；如果跑更长后期会有 score 回落。开启 KL penalty 应该让训练更稳定，但短期可能学得稍慢。

### 改什么

```yaml
algorithm:
  kl_loss_fn_args:
    kl_coef: 0.005
trainer:
  total_steps: 40
```

### 预期结果

| 指标 | baseline (kl=0) | new (kl=0.005) |
|---|---|---|
| Step 19 score | ~82 | ~78–80（略低）|
| Step 40 score | ~74（回落）| ~80（仍稳）|
| `ppo_kl` 全程均值 | 0.008 | 0.004（被压低）|
| `actor/kl_loss` | 0 | > 0 |

### 延伸思考

KL penalty 用 base policy 当锚，不让模型漂太远。代价是学得慢。**生产中 `kl_coef` 在 0.001–0.01 是甜区**。

---

## 6.2 实验 B：减小 G（每 prompt 采样数）

### Hypothesis

> baseline G=8 是性价比最优。G=4 advantage 噪声大、效果差；G=16 学得更稳但 wall time 翻倍。

### 改什么

```yaml
# 对照 1
algorithm:
  repeat_times: 4
buffer:
  train_batch_size: 32

# 对照 2
algorithm:
  repeat_times: 16
buffer:
  train_batch_size: 128
```

注意 `train_batch_size = batch_size × repeat_times` 必须同步改。

### 预期结果

| 指标 | G=4 | G=8（baseline）| G=16 |
|---|---|---|---|
| Step 19 score | ~76（差）| ~82 | ~83（略好）|
| `reward_std/min` | < 0.10（不稳）| ~0.12 | ~0.13 |
| 每 step wall time | 14 min | 17 min | 28 min（×2）|
| 9 小时跑多少 step | 38 | 19 | 9 |

### 延伸思考

- G 太小 → group_std 噪声大；
- G 太大 → wall time 不划算；
- **G=8 是 v22 调好的最优**，你的任务可能不同。

---

## 6.3 实验 C：换更小的 model（Qwen3-1.7B）

### Hypothesis

> 4B → 1.7B 容量减半。1.7B 可能学得更快但 ceiling 低。

### 改什么

```yaml
model:
  model_path: Qwen/Qwen3-1.7B
```

### 预期结果

| 维度 | Qwen3-1.7B | Qwen3-4B（baseline）|
|---|---|---|
| Step 1 base score | 更低（容量小，base 能力弱） | ~72 |
| Step 19 score | 提升幅度可能更小 | ~82 |
| 单 step wall time | 更快（模型小） | ~28 min |
| ceiling | 复杂任务可能卡住 | 能推过大部分任务 |

### 延伸思考

**不是越小越好**：小模型 base 能力不够，复杂任务会一直卡住。**4B 是 Agentic-RL 甜区**——既不会 base 太弱，也不会跑不起。

如果你想用更大的模型（如 7B/14B），需要同时调整数据集难度——否则 base policy 已经全部满分，advantage 全为 0，训练无效。

---

## 6.4 实验 D：换 reward 设计（0/1 vs 多 check）

### Hypothesis

> reward 退化成 0/1 会让 advantage 大量退化为 0，效果显著差。

### 改什么

新写一个 workflow：

```python
# examples/copaw_rl/queries_simple/workflows/queries_simple_workflow_binary.py
class QueriesSimpleWorkflowBinary(QueriesSimpleWorkflow):
    def calc_reward(self, result):
        score = super().calc_reward(result)
        return 1.0 if score >= 0.99 else 0.0
```

yaml：

```yaml
buffer:
  explorer_input:
    taskset:
      default_workflow_type: queries_simple_workflow_binary
```

### 预期结果

| 指标 | binary | baseline |
|---|---|---|
| Step 1 score | ~50（满分率）| 72.4 |
| Step 19 score | ~52 | 82.4 |
| 学习速度 | **几乎不动** | +10 pp |
| `valid_ratio` | 经常 < 0.5（很多 group 全 0 或全 1）| > 0.875 |

### 延伸思考

**这是 RL 最重要的教训之一**：reward 信号颗粒度直接决定能不能学。base policy 一道任务都 0/8 满分时，binary reward 下 advantage 全 0。**多 check 帮你从"半成品"挤出梯度信号**。

---

## 6.5 实验 E：换数据集（自己造一个新任务）

最有趣也最难的 ablation。最小例子：让模型学会**写一个 Python `add(a, b)` 函数并通过测试**。

### 步骤

1. 新建 `examples/copaw_rl/queries_simple/data/my_task/tasks.json`：

```json
[
  {
    "task_id": "my_add_001",
    "name": "实现 add 函数",
    "question": "在 sandbox /opt/my/add.py 里实现 add(a, b) 函数，返回 a+b。然后跑 python -c 'from add import add; assert add(3,4)==7' 验证。完成后写 anchor 文件 /opt/my/anchor.txt 内容 'OK'。最后输出【DONE】。",
    "answer": "",
    "fields": {"FILE": "/opt/my/add.py", "ANCHOR": "/opt/my/anchor.txt"},
    "avg_score": 0.5,
    "score_uncertainty": 0.3
  }
]
```

2. 在 `test_cases/my_add_001/case.py` 写 verifier：

```python
def run_checks(sandbox):
    return [
        {"name": "file_exists", "passed": sandbox.exists("/opt/my/add.py")},
        {"name": "test_passes", "passed": sandbox.exec(
            "python -c 'from add import add; assert add(3,4)==7' && echo OK"
        ).strip().endswith("OK")},
        {"name": "anchor", "passed": sandbox.read("/opt/my/anchor.txt").strip() == "OK"},
    ]
```

3. yaml 指向新数据集：

```yaml
buffer:
  explorer_input:
    taskset:
      path: examples/copaw_rl/queries_simple/data/my_task
```

4. 跑 19 step 看会不会学。

### 注意事项

- **avg_score 必须填合理初值**（0.3–0.7）；
- **base policy 必须能至少做对 1 次**，否则 G=8 全 0 → 不学；
- **检查 reward 颗粒度**：必须密集（多 check），不要 0/1。

### 延伸思考

造任务的 80% 难点在 reward design。这是为什么本教程花了一整章（第 3 章）讲 reward。

---

## 6.6 实验 F：换训练后端（Tinker → TuFT）

### Hypothesis

> 同一份 yaml + 同样种子，把 `base_url` 从 Tinker 切到 TuFT，结果应该几乎完全一致（差异 ±1pp 内）。

### 改什么

```yaml
model:
  tinker:
    base_url: http://localhost:10610
```

启动本地 TuFT server。

### 预期结果

| 指标 | Tinker | TuFT |
|---|---|---|
| Step 19 score | 82.42 | 81–83 |
| `ppo_kl` 趋势 | 健康 | 健康 |
| 两条曲线在 19 步内 | 大体重合 | — |

### 延伸思考

这个 ablation 价值不在 RL，而在**软件工程**——验证 Trinity-RFT "算法-后端解耦"的承诺。生产环境下可随时根据成本切换后端。

---

## 6.7 6 个实验做完后的能力清单

如果都跑完了：

✅ 你**知道每个超参的边界**：lr / G / kl_coef 多大不该多大；
✅ 你**会自己造任务**：能从新业务问题造出可被 RL 训练的 task + verifier；
✅ 你**能切换后端**：根据成本和场景选 Tinker 或 TuFT；
✅ 你**真的"理解"了 RL**：不是公式上的理解，是"我看过这个变化、它就是这样"的肌肉记忆。

---

## 6.8 教程到此结束 — 接下来去哪

1. **更复杂的 reward**：看 [`examples/grpo_rubric_as_reward`](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/grpo_rubric_as_reward) — LLM 当 verifier；
2. **更长 horizon**：看 [`examples/grpo_alfworld_general_multi_step`](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/grpo_alfworld_general_multi_step) — household task；
3. **生产场景**：参考 v22 实验报告 [`docs/2026-06-08_agentic_rl_v22_tutorial.md`](../2026-06-08_agentic_rl_v22_tutorial.md) 看更完整的实验记录；
4. **后续实验**：v25 (G=16)、v26 (length penalty)、v27 (kl_coef=0.01) 单变量改进。

最后，欢迎给 [Trinity-RFT](https://github.com/agentscope-ai/Trinity-RFT) 提 issue / PR——让 Agentic RL 从"少数大厂能做的事"变成"每个有兴趣的工程师都能上手的事"。

---

**上一章**：[第 5 章：模型权重怎么更新](./ch5_权重更新.md) ｜ **返回**：[第 0 章：要不要 RL？框架选型？](./ch0_要不要RL与框架选型.md)
