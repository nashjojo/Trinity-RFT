# 第 3 章：拆开黑盒 — reward 怎么算的

> **本章模式：拆解**。第 2 章你看到 trajectory 末尾会有 `score ∈ {0, 0.33, 0.67, 1.0}`。这一章讲：从那个 score，到 RL 真正用来训练的 reward，中间发生了什么。

---

## 3.0 为什么 reward 设计是 RL 的命门

RL 跑得好不好，**80% 取决于 reward 怎么算**。这不是夸张：

- reward 太稀疏（0/1）→ advantage 大部分时候为 0，梯度信号几乎为零；
- reward 太密集但有 hack 路径 → 模型学到"水分"行为（reward hacking）；
- reward 量纲飘忽 → advantage 的 normalize 出问题，训练发散。

本 tutorial 用的 `queries_simple v20_top8` 数据集已经把 reward 设计好了。这一章我们从外到内拆 4 层：

```
[verifier 输出 raw checks]
       ↓ 聚合
[task-level score ∈ {0, 1/N, 2/N, ..., 1}]
       ↓ trajectory 级处理（over_length / over_time）
[trajectory-level reward]
       ↓ group-level normalize
[advantage]   ← 这才是真正进入梯度的数（第 4 章详细讲）
```

---

## 3.1 第一层：verifier 跑 checklist

每个任务在 `examples/copaw_rl/queries_simple/test_cases/<task_id>/` 下有一个 `case.py`，里面是 N 个原子 check：

```python
# 以 simple_098（sqlite KV）为例
def run_checks(sandbox):
    checks = []

    # check 1: anchor 文件存在
    checks.append({
        "name": "anchor_file_exists",
        "passed": sandbox.read_file("/opt/cpw_simple_098/anchor.txt") == "CPW-SIMPLE-098",
    })

    # check 2: sqlite 表里有对应行
    checks.append({
        "name": "kv_row_exists",
        "passed": sandbox.exec("sqlite3 /opt/cpw_simple_098/kv.db 'SELECT v FROM kv WHERE k=...'")
                  == "CPW-VAL-98",
    })

    # check 3: OUTPUT_TXT 严格等于值
    checks.append({
        "name": "output_strict_match",
        "passed": sandbox.read_file("/opt/cpw_simple_098/result.txt").rstrip("\n") == "CPW-VAL-98",
    })

    return checks
```

3 个 check，每个返回 True/False。这是 reward 的最底层信号。

**关键设计原则**：

1. **每个 check 必须确定性**——同一个 sandbox 状态跑两次，结果必须一样；
2. **每个 check 必须由代码验证**——不能要求人工评判，否则没法 RL；
3. **check 之间应该正交**——每个 check 测一个独立的能力点，互不蕴含。

---

## 3.2 第二层：score = 通过 check 的比例

```python
score = sum(c["passed"] for c in checks) / len(checks)
# simple_098 有 3 个 check → score ∈ {0/3, 1/3, 2/3, 3/3} = {0.000, 0.333, 0.667, 1.000}
```

这就是第 2 章 session.json 里看到的 score。**为什么不是 0/1**：

| reward 形态 | 训练信号质量 | 适合场景 |
|---|---|---|
| 0/1（稀疏）| ❌ 弱：base policy 全 0 时 advantage 全 0，没法学 | base policy 已经能做对的简单任务 |
| 多 check 比例（密集）| ✅ 强：base policy 即便从未拿满分，也能区分"过 0 个 check"和"过 1 个 check" | **本教程的场景** |
| 加权 check 比例 | ⭐ 更强：不同 check 重要性不同，加权能更精细引导 | 进阶设计 |

**实测验证**：第 2 章看到 simple_085 step 1 的 score 分布是 `0.67 ×4, 0.33 ×4`——**没有一条满分**。如果用 0/1 reward，这 8 条 trajectory 的 reward 全是 0，advantage = 0，梯度 = 0，**这一步训练等于白跑**。但用密集 reward，0.67 比 0.33 高，模型仍能学到"哪些行为相对更好"。

> **教训**：如果你设计自己的任务（第 7 章），永远把 reward 拆成多个独立 check，不要给单一 0/1。

---

## 3.3 第三层：trajectory-level reward（含截断惩罚）

到这一步 reward 还没真正进入训练。trinity 在写入 experience buffer 之前还会做几件事：

### 3.3.1 over_length 处理

如果 trajectory 因为输出太长被截断（没跑完），verifier 跑不出有效结果，**reward 直接置 0**：

```python
if trajectory.over_length:
    reward = 0.0  # 不管 verifier 给了多少
```

这导致 §2.4 看到的现象：simple_085 早期 7.8% 的 trajectory 被 over_length 清零；step 19 降到 4.7%。

**为什么这么严苛**：模型必须学会**简洁完成任务**，不能靠"啰嗦地写一大堆 thinking"来 hack reward。

### 3.3.2 over_time 处理

类似地，如果 trajectory 在 sandbox 里运行超过 `MAX_TRAJ_SECONDS`（默认 1200 秒）也会被中断、reward 置 0。

### 3.3.3 transient_failure 处理

E2B sandbox 偶尔会因为网络抖动等返回临时错误。这种 trajectory 的 reward 通常也是 0，但归类为 `transient_failure`（与 over_length 区分），目前简单处理为 0。

### 3.3.4 最终 trajectory reward

```
trajectory_reward = task_score   if 正常完成
                  = 0.0          if over_length / over_time / transient_failure
```

这就是 explorer.log 里看到的 `rollout/score/mean`（×100 后变成 0–100）。

---

## 3.4 第四层：group-level normalize → advantage

**本节是预告**，下一章详细讲。

到 §3.3，每条 trajectory 都有一个 reward ∈ [0, 1]。但 RL 训练并不直接用这个 reward——而是用 **advantage**：

```
对于同一个 prompt 的 G=8 条 trajectory:
  group_mean = mean([r₁, r₂, ..., r₈])
  group_std  = std([r₁, r₂, ..., r₈])
  advantage_i = (rᵢ - group_mean) / (group_std + ε)
```

直觉：advantage > 0 的 trajectory 是"组内相对好的"，模型应该多学；advantage < 0 的是"组内相对差的"，模型应该少学。

**为什么不直接用 reward**：

- 不同 prompt 难度不同，绝对 reward 不可比（simple_098 的 0.5 vs simple_075 的 0.95 没法直接比）；
- group-relative 的归一化让 PPO/GRPO 在不同难度任务上行为一致。

详见第 4 章。

---

## 3.5 在我们实验中验证 reward 设计：看每个 step 的 reward 统计

trainer.log 里每个 step 都打印：

```
'experience_pipeline/group_advantages/reward_mean/mean': 0.685    # 8 个 group 的平均 reward
'experience_pipeline/group_advantages/reward_mean/max':  0.958    # 最容易的 group（一组 8 条 trajectory 的均值）
'experience_pipeline/group_advantages/reward_mean/min':  0.312    # 最难的 group
'experience_pipeline/group_advantages/reward_std/mean':  0.286    # 8 个 group 内 std 的平均
'experience_pipeline/group_advantages/reward_std/max':  0.502
'experience_pipeline/group_advantages/reward_std/min':  0.118
```

**怎么读这些数字**：

- `reward_mean/min` = 0.31 → 8 个任务里最难的那个，G=8 次采样平均也只有 31% 通过；如果出现 `reward_mean/min < 0.05` 警惕"全军覆没" → advantage 退化成 0；
- `reward_std/min` = 0.12 → 即使最齐整的 group，G=8 条 trajectory 内部也有差异；如果出现 `reward_std < 0.05` → 8 条 trajectory 全一样，无可学习的 advantage；
- 整个训练过程中这两个 min 都 > 0.05 是健康信号（v22 实验中都是这样）。

---

## 3.6 reward 设计的 3 个常见坑

如果你后面要自己设计任务（第 7 章），避开这些：

### 坑 1：reward hacking 路径

如果 check 写得不够严，模型会找捷径。比如：

```python
# ❌ 错误的 check
"output_contains_value": "CPW-VAL-98" in sandbox.read_file(...)

# ✅ 正确的 check
"output_strict_match": sandbox.read_file(...).rstrip("\n") == "CPW-VAL-98"
```

错误版本下，模型会输出 `"prefix CPW-VAL-98 suffix"` 拿到 reward。

### 坑 2：reward 与目标不对齐

某次实验我们想让模型"用 sqlite 完成任务"，但 check 只验证了"OUTPUT_TXT 内容对" → 模型直接 `echo CPW-VAL-98 > /opt/.../result.txt`，绕过 sqlite。

**修复**：增加一个 check 直接 `sqlite3 ... 'SELECT ...'` 验证表里真有数据。

### 坑 3：reward 量纲发散

如果不同 check 的"难度"不同，简单地 `mean(checks)` 会让模型只学最简单的 check。

**修复**：要么 check 设计成等难度，要么显式加权（高级用法）。

---

## 3.7 这一章你应该带走的

✅ **reward 是分层的**：raw checks → task score → trajectory reward → advantage。
✅ **多 check 比 0/1 强**：base policy 没满分时也能学到方向。
✅ **over_length / over_time 直接清零**：模型必须学会简洁完成任务。
✅ **group_std / reward_mean 是健康度看板**：min 都 > 0.05 = 训练有效。

❌ **你还不需要懂**：
- advantage 的具体公式与 PPO clip 怎么配合（→ 第 4 章）
- 这个 advantage 怎么变成梯度更新到 LoRA 权重上（→ 第 5 章）

**留个问题给自己**：同一个 prompt 的 8 条 trajectory，reward 分别是 `[1.0, 1.0, 0.67, 0.67, 0.67, 0.33, 0.33, 0.33]`。模型应该"学习"哪几条、"远离"哪几条？group-relative advantage 给出的答案，比你直觉想的更精巧。第 4 章揭晓。

---

**上一章**：[第 2 章：单次 rollout 内部](./ch2_单次rollout内部.md) ｜ **下一章**：[第 4 章：GRPO advantage](./ch4_GRPO_advantage.md)
