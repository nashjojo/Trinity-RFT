# 第 4 章：拆开黑盒 — GRPO 的 advantage 怎么来的

> **本章模式：拆解**。第 3 章你看到 reward 是 ∈ [0, 1] 的连续值。这一章讲 GRPO 怎么把 reward 变成 advantage，以及为什么这个变换让多步 agentic 任务真的能学起来。

---

## 4.0 为什么不能直接用 reward 当训练信号

最朴素的想法：reward 高就鼓励、reward 低就抑制。代码大概是这样：

```python
loss = -reward * log_prob(trajectory)  # ❌ 朴素 REINFORCE
```

这在玩具任务上能跑，但在真实场景里会爆炸：

| 问题 | 现象 |
|---|---|
| **量纲不可比** | simple_075 的 reward 平均 0.95，simple_098 平均 0.50。直接相乘梯度被简单任务支配。|
| **方差大** | reward 在 [0, 1] 范围内波动，没 baseline 减去；梯度噪声极大。|
| **没有"组内相对"信号** | 所有 reward 都 > 0 时，模型会无差别"鼓励所有 trajectory"，无法学到"什么更好"。|

GRPO（Group Relative Policy Optimization）就是为了解决这三个问题设计的。

---

## 4.1 一句话讲清 GRPO

> **同一个 prompt 采 G 条 trajectory，把每条 trajectory 的 reward 减去这 G 条的均值再除以标准差，就是 advantage。**

```
对于一个 prompt p，采样 G=8 条 trajectory:
  rewards = [r₁, r₂, ..., r₈]
  μ = mean(rewards)
  σ = std(rewards)
  advantageᵢ = (rᵢ - μ) / (σ + ε)
```

ε（epsilon）是个小常数防除零，trinity 默认 0.1。

**直觉**：

- 比组内平均好 → advantage > 0 → 鼓励
- 比组内平均差 → advantage < 0 → 抑制
- 组内全一样（σ=0）→ advantage 全 0 → 这一组**不学**（无信息）

**对比经典 PPO**：经典 PPO 用 critic 网络估计 baseline；GRPO 直接用"组内均值"做 baseline，**不需要训练 critic**——这是 GRPO 的核心简化。

---

## 4.2 用一个具体例子算一遍

回到第 3 章末尾的例子。同一个 prompt 的 8 条 trajectory：

```
rewards = [1.0, 1.0, 0.67, 0.67, 0.67, 0.33, 0.33, 0.33]
μ = mean = 0.625
σ = std  ≈ 0.286
advantages = (r - μ) / (σ + 0.1)
           = (rewards - 0.625) / 0.386
           ≈ [+0.97, +0.97, +0.12, +0.12, +0.12, -0.76, -0.76, -0.76]
```

读这组数：

| 原 reward | advantage | 解释 |
|---:|---:|---|
| 1.0 | **+0.97** | 鼓励力度大：这条比组里平均好得多 |
| 0.67 | +0.12 | 轻微鼓励：比平均稍好一点 |
| 0.33 | **-0.76** | 强力抑制：比平均差得多 |

**关键洞察**：哪怕 reward 全 ≥ 0.33（没有任何"完全失败"），GRPO 也能从 *相对差异* 里挖出梯度信号。这就是它在 Agentic-RL 任务上比朴素 REINFORCE 好得多的原因。

**反例对比**：如果是 0/1 reward 退化版（见 §3.6），8 条 trajectory 的 reward 会是 `[1, 1, 0, 0, 0, 0, 0, 0]`，μ=0.25, σ≈0.43，advantage 是 `[+1.4, +1.4, -0.47, ...]`——能学，但信号粒度比多 check 版本差很多。

---

## 4.3 在我们实验里看 advantage 的健康度

trinity 的 explorer.log 每步打印：

```
'experience_pipeline/group_advantages/reward_std/mean':  0.286   # 8 个 group 内 std 的均值
'experience_pipeline/group_advantages/reward_std/min':   0.118   # 最齐整的 group 也有 std=0.118
'experience_pipeline/group_advantages/reward_std/max':   0.502
'experience_pipeline/selection/valid_ratio':             1.000
'experience_pipeline/selection/strict_valid_ratio':      0.875   # 8 个 group 中 7 个 std > 阈值
```

**怎么读这些 ratio**：

- `valid_ratio` = 1：所有 8 个 group 都有非零 advantage 进入训练（理想状态）；
- `strict_valid_ratio` = 0.875：8 个 group 中有 7 个的 std 大于一个更严格的阈值（trinity 默认 std_threshold=0.05）；
- 如果某一步 `valid_ratio` 跌到 0 → 所有 group 内 reward 都一样（要么全成功要么全失败）→ **这一步等于白跑**。

**v22 实验中**这两个 ratio 始终在 0.875 ~ 1.0，是健康的。如果你跑自己的任务发现 valid_ratio 持续 < 0.5，说明 G=8 不够多样化，可以增大 repeat_times（但代价是 wall time 变长）。

---

## 4.4 advantage 怎么进入梯度（PPO clip 概念）

到 §4.3 advantage 算好了。下一步，trinity 用 PPO 的 surrogate loss 把它变成梯度：

```python
# 简化版 PPO loss
ratio = exp(log_prob_new - log_prob_old)   # 新旧 policy 在同一 token 上的概率比
clipped = clip(ratio, 1 - eps, 1 + eps)    # eps = 0.2，把 ratio 限制在 [0.8, 1.2]
loss = -min(ratio * advantage, clipped * advantage)
```

**两件事一起发生**：

1. **advantage 决定方向**：advantage > 0 时这个 token 的 log_prob 会被推高；< 0 时会被推低。
2. **clip 决定步长**：如果某个 token 的概率比已经 > 1.2 或 < 0.8，clip 会限制更新幅度，防止单步走太远。

**为什么需要 clip**：

- 不 clip 时，advantage 大的 trajectory 会让某些 token 的概率比一路飙到 5、10 倍，等价于一次性"跳"到完全不同的 policy；
- clip 限制单步只能小幅修改 policy，强制让 RL 训练像"温和上爬"而不是"瞬间飞跃"。

trainer.log 里能看到这两个指标：

```
'actor/ppo_kl': 0.0095        # 新旧 policy 的平均 KL（每步）
'actor/pg_clipfrac': 0.0114   # 被 clip 截断的 token 占比
```

**v22 实验中**：

- `ppo_kl` 全程在 0.005–0.011，远低于 0.03 的过更新警戒线；
- `pg_clipfrac` 在 0.008–0.014，只有 ~1% 的 token 被 clip 触发；
- 两个指标都很健康 → 梯度信号既不发散也不被 clip 吞掉。

**警惕信号**：

| 指标 | 健康范围 | 异常解读 |
|---|---|---|
| `ppo_kl` | 0.005–0.020 | > 0.03 → lr 太大或 advantage 量纲炸了，立即 kill |
| `pg_clipfrac` | < 0.05 | > 0.10 → 大量 token 被 clip 截掉，训练效率低，多半是 lr 偏大 |

---

## 4.5 multi-step GRPO：trajectory 内每个 token 都拿同一个 advantage

agentic 场景的特殊点：一条 trajectory 有 N 个 step、每个 step 有 M 个 token。怎么把"trajectory-level 的 advantage"分配到 token 上？

trinity 的 `multi_step_grpo` 用最简单的方案：**整条 trajectory 的所有训练 token，共享同一个 advantage**。

```python
# 伪代码
for token in trajectory.assistant_tokens:  # 只对 assistant 产出的 token 算 loss
    loss += -advantage_traj * log_prob_new(token) * (PPO clip stuff)
```

→ 一条 trajectory 拿 +0.97 advantage，则它内部**所有** assistant token 的 log_prob 都被推高。

**为什么这个简单方案 work**：

1. **Credit assignment 难题被回避**：我们不需要决定"是第 3 步的 tool call 让任务成功还是第 7 步"，统一推所有 token；
2. **统计有效**：G=8 条 trajectory + 8 个 prompt = 64 条 trajectory 同时给信号，相对差异在统计上能正确把"通常导致成功"的行为模式推上去；
3. **实测可行**：v22 +10pp 的提升就是这么得来的。

**直觉的具体例子**：第 2 章那个 simple_085 step 19 trajectory，10 个 assistant turn 都拿到同一个正 advantage，意味着：

- 那个"用 `new_event_loop` 启动" 的 token 序列被推高 ✓
- 那个"sleep 1 等待就绪"的 token 序列也被推高 ✓
- 那个"成功 curl"的 token 序列也被推高 ✓

哪怕其中某些 token 其实是无关的（比如 thinking 里的废话），统计上"经常导致成功"的有效行为会胜出。

**进阶**：如果你想做 token-level credit assignment（process reward），trinity 也支持其他算法。但本教程用最简单的 trajectory-level，已经够好。

---

## 4.6 一个常被忽略的细节：KL penalty

完整的 GRPO loss 还有一项可选的 KL penalty：

```
loss = -PPO_surrogate(advantage) + β · KL(π_θ || π_ref)
```

其中 π_ref 是初始 policy（base model），β = `kl_coef`。这项的作用是**防止训练过程中模型漂离 base 太远**。

**两类 KL 别混淆**：

| 名字 | 公式 | 哪里用 |
|---|---|---|
| `actor/ppo_kl` | KL(π_θ_old \|\| π_θ_new)，新旧 policy 之间 | PPO clip 健康度监控 |
| KL penalty（本节）| KL(π_θ \|\| π_ref)，当前 policy vs base | loss 里加的正则项 |

本教程的 yaml 里：

```yaml
kl_loss_fn_args:
  kl_coef: 0.0   # 教学版关掉
```

**为什么教学版关掉 KL penalty**：

- KL=0 时模型可以更激进地更新，前 19 步信号最干净；
- 实际生产中你可能想 `kl_coef=0.001 ~ 0.01`，防止后期模型"漂走"或"灾难性遗忘"——尤其当 base model 的某些通用能力（比如多语言、coding 风格）你不想让 RL 训练破坏。

第 7 章会让你试着开启 KL，对比有什么区别。

---

## 4.7 这一章你应该带走的

✅ **GRPO 公式**：`advantage = (reward - group_mean) / (group_std + ε)`，本质是**组内相对**。
✅ **GRPO 不要 critic**：组内均值就是 baseline，比经典 PPO 简单一倍。
✅ **3 个健康度指标**：`reward_std/min`（每组要有差异）、`ppo_kl`（< 0.03）、`pg_clipfrac`（< 0.05）。
✅ **trajectory-level credit assignment**：整条 trajectory 共享 advantage，简单但有效。
✅ **KL penalty 是可选项**：教学版关掉，生产版可加。

❌ **你还不需要懂**：
- 这些 advantage 经过 PPO clip 后，具体怎么 backward 到 LoRA 矩阵 A、B 上（→ 第 5 章）
- 为什么 batch_size=64、mini_batch_size=9999 那行 yaml 配置很关键（→ 第 5 章）

**留个问题给自己**：advantage 已经算好了，剩下的就是"梯度怎么传到 LoRA 上"。LoRA 只有 ~17 MB 参数，每步只更新这部分——但训练用的 forward pass 仍然是 4B 全模型。这中间的"省钱魔法"具体怎么做？第 5 章揭晓。

---

**上一章**：[第 3 章：reward 怎么算的](./ch3_reward怎么算.md) ｜ **下一章**：[第 5 章：模型权重怎么更新的](./ch5_权重更新.md)
