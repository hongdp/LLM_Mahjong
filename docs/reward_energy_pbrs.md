# 能量一致的奖励设计：势函数塑形（PBRS）

> 2026-08-01 引入。动机：`MahjongStepReward` 的步级奖励是**绝对分**（最优切牌 +2/步），
> 与终局结算不自洽——20 步轨迹光整形就能攒 +40，而结算量级只有 ±4，真目标被淹没，
> 且「多走步数」本身可刷分。本文档给出严格修正及其数学保证。

## 能量（势函数）定义

对 13 张等效手牌 h：

```
Φ(h) = −C_SHANTEN·shanten(h) + C_UKEIRE·|ukeire(h)|
     = −2.0·向听数 + 0.05·受入种数
```

约束 `C_UKEIRE × 34 < C_SHANTEN`（0.05×34 = 1.7 < 2.0）：受入项**永远无法颠倒
向听排序**，与 `evaluate_discards_ranked` 的字典序原则一致（SKILLS.md 教训：
纯受入贪心会拆搭子退向听）。

## 塑形项（对单个玩家的轨迹）

记 afterstate 能量 ψ_i = Φ(第 i 步动作后的手牌)，ψ_pre = 初始手牌可达的最优能量：

```
F_0     = γ·ψ_0   − ψ_pre
F_i     = γ·ψ_i   − ψ_{i−1}        (0 < i < n−1)
F_{n−1} =    0    − ψ_{n−2}        （终局 afterstate 能量 := 0，
                                     终局价值全部由结算承载）
```

## 一致性保证

折扣和严格 telescoping：

```
Σᵢ γ^i·F_i = −ψ_pre        （与所采取的动作序列无关）
```

因此 **整形总和 = 发牌决定的常数**，塑形后的回报 = 真实结算回报 + 常数：

1. **中间奖励与最终奖励一致**：RL 最大化的目标与「赢点棒 + 顺位」完全同向；
2. **无法刷分**：任何策略的塑形总和相同（`test_policy_independent_total`）；
3. **最优策略不变**：Ng, Harada & Russell (1999) 的 PBRS 定理；
4. **信用分配仍然即时**：坏切牌在当步就吃到 γ·ψ_i − ψ_{i−1} 的负项
   （`test_bad_discard_scores_lower_at_the_step`）。

格式/合法性罚项（−10 无 action 标签、−5 打不存在的牌）保留为**约束项**，
显式位于能量体系之外——合规率到 1.0 后自动消失。

## 已知近似（有意为之）

- 非法动作会被引擎回滚强制 fallback，塑形仍按“尝试的 afterstate”计算；−5 约束罚项
  主导误差。
- 副露/荣和等中断决策点手牌不变，能量即手牌自身能量。
- ψ_pre 随发牌不同而变化，在跨对局的 advantage 归一化中只增加方差、不引入偏置。
  未来可选改进：按局减去 ψ_pre 修正项。

## 使用方式

```jsonc
// config JSON 中：
"reward_model": "potential"   // 默认 "step"（旧模型，保证既有实验可复现）
```

选择逻辑在 `src/tasks/mahjong/rewards.py::REWARD_MODELS`（registry 模式，符合
CLAUDE.md 模块化红线）。实现：`MahjongPotentialReward`；
测试：`tests/test_potential_reward.py`（telescoping / 反刷分 / 步级判别 / 约束项，6 项全绿）。

## 状态

- 代码已合入，**默认关闭**（`step`）。
- 进行中的 GCP run `v2_engine_full_run_20260802_005918` 使用旧奖励，不受影响
 （其代码在启动时已同步，本变更也不改默认行为）。
- 启用它属于**新实验设计**：需新的 EXPERIMENT.md、预注册成功标准并经用户确认后
  才能开跑（建议与本轮 run 的结果做同基线对比）。
