# exp46（终稿预注册）— 强先验之上的 RL：补对真 Mortal 的 27 分

- **Date**: 2026-08-28 终稿（骨架 2026-08-26）  **Status**: armed（等 exp51 定型判决后发射）
- **谱系**: 人类先验谱系。用户指令（2026-08-28）：v3r2 定模型后 RL 补差距。

## Purpose & Hypothesis

exp50 量化了目标：旗舰 BC（1191.4）距真 Mortal（1218.6）27±17。风格分解显示差距
恰好在 Mortal 对人类先验的 RL 偏离处（立直 +5pp、副露 −4pp、流局站听 +6pp、放铳 −0.5pp）
——正是 RL-on-prior 应该学出来的东西。
- H1：KL-to-BC 锚臂爬得比裸臂更高/更稳（主要失败模式=RL 侵蚀先验，防守先被吃）；
- H2（侵蚀预言）：裸臂 defense_iq 随训练滑向 0；锚臂保住 ≥ 起点一半。

## Method

起点 = exp51 判决冻结的旗舰 ckpt（v3r 或 v3r2 的 conv46）。两臂 × 1M 局，本地 4080：
- **A 裸臂**：--init 旗舰 + lr 6e-5 + warmup 150 + 恒定熵 0.01（exp14 方法学现代版）；
- **B 锚臂**：A + --bc_anchor 旗舰 --bc_kl_coef 0.3（冒烟值：3 迭代 bc_kl 0.006→0.043 温和漂移）。
里程碑每 100k：ladder Elo（exp39 口径全程回归）+ defense_iq + 风格三件套。
基建：--bc_anchor KL 泄漏正则已实现并修复 where 反向 NaN 毒化（见 SKILLS 教训）。

## Success Criteria

1. **主判据**：任一臂 r300 Elo > 1191.4 + 20（RL 在 BC 之上净增益成立）；
   **达标线**：≥ 1218.6（Mortal 参照线）→ 北极星内尺达成（外尺=maka 实战另测）；
2. H2 判据：defense_iq 轨迹按预言分岔（裸衰减/锚保持）→ 锚进配方；
3. 风格侧写：立直率是否向 Mortal 的 0.233 方向移动（探索性，不作判据）；
4. 双臂均 ≤ +20 → "BC 即近上限"入账，杠杆转数据扩容（10 万局）与多局结构。

## Progress
- [2026-08-28] 基建三件套冒烟通过（--init warm-start / 46 槽 rollout / KL 锚）；
  发现并修复 torch.where 未选分支 -inf 毒化反向梯度的 NaN（值有限、梯度 nan——
  与熵项的 safe 防护同源教训）。等 exp51 判决。
