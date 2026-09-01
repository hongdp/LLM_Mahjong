# exp59 — 价值方法线 v1：DQN on 人类先验（单局刻度）

- **Date**: 2026-08-31  **Status**: running（实现 + 本地冒烟阶段）
- **Git**: 基线 fcdbe3a；设计 [designs/design_dqn_value.md](../designs/design_dqn_value.md)
- **Env**: 本地 4080（冒烟）；正式 tranche 走 RunPod 3090Ti（exp58 runbook）

## Purpose & Hypothesis
exp55-D 证明策略梯度读不动排位价值信号（EV 0.09、漂移弥散、强度打平）。本系列检验：
**价值方法（DQN：直接回归 Q(s,a)，策略=argmax）能否在同一引擎与先验上有效学习**。
v1 只回答机器问题（单局刻度、常规奖励）；W 信用接 Q 目标的排位版是 v2。
结构性论点：①目标策略=贪心=部署形态（无采样税错位）；②回放重复采样磨高方差回报；
③league 冻结 T=0 三席 ⇒ 环境平稳（Q 收敛前提）。

## Method
- 新训练器 `scripts/train_dnn_dqn.py`：rollout 侧零改动（collect_parallel + GPU 批推理 +
  Boltzmann(Q,T=1) 行为），episodes → (s,a,r,s',mask',done) transitions → 环形回放（1M）
  → Double DQN + Huber，目标网硬更新。
- 热启动：bc49 trunk + policy 头照抄为 Q 头（argmax 保序 ⇒ 行为从 bc49 贪心起步）。
- 奖励：registry 原样，scale 0.05；γ=0.995。
- 对手：league={bc49, exp46I} 冻结 T=0，学习席 ×1（league_frac=1, learner_seats=1）。
- 监控：TD loss、Q 均值/回报均值比率、贪心快评（对 bc49）、风格九项。

## Config（冒烟）
batch 512、lr 1e-4、target_every 2000 次更新、回放 1M、每迭代 rollout 512 局 +
若干梯度步（更新数 : 新样本数 ≈ 1 : 8 起步）。快照见本目录 config 文件。

## Success Criteria（预注册）
- **冒烟（本地，~50k 局）**：①TD loss 收敛趋势（不振荡发散）；②Q/回报均值比率 ∈ [0.5, 2]
  （>2 = 过估计警报）；③贪心行为不崩：vs bc49 快评（n=400 单局，T=0）≥ 0.45。
- **正式（云，200k 局）**：贪心 vs bc49 单局 n=2000 ≥ 0.48 ⇒ 「DQN 在本引擎上能学」成立，
  批准 v2（半庄 + W 信用目标）；< 0.45 ⇒ 诊断（过估计/覆盖/量纲）后再决定。
- 失败也有价值：与 PPO 的对照钉死「是信号问题还是优化器问题」。

## Progress
- [08-31] 设计定稿；训练器实现中。

## Results
（待）

## Conclusion
（待）

## Artifacts
| Path | Size | Description |
|---|---|---|
