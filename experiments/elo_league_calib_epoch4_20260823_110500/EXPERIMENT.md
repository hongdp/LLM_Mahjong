# Elo 锚点池纪元 4 重校（规则审查修复 + 场因素随机化 + 锚点扩池至 9 员）

- **Date**: 2026-08-23 11:05 PDT  **Status**: running
- **Git**: 合并 `engine/claim-audit`（ae2b5a0）：抢杠见逃振听、国士抢暗杠、起始点数/供托随机化、西场 10%、奖励改起点差分
- **Env**: 本地 RTX 4080 + 24 vCPU

## Purpose & Hypothesis
三项引擎变更触发纪元规则；同时把第一批冠军 **exp27-A（cnn_m_r 纪元 3）加入锚点池**（9 员），
成为池内第一个纯血纪元 3 锚点。假设：(1) 随机化的场上下文加大单局方差 → SE 略增但 <40；
(2) 排序与纪元 3 一致（Spearman ≥ 0.9）；(3) A 入池后评分 ≈ 其纪元 3 候选分 1069 ± 2σ。

## Method
`run_elo_league.py calibrate --deals 200 --seed0 20260824 --parallel 20`，9 锚点 36 对。
纪元 3 池存档 `experiments/elo_league_epoch3/`。复式牌按 seed 共享随机上下文（两方向同起点）。

## Success Criteria
Spearman(与纪元 3) ≥ 0.9；SE < 40；A 的池内分 ∈ 1069 ± 30。

## Progress
- [11:05] 合并 + 测试 + 发射。

## Results
（待）

## Conclusion
（待）
