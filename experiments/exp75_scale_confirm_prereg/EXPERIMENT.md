# exp75 规模确认：纯血最强配方 4M 局单臂（$200 计划阶段 C 缩减版，2026-09-07）

- **Date**: 2026-09-07  **Status**: running（发射）
- **Git**: 见 Progress；RunPod Secure L40S 单 pod 单臂；纯血同 exp74
- **Env**: `train_dnn_ppo.py --shaping --shaping_schedule 0:1.0,400000:0.0`（exp74-A1 配方）+ `--total_games 4000000`

## Purpose & Hypothesis
exp74 的 H2 读数（A2 在 50–60 万局达 P3 终点水平后 40 万局持平）指向"1M 局平台 = 上限"。本实验用当前最强配方（exp74-A1，vs P3 0.517）
把预算放大 4× 做最终确认，产出纯血线 share-vs-局数 缩放曲线（$200 计划的交付物之一）。
**H0（上限）**：2M 局 ckpt 对 P3 终点 ≤ 0.53 且 1M→2M 斜率 < +1 pp ⇒ 早停，结论"设定上限"；
**H1（速度）**：2M ≥ 0.55 且仍在上升 ⇒ 跑完 4M，记录曲线，进入规模路线。

## Method
- 配方 = exp74-A1（P3 钉死旗标 + 塑形退火 @40 万局）；`--total_games 4000000`；熵表 `0:0.03,600000:0.01` 不变（地板 0.01 自 60 万局起）；
  milestones 每 50 万局；ckpt_every 25 迭代（≈5 万局）。
- 在轨：每 ckpt 对 P3 终点 T=0 n=600（`exp75_track_loop`，去掉同局数 P3 对照，P3 只有 1M）。
- 早停规则（预注册）：**2M 局**读数（20k 对终审口径）≤ 0.53 且相对 1M 局读数增幅 <+1 pp ⇒ 终止；否则 3M 再读一次同规则。
- 终评：终点 vs P3 / exp27A / bc49 双段 20k 对；若跑完 4M 加梯子 Elo。

## Success Criteria（预注册）
1. 缩放曲线（0.5M…终点 vs P3 终点）写入 FINDINGS，无论结果。
2. H1 成立仅当 4M 终点 vs P3 ≥ 0.55 且 vs bc49 ≥ 0.36（M1 方向）。
- 预算：≈80 局/s ⇒ 4M ≈ 14 h ≈ $15；2M 早停 ≈ $8。**上限 $20**。

## Progress
