# exp81 v3r 线续训：V(32M) → 64M，学习率 1e-4→3e-5（exp78-A 配方移植到 v3r）（2026-09-10）

- **Date**: 2026-09-10 08:00 PDT（预注册）；09:45 PDT 发射  **Status**: running
- **Git**: f4d4643+（riichi_rs uint8 平面 + v3r 编码器）；社区 RTX 3090 Ti；纯血谱系
- **Env**: `train_dnn_ppo.py --engine rust --arch cnn_m_v3r --resume V_final.pt --total_games 64000000 --lr 3e-5`

## Purpose & Hypothesis
exp79（v3r 从零 32M）在轨九点全部领先 X32，24/28M 达 0.4525/0.459（X32 同期 0.404/0.445）。exp78-A 证明 v1r 线上 lr 1e-4→3e-5 续训 32M 局 +1.6 pp
（0.429→0.445）。把同一收尾步骤用到 v3r 线：**H1**：W(64M) vs bc49 ≥ V(32M) + 1.5 pp，且若 V ≈0.45–0.46 则 W ≥ **0.47**；**H0**：≤ V + 0.5 pp（v3r 线在 32M 已到平台）。
副问题：v3r 线的学习率衰减增益是否大于 v1r 线（更丰富输入 ⇒ 更多可细化的结构）。

## Method
- 起点 `exp79_V/games_final.pt`；`--resume` 续 32M 局至 64M；`--lr 3e-5`、熵 0.01 恒定（exp80 判定熵地板无关），其余 = exp79 旗标（塑形已归零）。
- milestones 36/40/…/60M，ckpt 每 400 迭代；在轨每 4M vs P3 终点 + bc49（n=600）。早停：40M 处 vs bc49 < 0.42 ⇒ 停（低于 V 终段带）。
- 终评：W vs bc49 / V / A 各 20k 对、梯子、风格探针 @bc49 桌。

## Success Criteria（预注册）
1. vs bc49 ≥ V + 1.5 pp（且 ≥ 0.47 若 V ≥ 0.455）→ 收尾退火在 v3r 线成立；≥ 0.50 = 总目标达成（走加冕流程：双方 T=0 显著为正 + 半庄 n≥1200）。
- 预算：32M 局 / (≈900 局/s，v3r uint8 版) ≈ 10 h × $0.27 ≈ **$2.7，上限 $5**。

## 门的偏离声明（2026-09-10 09:45 PDT）
exp79 终评 vs bc49 = 0.4394，**未达**本预注册的发射门 0.449。仍发射的理由：(1) v3r 是剩余杠杆中唯一带正号（+1.0 pp，z≈2）且风格有实质变化（放铳 0.230→0.163）的；
(2) 本实验直接回答"v3r 线渐近是否高于 v1r 线"（A(64M) 0.4453 是 v1r 线在同 64M 预算下的读数）。判据相应改为：**W vs bc49 ≥ 0.46**（高于 v1r 线渐近 ≥1.5 pp）成立；
0.445–0.46 中性（两线渐近相同）；< 0.445 负。预算不变（≈$2.7，上限 $5）。

## Progress
- [09-10 10:05 PDT] 发射受阻：RunPod MCP `create-pod` 对与 1 小时前完全相同的请求体持续返回 `400 Provide imageName, or templateId`（插件后端在 v1/v2 schema 间切换：`delete-pod` 一度要 `podId`、`create-template` 运行时要 `imageName`，而公布的 schema 仍是 v2 `body`）。
  已试：v2 body、v2 body 加 startSsh、v1 扁平字段（被类型校验拒）、body 内混合 v1/v2 字段、切换 GPU 型号——均失败。exp82 不受影响（已在跑）。计划：exp82 88M 读数到时重试；若仍失败，等插件恢复后再发。
