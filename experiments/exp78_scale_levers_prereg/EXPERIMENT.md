# exp78 突破缩放饱和：A 续训 + 学习率衰减 / B 大网络从零（2026-09-09）

- **Date**: 2026-09-09  **Status**: running（发射）
- **Git**: 见 Progress；两台社区 RTX 3090（$0.22/h 各）；纯血同 exp77（pod 上只有纯血 ckpt）
- **Env**: `train_dnn_ppo.py --engine rust`

## Purpose & Hypothesis
exp77 的缩放曲线 vs bc49 每倍增 +4.0 → +2.2 → +1.15 pp，24M→32M 持平（0.429），训练侧 KL 单调降到 0.0023、熵持平 0.51。
两个互斥的解释，各一臂：
- **A（学习率受限）**：从 exp77_X32 终点续训 32M→64M，学习率 1e-4 → **3e-5**（其余不变）。若 vs bc49 再升 ≥ +2 pp → 饱和是步长问题，续训路线成立。
- **B（容量受限）**：cnn_l_r（128 通道 × 4 块，4.0M 参数，2× cnn_m_r）从零、A1 配方、32M 局。exp41 的"大网络从零更差"是 1M 局的结论；
  32M 局下经典缩放律预期反转。若 B@32M vs bc49 > X32 的 0.429 ≥ +2 pp → 容量是瓶颈，后续按"更大网络 × 更多局"走。
**反假设**：两臂都 ≤ +1 pp ⇒ 单局自对弈 PPO 在 0.43–0.45 附近有真实的设定上限，需要新杠杆（对手多样性/异步大批量等）。

## Method
- A：`--resume exp77_X32/games_final.pt --total_games 64000000 --lr 3e-5`，其余 = exp77 旗标；milestones 36/40/44/48/52/56/60M；ckpt 每 400 迭代。
- B：`--arch cnn_l_r --total_games 32000000`，其余 = exp77 旗标（含退火塑形）；milestones 1/2/4/8/12/16/20/24/28M。
- 在轨：每 1M 倍数 ckpt 对 P3 终点与 bc49 各 n=600（`exp78_track_loop`）。早停：B 在 8M 处 vs bc49 < 0.36（明显慢于 X32 同期 0.396）⇒ 停 B。
- 终评：A/B 终点 vs bc49 双段 20k 对 + 梯子 Elo；A vs X32、B vs X32 各 20k 对。

## Success Criteria（预注册）
1. A：vs bc49 ≥ 0.449（X32 0.429 + 2 pp）→ 学习率路线成立；B：vs bc49 ≥ 0.449 → 容量路线成立。
2. 任一臂 ≥ 0.45 = **M2 达成**；≥ 0.50 = 目标达成（触发加冕流程：半庄 n≥1200 + Mortal）。
- 预算：A 32M 局 ≈ 11 h ≈ $2.5；B（大网络，16 vCPU 主机）≈ 16–20 h ≈ $4；**上限 $10**。

## Progress
- [09-09 08:52 PDT] **发射**：A = 社区 3090 pod `in85f6icots6rz`（32 vCPU，ssh 64.119.209.250:17426），从 exp77_X32 终点 `--resume`（含优化器矩），lr 3e-5，→64M；
  B = 社区 3090 pod `mqrlzqeoqyr6qd`（16 vCPU，ssh :19802），cnn_l_r 从零 32M。两 pod 各自构建 riichi_rs、GPU 数值校验通过。提交 427353e。
  心跳（25h 窗）×2、拉取 ×2（含 TB 镜像）、在轨循环（两臂 1M 倍数 ckpt vs P3 终点 + bc49）已挂；TB 加 exp78_A/B_LIVE。

