# exp22-r2：联赛续训 700k→1.4M（样本饥饿假说 + 防守是否继续抬头）

- **Date**: 预注册 2026-08-22 21:40  **Status**: running
- **Git**: 4bc4b76  **Env**: mahjong-dnn-c3，统一基建（`--gpu_infer --gpu_infer_opponents`）
- **对照**: exp22 r1 终点（1041.1，defense_iq +0.016）；exp17-C（1079.7，+0.011）

## Purpose & Hypothesis
r1 在 700k 终点 ladder 仍陡升（563k 967 → 686k 1054），且联赛局只贡献 1-2 座轨迹（学习样本 ~62%）
——r1 很可能**样本饥饿**而非到顶。同时 defense_iq +0.016 是历代最高但不显著。假说：同一联赛设定
再训 700k（总 1.4M）⇒ ①评分追平/超过 GAE 镜像基线；②defense_iq 继续抬头并过 +0.03 线。
若评分追平但防守仍 ≈0 ⇒ 静态池联赛不足以改变生态（下一变量：PFSP 加权 / 自快照刷新 / frac 1.0）。

## Method / Config
`--resume <r1 games_final.pt>`（games 计数自 700k 继续），其余与 r1 全同：league pool v1（静态）、
frac 0.5、GAE 0.95、熵 0.01（台阶已过）、`--total_games 1400000 --milestones 1000000,1400000`。
**统一基建**：`--gpu_infer --gpu_infer_opponents`（对手池也上 GPU，首次云端使用）。

## Success Criteria（发射前定死）
1. 防守：defense_iq ≥ +0.03 且曝露放铳率 < 0.15（主）。
2. 强度：正式评分 ≥ 1079.7（追平 GAE 镜像基线）。
3. 趋势：defense_iq(1.4M) > defense_iq(700k)=0.016（方向性）。
4. 健康：吞吐 ≥ 30 局/s（多模型 GPU 托管实测）；KL/熵正常。

## Progress
- [2026-08-22 21:40] 预注册。
- [2026-08-22 22:10] 发射确认：`⏩ resume 700416 games`、league 7 对手、gpu_infer；**对手池首次 GPU 托管，
  38.8 局/s**（r1 对手走 CPU 为 45.0——小模型池上 GPU 慢 ~14%，与本机预测一致；统一基建裁定下接受）。
  心跳/ladder 已挂。**Status: running**，ETA ~5h。

## Results
| Metric | This run | r1 / exp17-C | Success criterion |
|---|---|---|---|
| （待运行） | | | |

## Conclusion
（待运行）

## Artifacts
| Path | Description |
|---|---|
| gs://llm-mahjong-experiments/dnn_exp22_league_20260822r2/ | 云端主目录 |
