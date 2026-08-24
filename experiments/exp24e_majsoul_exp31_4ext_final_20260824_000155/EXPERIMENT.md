# exp24e_majsoul_live — 雀魂实战（exp31-4ext，2.0M 收官）

- **Date**: 2026-08-24 00:01  **Status**: running（服务已起，等打牌机接入）
- **Git**: ec117a0
- **模型**: `experiments/_cloud_ckpts/dnn_exp31_4ext_20260823/games_final.pt`
  = exp31-4 handset_lowent 续训 **2.0M 局收官**（1.0M→2.0M，handset_xl_cnn_m_r，entropy_coef 0.01 恒定）。
  **该 run 的 Elo 评级/竞技场判定尚未记入 experiments/dnn_exp31_4ext_20260823/EXPERIMENT.md**
  （仍是 launching 状态，判据 T1≥1032）；上次实战（exp24d）用的是同谱系 1.6M 中途快照。
- **模式**: 贪心 T=0，端口 8765，device cuda（该架构 CPU 249 ms/决策，GPU 7.7 ms）
- **前置校验**: `verify_mjai_bridge.py --games 12 --device cuda` OK（208 决策张量/合法集逐位一致）

## Purpose
观察 handset 续训收官（2.0M 局）在真人对局中的行为；与 exp24d（同谱系 1.6M 中途快照）、
exp24c（前身 1.0M）、exp24b（冠军 exp27-A）对照。**强度不作判定**：本 ckpt 尚未 Elo 评级。

## Progress
- [00:01] 服务启动（替换 exp24d 的 1.6M 中途快照；exp24d 已打 3 局东风战 11 小局，另行归档）。
