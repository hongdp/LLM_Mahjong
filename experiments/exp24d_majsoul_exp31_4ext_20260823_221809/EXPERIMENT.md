# exp24d_majsoul_live — 雀魂实战（exp31-4ext 中途 ckpt，1.60M 局）

- **Date**: 2026-08-23 22:18  **Status**: running（服务已起，等打牌机接入）
- **Git**: 512d84a
- **模型**: `experiments/_cloud_ckpts/dnn_exp31_4ext_20260823/latest_interim.pt`
  = exp31-4 handset_lowent 续训（1.0M→2.0M）的**中途快照 games=1597440 / iter=780**，
  arch handset_xl_cnn_m_r，entropy_alpha 0.01，20.1M 参数，编码器 v1r。
  **该 run 仍在云端训练中**（目标 2.0M），此 ckpt 未经 Elo 评级；前身 1.0M 数 T1 1017.5。
- **模式**: 贪心 T=0，端口 8765，device cuda（该架构 CPU 249 ms/决策，GPU 7.7 ms）
- **前置校验**: `verify_mjai_bridge.py --games 12 --device cuda` OK（232 决策张量/合法集逐位一致）

## Purpose
观察 handset 续训中途快照（1.6M 局，熵高原是否开始变现）在真人对局中的行为；与
exp24c（同谱系 1.0M 数）、exp24b（exp27-A 冠军）对照。**强度不作判定**：ckpt 未评级、样本小。

## Progress
- [22:18] 服务启动（替换 exp24c 的 1.0M 版本；exp24c 已打 1 局东风战 5 小局，3 位，另行归档）。
