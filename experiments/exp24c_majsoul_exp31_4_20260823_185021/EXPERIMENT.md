# exp24c_majsoul_live — 雀魂实战（exp31-4 handset_lowent）

- **Date**: 2026-08-23 18:50  **Status**: running（服务已起，等打牌机接入）
- **Git**: d227b39
- **模型**: exp31 臂 4 `handset_xl_cnn_m_r`（warmup 150 / entropy 0.01），20.1M 参数，编码器 v1r
  `experiments/_cloud_ckpts/dnn_exp31_4_handset_lowent_20260823/games_final.pt`
- **已知强度（用户请求前已在案）**: Elo T=1 **1017.5 ± 12.1**（冠军 exp27-A T=1 1069.2）；
  500 副竞技场对旧锚点 exp22-r2 **−2170 ± 1479（显著为负）**。即本模型比冠军和旧锚点都弱——
  用户明确要求部署实战，按请求执行；本次会话定位为**风格/行为观察**，不作强度比较用途。
- **模式**: 贪心 T=0，端口 8765，**--device cuda**（CPU 249 ms/决策，GPU 7.7 ms；本机 GPU 空闲 15 GB）
- **前置校验**: `verify_mjai_bridge.py --games 12 --device cuda` OK（188 决策张量/合法集逐位一致）

## Purpose
观察 handset_xl 架构 + 低熵配方在真人对局中的行为（副露/立直风格、放铳），与 exp27-A（exp24b）和
exp17-C（exp24）对照。强度判定不使用本会话数据（模型已知偏弱且样本小）。

## Progress
- [18:50] 服务启动。
