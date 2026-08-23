# exp25_greedy_vs_sample — 同一冠军 贪心 vs T=1 采样（复式牌竞技场）

- **Date**: 2026-08-23 01:16  **Status**: running
- **Git**: f6ef75e + 本次未提交改动：arena 支持每边独立温度（`--dnn_temperature_a/_b`）
- **Env**: 本机 RTX 4080 / CPU，rlhf_mahjong；被测 `_cloud_ckpts/dnn_exp17c_gae_20260818/games_final.pt`

## Purpose & Hypothesis
雀魂实战（exp24）要选出牌温度。Elo 池与历次竞技场全部在 T=1 采样下测，贪心从未量过。
H1：贪心更强（实战日志 mean p(pick)≈0.73，采样把 ~1/4 质量落在次选以下，麻将无需混合策略）。
H0：无显著差异。反向风险：PPO 策略按 T=1 分布优化，贪心可能放大激进偏好。

## Method
`run_arena_dnn.py`：A = 同一 checkpoint 贪心（T=0, argmax），B = 同一 checkpoint T=1.0；复式牌配对，
200 副（seed0 20260823），parallel 12。唯一变量 = 温度。

## Success Criteria
配对点差 95% CI 不含 0 即判定；方向决定雀魂自动模式温度。CI 含 0 → 仍选贪心（方差更低、可复现）。

## Progress
- [01:16] 发射。
- [01:17] 200 副：+658.5±914.2，wins 185:145，CI 含 0 → 追加 1000 副（seed0 30000000）收窄 CI。
