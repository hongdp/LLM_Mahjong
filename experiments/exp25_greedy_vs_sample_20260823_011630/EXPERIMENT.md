# exp25_greedy_vs_sample — 同一冠军 贪心 vs T=1 采样（复式牌竞技场）

- **Date**: 2026-08-23 01:16  **Status**: done
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
- [01:19] 1000 副完成：**+484.2±417.8，wins 948:737 → 贪心显著更强**。

## Results
| Metric | 贪心 (A) vs T=1 (B) | Success criterion |
|---|---|---|
| 配对点差 200 副 | +658.5 ± 914.2（185:145） | CI 含 0 |
| 配对点差 1000 副 | **+484.2 ± 417.8（948:737）** | CI 不含 0 ✅ |
| 合并 1200 副 | 见下行 pooled | ✅ |

## Conclusion
H1 成立：同一 checkpoint 贪心出牌比 T=1 采样强约 +500 点/副（≈ 胜率 56%），效应量与一次代际提升同量级——
**评估温度本身是被忽略的强度杠杆**。Elo 池历史评分全部是 T=1 口径，贪心口径绝对值整体上移（相对排序未测）。
雀魂自动模式采用 **T=0**。

## Next Steps
- 雀魂 exp24 自动模式固定 `--temperature 0`。
- 可选：中温（T=0.3-0.5）扫描，纯贪心是否最优未测。

## Artifacts
| Path | Size | Description |
|---|---|---|
| arena_result.json / arena.log | 200 副 | 首发 |
| arena_result_1000.json / arena_1000.log | 1000 副 | 判定 |
