# dnn_exp31_6_cnnm_lowent_20260824 — 2×2 补格：cnn_m + 恒定 0.01（无 schedule）

- **Date**: 2026-08-24  **Cost**: G4 flex ~$2.5  **Status**: launching
- 动机：b2_4（handset 恒定 0.01）自组织熵平台并学出立直，但 b2_3（cnn_xl 恒定 0.01）崩盘 →
  "schedule-freedom" 归因需要缺失格：**小模型 + 恒定低熵**。若 cnn_m 恒定 0.01 也崩（早期锐化
  → 副露锁死），则自组织平台是 handset 架构特有；若它也活，则"schedule 只是历史包袱"。
- Command: 冠军配方，仅 `--entropy_coef 0.01`（恒定，无 --entropy_schedule）。

## Success Criteria
- 主判读（归因）：熵曲线是否出现 ≥0.4 的自持平台 + riichi_rate ≥ 0.05。
- Elo 参考线：对照 exp31-5（T1 1031.9）；崩盘线 < 950。

## Progress
