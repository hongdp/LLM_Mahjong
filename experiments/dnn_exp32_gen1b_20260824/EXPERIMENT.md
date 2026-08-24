# dnn_exp32_gen1b_20260824 — 跨代联赛 gen1 重试（league_frac 0.25）

- **Date**: 2026-08-24  **Cost**: G4 flex ~$3  **Status**: launching
- 触发预注册风险预案：gen1（frac 0.5）T1 926.3 ≪ 对照 1031.9，诊断 = 副露锁死
  （call_rate 95.2%、riichi 1.5%、defense_iq −0.144）——冻结强对手池挤压探索窗口，
  立直风格未孵化。gen1b 将对手池份额降到 0.25（75% 座位自对弈），让课程随学习者成长。
- Command: 同 gen1，仅 `--league_frac 0.25`（pool = exp27A_G0）。

## Success Criteria
1. T1 ≥ 1000（gen1 926.3 之上明显恢复）；理想线 ≥ 1032（追平非联赛冠军配方）。
2. riichi_rate ≥ 0.05 且 call_rate ≤ 0.85（脱离副露锁死）。
3. defense_iq > gen1 的 −0.144；若仍为负，第二阶段按预注册加 rank-bonus ×2 臂。

## Progress
