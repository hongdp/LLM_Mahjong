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
- 2026-08-24 收官：**三条判据全过**。T1 1005.7±12.1（≥1000 ✓，vs gen1 926.3 恢复 +79.4，
  但仍低于非联赛冠军配方 1031.9）；riichi 11.1% / call 73.8%（✓ 脱离 gen1 的极端锁死 call 95.2%/riichi 1.5%）；
  **defense_iq +0.025**（✓ 越过 gen1 的 −0.144，逼近全项目史上最高 exp22 联赛 +0.016，
  是史上第二高读数）。竞技场 vs exp22r2 明显落后（−2286.8±1466.8，331:540，"B stronger"）——
  强度有代价，但这是本项目防守涌现主线目前最强的正面信号。等 gen1c（探索杠杆臂）收官后
  做课程 vs 探索归因；若 c 弱 b 强，判课程（frac）为主因，gen2 沿用 frac 0.25。
