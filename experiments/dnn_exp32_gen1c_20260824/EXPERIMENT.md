# dnn_exp32_gen1c_20260824 — gen1 熵杠杆臂（frac 0.5 + 拉长熵高位段）

- **Date**: 2026-08-24  **Cost**: G4 flex ~$3  **Status**: launching
- 与 gen1b 成对解耦 gen1 失败归因：gen1b 动课程杠杆（frac 0.25，schedule 不变），
  gen1c 动探索杠杆（frac 0.5 不变，`--entropy_schedule 0:0.03,800000:0.01` 延长孵化窗）。
  假设：强对手面前同样熵预算买到的探索更少，等效窗口提前关闭 → 拉长高位段应恢复立直孵化。

## Success Criteria
同 gen1b：T1 ≥ 1000；riichi_rate ≥ 0.05 且 call_rate ≤ 0.85；defense_iq > −0.144。
归因判读：若 b 好 c 差 → 课程主因；c 好 b 差 → 探索主因；都好 → 双杠杆可叠加（gen2 合并）。

## Progress
