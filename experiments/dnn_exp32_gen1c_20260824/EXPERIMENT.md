# dnn_exp32_gen1c_20260824 — gen1 熵杠杆臂（frac 0.5 + 拉长熵高位段）

- **Date**: 2026-08-24  **Cost**: G4 flex ~$3  **Status**: launching
- 与 gen1b 成对解耦 gen1 失败归因：gen1b 动课程杠杆（frac 0.25，schedule 不变），
  gen1c 动探索杠杆（frac 0.5 不变，`--entropy_schedule 0:0.03,800000:0.01` 延长孵化窗）。
  假设：强对手面前同样熵预算买到的探索更少，等效窗口提前关闭 → 拉长高位段应恢复立直孵化。

## Success Criteria
同 gen1b：T1 ≥ 1000；riichi_rate ≥ 0.05 且 call_rate ≤ 0.85；defense_iq > −0.144。
归因判读：若 b 好 c 差 → 课程主因；c 好 b 差 → 探索主因；都好 → 双杠杆可叠加（gen2 合并）。

## Progress
- 2026-08-24 收官（与 gen1b 归因配对）：**T1 940.4±12.0**（vs gen1 926.3，仅 +14，噪声内，未恢复）；
  riichi 1.3% / call 95.8%（几乎和 gen1 一模一样，锁死没解开）；defense_iq 0.008（≈0，未获益）。
  **归因判读：探索杠杆（拉长熵高位段到 800k）无效，课程杠杆（降 league_frac 到 0.25）有效**——
  gen1b 三项全过、gen1c 三项全不过，对照干净。机制解释：冻结强对手带来的训练信号会更快惩罚
  探索性（较弱）动作，纯粹拉长熵预算窗口填不平这个压力差，必须直接调小对手池占比。
  **结论：gen2 若继续，默认 league_frac 采用 0.25（而非拉长熵 schedule）。**
