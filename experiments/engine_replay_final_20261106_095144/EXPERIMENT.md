# 最终引擎重放 exp45 的 1520 局凤凰卓牌谱（论文 L4）（2026-11-06）

- **Date**: 2026-11-06 09:52 PST  **Status**: done（2026-11-06 10:10 PST）
- **Git**: 9ce250f（Python 引擎 table.py / claims.py 最后改动 13095b9 2026-08-27 = exp45 动作缺口修复；Rust 引擎与之逐位对拍）
- **Env**: 本机 CPU；$0

## Purpose & Hypothesis
论文会话 L4：在当前（最终）引擎上重放 exp45 中期保真门用的同一 1520 局（973,970 个人类决策，08-26 原读数非法 6 例 = 0.0006%，全部为当时的引擎动作空间缺口，08-27 修复），报告非法动作数 / 率；另报当前规则测试规模（pytest 用例数）。
预期：0 非法（修复后原 6 例已复放归零，见 docs/engine_known_issues.md）。

## Method
- 文件集：从 `data/tenhou/decisions_p1500.jsonl` 取 1520 个唯一 mjlog（全部在 raw 中找到），软链成临时 raw 目录，运行 `scripts/harvest_human_bc.py --raw <tmp> --out <scratchpad>`（逐座 MjaiDnnBot 回放、人类动作 ∈ 引擎合法集断言、预注册歧义弃样）。
- 规则测试：`pytest --collect-only -q` 计数，另列引擎规则相关测试文件的用例数。

## Success Criteria
描述性；决策数应与 08-26 一致（973,970 保留 + 6 非法）或说明差异。

## Progress
- [2026-11-06 09:52 PST] 文件集重建（1520/1520），发射重放。

- [2026-11-06 10:10 PST] 重放完成（≈15 min 单进程）；规则子集测试 138 通过 / 1 跳过（71 s）。

## Results
| 项 | 08-26（修复前引擎） | 11-06 最终引擎 |
|---|---|---|
| 局 / 座位回放 | 1520 / 6080，零崩溃 | 1520 / 6080，零崩溃 |
| 保留决策（turn / claim / chankan） | 973,970（775,747 / 198,220 / 3） | **973,976**（775,753 / 198,220 / 3） |
| 非法 | 6（0.0006%；5 例完成形拒和转立直未提供、1 例 4 副露加杠被误禁） | **0（0.0000%）** |
| 歧义弃样（预注册规则） | 4,722 | 4,722 |

总决策数一致（973,970 + 6 = 973,976）：原 6 例非法在修复后全部变为合法并被保留。
**规则测试规模**：`pytest --collect-only` 共 **345** 个用例（全仓）；其中引擎 / 规则 / 对拍相关 139 个（test_engine 49、test_majsoul_rules 23、test_rust_random_parity 13、test_mjai_bridge 11、test_claim_rules 10、test_hanchan 8、test_mjlog_to_mjai 6、test_rust_rollout_parity 5、test_hanchan_gen 5、test_rust_shanten_parity 3、test_oracle_rust_parity 3、test_ron_cost 3），本次实跑 138 通过、1 跳过。

## Conclusion
最终引擎在 97.4 万个凤凰卓人类决策上**零非法动作**（08-26 的 6 例均为当时的动作空间缺口，08-27 修复）。范围戳：Python 引擎（table.py / claims.py @13095b9）经 MJAI 回放桥；训练用 Rust 引擎由 24 个对拍用例与之逐位对齐；天凤 vs 雀魂规则差异本身不在此测量内（测的是"人类在天凤做出的每个动作，我们的引擎是否也允许"）。

## Artifacts
| Path | Size | Description |
|---|---|---|
| 本目录 `replay.log` | — | 重放统计与摘要 |
| 会话 scratchpad `p1500_final_decisions.jsonl`、`p1500_raw/`（软链） | — | 决策标签输出、文件集（临时，不入库） |
