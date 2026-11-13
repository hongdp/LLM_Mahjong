# R6：Mortal vs bc49 在 R1 新牌山上的复式份额（2026-11-07）

- **Date**: 2026-11-07 02:20 PST  **Status**: done（2026-11-07 20:00 PST）
- **Code**: L1 逐局驱动 `../paper_ref_rows_20261106_062737/dup_driver.py`（与 `paper_ref/mortal_merged.json` 同一桥、同一驱动；已对向量评测存档 300/300 校验）
- **Env**: 本机 CPU + 4080；$0
- **Approval**: 论文会话转达其用户 11-07 的批准

## Purpose & Hypothesis
摘要把 192M 的领先（30,500 牌山合并）与 Mortal 对比，但 Mortal 只在标准判决牌山上有读数。在 R1 的新牌山（101152000–101161999）上补 Mortal ⇒ 得到同牌山配对的 192M − Mortal。描述性。

## Method
- Mortal 298k（mjai 桥，原生贪心）坐 A 方位，bc49 T=0 坐 B 方位，1 万牌山，seed0 101152000，8 分片（每片 1250）。
- 合并为通常的 pairdiff 格式（`share`、`diff` 数组，按牌山顺序）。
- 配对：R1 的 192M / 160M / 133M / 112M 逐牌山文件。

## Success Criteria
描述性；桥回退率应为 0（L1 为 619,458 决策零回退）。

## Progress
- [2026-11-07 02:20 PST] 预注册；排队。

- [2026-11-07 18:55 PST] akochan 第一轮结束后自动开始（8 分片）；20:00 完成。

## Results
1 万新牌山（101152000–101161999），Mortal 298k（A）对 bc49 T=0（B）：**份额 0.5071 ± 0.0042，点差 +273 ± 117 / 牌山**；桥 590,616 决策 **0 回退**。

同牌山逐牌山配对（R1 的 exp124 A 逐牌山文件）：

| 模型 − Mortal | 份额差 | 点差 / 牌山 |
|---|---|---|
| 192M | **+0.64 ± 0.61 pp** | +122 ± 172 |
| 160M | +0.26 ± 0.61 | +120 ± 174 |
| 133M | −0.55 ± 0.61 | −139 ± 175 |
| 112M | −0.51 ± 0.61 | −99 ± 174 |

附：标准判决牌山前 1 万（101112000–101121999，`paper_ref/mortal_merged.json` 前 1 万 vs 192M 同牌山文件）192M − Mortal **+0.85 ± 0.61 pp**；两段合并（2 万牌山，逆方差加权） **+0.75 ± 0.43 pp（z 1.7）**。

## Conclusion
- Mortal 在新牌山上对 bc49 0.5071（标准牌山 0.5039），两段一致，Mortal 比 bc49 高 ≈ 0.5 pp。
- **192M 与 Mortal 同档、点估计略高**：新牌山 +0.64 ± 0.61 pp，合并 2 万牌山 +0.75 ± 0.43 pp（< 2σ），不能宣称超过 Mortal；112M → 192M 相对 Mortal 单调上升（−0.51 → +0.64）。
- **范围戳**：单局复式 2v2（对手 bc49）、T=0；Mortal 为原生贪心（按其半庄名次头训练），DNN 为单局点差目标；单种子。

## Artifacts
| Path | Description |
|---|---|
| `experiments/probes/pairdiff_R6_mortal298k_101152000.json` | Mortal 逐牌山（share / diff） |
| `experiments/probes/r6_mortal/mortal_shard{0..7}.json` | 分片 |
| 本目录 `merge.py` / `run.sh` / `result.txt` / `shard*.log` | 合并、日志 |

