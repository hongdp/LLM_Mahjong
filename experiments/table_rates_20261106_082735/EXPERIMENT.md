# 四方桌和牌率 / 放铳率（论文 L2）（2026-11-06）

- **Date**: 2026-11-06 08:30 PST  **Status**: running
- **Git**: d194f60；驱动 `table_flows.py`（11-04 flows 版 table4_mortal 加 MODE 参数），本目录
- **Env**: 本机 RTX 4080 + 24 核；$0

## Purpose & Hypothesis
论文会话 L2：三张四方半庄桌每个实体的每局和牌率、放铳率（以前只存了 uma 与顺位）。描述性。

## Method
- 原样重放三张桌（同种子、同座位轮换、同检查点），逐局记录 hora 事件（actor, target）：
  - **A**：exp124 A 132.7M 快照 / 76.2M 快照 / bc49 / exp113 C 384M，种子 101112000，1000 牌山 × 4 轮换，T=0（原为向量版 table4.py，四个轮换共用同一牌山 = MODE dup）；
  - **B**：Mortal 298k / bc49 / exp124 A 132.7M / exp113 C，种子 91000000，400 × 4（原为逐局 table4_mortal.py，轮换各用不同牌山 = MODE rot）；
  - **C**：同 B，exp124 换 171.2M 快照（11-06 桌）。
- **复现校验（已过）**：C 的 rot 模式前 3 牌山 12 场 uma 与存档 `table4m171_shard0.json` 逐场一致（12/12）；A 的 dup 模式 3 牌山各实体 uma 均值与向量版 table4.py 完全相同（27016.7 / −8683.3 / −11708.3 / −6625.0）。
- 指标：每实体 每局（per seat-deal）和牌率、自摸率、荣和率、放铳率；SE 按牌山聚类。
- **附带发现**：Mortal 桌（B、C 以及 11-01 的 45M 桌）的 `play_hanchan(seed=(seed0+w)*4+rot)` 让四个轮换各打不同牌山——不是复式配对；配对差仍无偏、按牌山组聚类的 SE 仍有效，只是没有同牌山方差削减。

## Success Criteria
描述性；完整重放的各实体 uma 均值须与存档一致（A 对 `table4_exp124A_133M_76M_bc49_exp113C_hanchan.json`，B/C 对 shard 文件）。

## Progress
- [2026-11-06 08:30 PST] 校验通过，发射 A / B / C 各 12 分片。

## Results

## Conclusion

## Artifacts
