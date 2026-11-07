# R5：192M 四方半庄桌（Mortal / bc49 / exp124 192M / exp113 C）（2026-11-07）

- **Date**: 2026-11-07 01:50 PST  **Status**: queued（akochan 主行与 L6b 之后）
- **Code**: L2 的 `table_flows.py` / `summarize.py`（`../table_rates_20261106_082735/`，rot 模式）+ L2 勘误的修正和牌定义（`rates_corrected` 规则：自摸需本家得点；荣和赢家 = 除放铳者外所有得点座位）
- **Env**: 本机 CPU + 4080；$0
- **Approval**: 论文会话转达其用户 11-07 的批准

## Purpose & Hypothesis
把 C 桌（11-06 171M 桌）里的 exp124 171.2M 换成 **192M 终点**，其余三家（Mortal 298k、bc49、exp113 C 384M）与种子（91000000，400 × 4 = 1600 场，rot 模式）完全相同 ⇒ 171M → 192M 在同种子上配对。描述性。

## Method
- 同 C 桌：table_flows.py rot 模式，种子 91000000，400 牌山 × 4 轮换；DNN T=0，Mortal 原生贪心。exp124 检查点 = `exp124_A_p2/games_192000000.pt`（md5 见镜像清单）。
- 报告（L2 格式）：每实体 uma ± SE、平均顺位、一位率 / 四位率、修正定义下的和牌 / 自摸 / 荣和 / 放铳率；两两配对 uma 差矩阵；与 C 桌逐种子配对的 192M − 171M。
- 注意：Mortal 桌每个轮换用不同牌山（非复式，见 L2）；配对差无偏、按牌山组聚类 SE 有效。

## Success Criteria
描述性。

## Progress
- [2026-11-07 01:50 PST] 预注册；排队。

## Results

## Conclusion

## Artifacts
