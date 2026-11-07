# R6：Mortal vs bc49 在 R1 新牌山上的复式份额（2026-11-07）

- **Date**: 2026-11-07 02:20 PST  **Status**: queued（akochan 主行完成后开始，与 L6b 并跑，先于 R5）
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

## Results

## Conclusion

## Artifacts
