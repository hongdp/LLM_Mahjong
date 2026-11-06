# 四方同桌半庄：exp124 A 171.2M + Mortal + bc49 + exp113 C（2026-11-06）

- **Date**: 2026-11-06 02:10 PST  **Status**: running
- **Git**: a829fc8（驱动 `table4_mortal.py` 复制自 11-01/11-04 两桌所用脚本，在本目录）
- **Env**: 本机 RTX 4080；conda rlhf_mahjong；$0

## Purpose & Hypothesis
用户 11-06 要求："拿 124 最新 checkpoint 和 mortal、bc 以及 cnn 打一桌测试"。
沿用 11-01（45M）与 11-04（133M）两桌的完全相同设置，只把 exp124 换成最新冻结快照，看 133M → 171M 在半庄口径是否继续上升。
CNN 选纯血线最强 CNN：exp113 C 384M（cnn_l，与前两桌同一个，保证可比）。

## Method
- 四方：Mortal 298k（mjai 黑盒桥，原生贪心）、bc49（`_anchors_epoch6/bc49.pt`）、exp124 A 171.2M（`probes/snap_exp124_A_games_171212800.pt`，md5 226e907a…）、exp113 C 384M；DNN 侧 T=0。
- 400 牌山 × 4 轮换 = 1600 场/模型，种子 91000000（同前两桌），8 分片 batch-1 驱动。
- 指标：每场 uma、平均顺位、一位率、四位率；配对差（同场）按牌山聚类的 SE。

## Success Criteria
描述性测试（无 H1/H0 门槛）。读法：exp124 − bc49、Mortal − exp124 的配对差与 133M 桌（+1028 / +298，SE ≈1.1k）比较；差异 < 2 SE 视为同档。

## Progress
- [2026-11-06 02:10 PST] 冻结快照、目录建立，发射 8 分片。

## Results

## Conclusion

## Artifacts
