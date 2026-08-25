# exp24e_majsoul_live — 雀魂实战（exp31-4ext，2.0M 收官）

- **Date**: 2026-08-24 00:01  **Status**: done（1 局东风战完成，服务已停）
- **Git**: ec117a0
- **模型**: `experiments/_cloud_ckpts/dnn_exp31_4ext_20260823/games_final.pt`
  = exp31-4 handset_lowent 续训 **2.0M 局收官**（1.0M→2.0M，handset_xl_cnn_m_r，entropy_coef 0.01 恒定）。
  **该 run 的 Elo 评级/竞技场判定尚未记入 experiments/dnn_exp31_4ext_20260823/EXPERIMENT.md**
  （仍是 launching 状态，判据 T1≥1032）；上次实战（exp24d）用的是同谱系 1.6M 中途快照。
- **模式**: 贪心 T=0，端口 8765，device cuda（该架构 CPU 249 ms/决策，GPU 7.7 ms）
- **前置校验**: `verify_mjai_bridge.py --games 12 --device cuda` OK（208 决策张量/合法集逐位一致）

## Purpose
观察 handset 续训收官（2.0M 局）在真人对局中的行为；与 exp24d（同谱系 1.6M 中途快照）、
exp24c（前身 1.0M）、exp24b（冠军 exp27-A）对照。**强度不作判定**：本 ckpt 尚未 Elo 评级。

## Progress
- [00:01] 服务启动（替换 exp24d 的 1.6M 中途快照；exp24d 已打 3 局东风战 11 小局，另行归档）。
- [00:32] 该半庄结束，**4 位**（−311/局，9 小局）；此后无新对局，会话结束（跨日无活动，服务已手动关闭）。

## Results（1 局东风战 / 9 小局，样本极小，ckpt 未 Elo 评级，不作强度判定）
| Metric | exp31-4ext 2.0M（本次） | exp24d（同谱系 1.6M 中途） | exp24c（前身 1.0M） | exp24b（冠军 exp27-A） |
|---|---|---|---|---|
| 顺位 | 4（n=1） | 2.0±0.7（1,3） | 3（n=1） | 3（n=1） |
| 和牌率 / 局 | 0.000 | 0.400 | 0.200 | 0.200 |
| 放铳率 / 局 | 0.111 | 0.200 | 0.200 | 0.200 |
| 立直率 / 局 | 0.000 | 0.364 | 0.000 | 0.000 |
| 副露率 / 局 | 0.000 | 0.091 | 0.600 | 0.600 |
| 平均点差 / 局 | −311 | +1690 | −640 | −9200 |
| 决策 | 110（dahai 83 / none 25 / reach 2） | 134 | 106 | 106 |

## Conclusion（中期，不可比强度）
- 桥接可用性达标：110 次决策 0 错误。
- 该 2.0M ckpt 表现出的风格最保守：本局立直 0、副露 0，9 小局全程几乎只推进牌效率、没有主动进攻，
  与同谱系 1.6M 中途快照（立直 36%、有副露）明显不同——可能是训练末段（1.6M→2.0M）风格漂移，
  也可能只是单局噪声（n=1）。**未评级、不可下强度结论**。

## Next Steps
- 待 `experiments/dnn_exp31_4ext_20260823/EXPERIMENT.md` 补上 Elo/竞技场判定后，再决定是否值得
  继续在雀魂上为这个 ckpt 累积样本；当前四次实战部署（exp24b/c/d/e）风格差异很大，
  建议下次实战优先用**已评级**的冠军（exp27-A）或已确认强于旧锚点的 checkpoint。

## Artifacts
| Path | Size | Description |
|---|---|---|
| mjai_session.jsonl | ~0.4 MB | 全量事件 + 110 条决策记录 |
| analysis.txt | — | analyze_majsoul_session.py 输出 |
