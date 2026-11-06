# exp56 跑批目录（记录本体在 prereg）

- **Date**: 2026-08-30  **Status**: running
- 完整预注册 / 进度 / 结果：[../exp56_hanchan_ladder_prereg/EXPERIMENT.md](../exp56_hanchan_ladder_prereg/EXPERIMENT.md)

## 本目录工件
| 路径 | 说明 |
|---|---|
| `config_calibrate.json` | 半庄 T=1 锚池校准配置快照（13 锚全循环，200 复式对/对） |
| `config_rate.json` | 候选 T=0 评分配置快照 |
| `chain.sh` | 阶段串行驱动（头对头 → T=0 池 → T=0 评分 → Mortal ×2） |
| `h2h_hanchan.py` / `rate_hanchan_board*.py` | 本轮使用的驱动脚本快照 |
| `logs/*.log` | 各阶段日志 |
| `mortal_vs_bc49_hanchan.json` | Mortal vs bc49 半庄 n=300（现行规则，零 fallback） |
