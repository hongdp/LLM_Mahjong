# exp2_arena_20260808_1050

- **Date**: 2026-08-08  **Status**: running  **类型**: 评估 run（exp2_settlement_vs_pbrs 的预注册主判据执行）
- **判据出处**: 两臂 EXPERIMENT.md（exp2_settlement_20260807_070806 / exp2_pbrs_20260807_065729）Success Criteria #2/#3 —— 本 run 只是执行，不新增判据。
- **对局**: 3 场，各 64 副复式对局牌（duplicate deals，seed0=20260802 与 exp1 竞技场同一副牌集）× 双方向 2v2 对角座位，RAW 终局点数配对差分 + 95% CI：
  1. **s_vs_p**（主判据）：S 臂 checkpoint_epoch_50 vs P 臂 checkpoint_epoch_50
  2. s_vs_anchor：S 臂 ep50 vs SFT 锚点
  3. p_vs_anchor：P 臂 ep50 vs SFT 锚点
- **Checkpoint 规则**（预注册在两臂文件）：竞技场用最终 epoch（ep50），不用 best-by-reward（S 臂零和使该指标无意义；P 臂 top-3 全在 ep13-16 恰证明 PBRS reward 曲线与强度脱钩）。
- **Env**: 临时 flex-start A100（mahjong-flex-a, us-central1-b），代码 ee0c60c，bf16 base + 双 named adapter，engine=post-audit RCR（与两臂训练一致）。
- **Results sink**: `gs://llm-mahjong-experiments/exp2_arena_20260808_1050/`（每场打完即增量上传）；EXIT-trap 关机自毁。
- **判定表（预注册）**:
  - s_vs_p CI 不含 0 → 胜方 reward 模式成为默认；含 0 → 无差异（有效结论，转投 critic 路线）
  - 任一臂 vs 锚点显著为正 → 首次录得 RL 真实强度增益；否则重现 exp1 的 null。

## Progress
- [2026-08-08 ~10:50 UTC] 预注册完成，VM 创建中。

## Results
(pending)

## Conclusion
(pending)
