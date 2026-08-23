# exp5_confirm（预注册；run dir 由 VM 生成后挂靠）

- **Date**: 2026-08-11 预注册  **Status**: pre-registered, pending launch
- **类型**: 确认性实验（replication）。exp4 阶梯赛在 6 场比较中发现 ep40 vs SFT 锚点 +2950（p=0.0027，Bonferroni 后仍显著），但同批的趋势检验 p=0.14 不支持「单调变强」。**发现与确认必须分开**：exp4 是探索性发现，本实验用**全新牌种子**做独立复现。

## 主假设（单一预注册检验，无多重比较问题）
**H1**: exp4 的 ep40 checkpoint 在**全新的 96 副复式对局牌**（seed0=20260811，与 exp4 的 20260802 无重叠）上仍显著强于 SFT 锚点。
- 判据：配对差分 95% CI 不含 0 且方向为正 → 确认；含 0 → **不能确认**（exp4 的发现降级为未复现的探索性结果，阶梯赛方法论结论相应减弱）。
- 功效估算：exp4 效应 +2950、32 副时 SE≈982；96 副 SE≈567。若真实效应仅为原估计的一半（回归均值后 +1475），z≈2.6，仍可检出。

## 次要问题（探索性，明确标注，不参与主判据）
| 对局 | 副数 | 问题 |
|---|---|---|
| exp4 ep35 vs 锚点 | 48 | 峰值形状：ep40 是尖峰还是平台？ |
| exp4 ep45 vs 锚点 | 48 | 同上（ep40→ep50 的下降是否单调） |
| exp2-S ep25 vs 锚点 | 48 | **现象可复现性**：exp2-S 只测过 ep50（null）。若其中段 checkpoint 也强于锚点，则「只看终点漏掉中途峰值」是普遍现象而非 exp4 独有 |

次要项一律按探索性报告（给 CI 但不宣称显著性），任何后续主张需再做确认实验。

## Method
标准竞技场：2v2 对角座位、复式对局牌（同一副牌 × 双方向）、RAW 终局点数配对差分。锚点 = `v2_engine_full_run_20260802_005918/checkpoints_sft_warmup_mahjong`（与 exp2/exp4 同一锚点）。引擎 = post-audit RCR。并发 24。

## Env / 成本
flex-start a2-highgpu-1g（STOP 终止 + 增量同步已加固），预计 480 局 ≈ 7-8h ≈ $16。

## Progress
- [2026-08-11] 预注册完成。

## Results
(pending)

## Conclusion
(pending)
