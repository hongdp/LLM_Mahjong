# exp63 顺位 pt 刻度重评：Mortal vs bc49 的"打平"是不是分数刻度的假象

- **Date**: 2026-09-04  **Status**: done
- **Git**: 见 Progress；本机 0 元（重算既有对局存档，不重打）
- **Env**: `experiments/elo_league/hanchan/matches/mortal298k_vs_bc49_*.json` 与 `mortal_shards/`（exp56 复式半庄存档，
  每场带四席 placements 与 uma_points）

## Purpose & Hypothesis
文献对照（docs/literature_review_2026-09.md §4-2）：Mortal 优化的是 GRP 顺位期望（pts [3,1,−1,−3] / jun_pt [90,45,0,−135]），
bc49 模仿人类分数分布。LEADERBOARD 上两者半庄胜份打平（0.4771±0.0144，uma pt −0.36±0.42），但早期批次 Mortal 首位率
29.8% vs 20.2%。假设：**换到天凤段位 pt 刻度（+90/+45/0/−135）后 Mortal 对 bc49 显著为正**，即 bc49 的"与 Mortal 同层"
是分数刻度的像，顺位刻度上仍有头顶空间；反假设：合并 n=1200 后首位率回落到 24.7%，顺位刻度也打平。

## Method
读取全部 Mortal vs bc49 半庄存档（同种子复式配对，Mortal 两席 vs bc49 两席），对每场按四席 placements 重算：
① 天凤 pt（鳳凰南、十段刻度）[+90,+45,0,−135]；② Mortal 纯顺位 pts [3,1,−1,−3]；③ 首位率 / 四位率；④ 现有 uma 刻度作对照。
统计量：Mortal 两席合计 − bc49 两席合计，逐场均值 ± SE（配对复式单位），以及逐席位率的二项 SE。
再对 exp46I、bc51 的同类存档做同样重算（对照锚），看刻度效应是否系统性。

## Success Criteria（预注册）
- 天凤 pt 刻度上 Mortal − bc49 ≥ +2.5σ → **刻度假象成立**：bc49 的目标函数（分数）与部署目标（顺位）错位，
  下一步把 GRP/顺位价值引入训练目标（exp 后继）；
- |z| < 2 → 顺位刻度也打平，"bc49 与 Mortal 同层"结论在两把尺子上都成立，转向数据 10×。

## Progress
- [09-04] 存档去重（按 (seed, a_seats)，shards 与 merged 文件重叠）后：Mortal vs bc49 **600 牌山 / 1200 场**（seed 49900000 + 51000000），
  vs exp46I 600/1200，vs bc51、exp46Cb 各 200/400。统计单位 = 牌山（两向合计），SE 用牌山间标准差。

## Results（Mortal 两席 − 锚两席，每席每半庄）

| 锚 | 牌山 | 天凤 pt [90,45,0,−135] | 纯顺位 [3,1,−1,−3] | uma（千点） | 首位率 M vs 锚 | 四位率 M vs 锚 |
|---|---|---|---|---|---|---|
| **bc49** | 600 | **+1.95 ± 5.65（z=+0.35）** | **+0.000 ± 0.147（z=0.00）** | −1.42 ± 1.67（z=−0.85） | 0.247 vs 0.253 | 0.245 vs 0.255 |
| exp46I | 600 | +15.97 ± 5.64（z=+2.83） | +0.323 ± 0.147（z=+2.20） | +1.99 ± 1.68（z=+1.18） | 0.254 vs 0.246 | 0.226 vs 0.274 |
| bc51_v3r2 | 200 | −5.85 ± 8.38（z=−0.70） | −0.120 ± 0.226 | −1.10 ± 2.59 | 0.247 vs 0.253 | 0.259 vs 0.241 |
| exp46Cb | 200 | +50.85 ± 9.61（z=+5.29） | +1.340 ± 0.257（z=+5.21） | +16.03 ± 2.97（z=+5.40） | 0.289 vs 0.211 | 0.193 vs 0.307 |

## Conclusion
**假设证伪：换到顺位刻度 Mortal 对 bc49 仍是零**（天凤 pt z=+0.35、纯顺位 z=0.00、首位率 24.7% vs 25.3%）。
早期"首位率 29.8% vs 20.2%"是 n=300 批的运气，合并 1200 场后消失。bc49 与 Mortal 在分数、uma、天凤 pt、纯顺位四把尺子上
全部打平——"同层"不是刻度假象。有意思的是刻度确实会改判决：Mortal 对 exp46I 在天凤 pt 上 +2.83σ（uma 上只 +1.18σ），
exp46I 的四位率 27.4% 比 bc49 高——RL 微调产物在"避四"上比 BC 差，这是顺位刻度才看得见的信息。
→ 文献路线 ②（把 GRP/顺位目标引入训练）失去"bc49 目标错位"这个动机，降级；数据 10×（exp65）与高分席位微调（exp64）升为主线。
评测口径建议：裁决级头对头同时报天凤 pt 刻度（存档已带 placements，零成本）。

## Next Steps
- LEADERBOARD 增加天凤 pt 列（由 placements 重算，不重打）。
- exp64 / exp65 继续。

## Artifacts
| Path | Size | Description |
|---|---|---|
| experiments/elo_league/hanchan/mortal_shards/*.json, matches/mortal298k_vs_*.json | — | 既有存档（每场 placements/uma） |
| 本文件 Results 表 | — | 重算结果（脚本为一次性 Python，逻辑见 Method） |
