# Elo 锚点池一次性联合校准

- **Date**: 2026-08-16 14:49 本地  **Status**: done
- **Git**: 5c66d11 + 未提交（scripts/run_elo_league.py 新增、docs/design_elo_league.md 新增）
- **Env**: 本地 RTX 4080 + 24 vCPU，conda rlhf_mahjong

## Purpose & Hypothesis
建立跨实验公共强度标尺（设计 docs/design_elo_league.md，源自用户提议）。
锚点池 7 员（bc_cnn 钉死 1000 定标；rf600 / ppo44_240 / ppo44_600 / reuse11_600 / e700 / vit240），
全循环赛 21 对 × 200 副复式 → 联合 MLE。假设：拟合分序与既有两两竞技场结论一致
（e700 > reuse11_600 > ppo44_240 > ppo44_600 之类的已证顺序），且残差无系统性 2σ 越界。

## Method
`scripts/run_elo_league.py calibrate`：sign(配对分差) 作每副胜负（平=0.5），
Bernoulli 似然 + 阻尼对角牛顿联合 MLE；SE 取 Fisher 信息。原始对局 JSON 全落
`experiments/elo_league/matches/`。数学冒烟：合成 1000/1200/1400 三方恢复 1207/1374（400 副/对）。

## Config
`--deals 200 --seed0 20260816 --parallel 20`；锚点路径表见脚本 ANCHOR_POOL。

## Success Criteria
1. 拟合序与既有显著竞技场结论零冲突（冲突即校准失败，排查非传递性/管线 bug）。
2. 全部锚点 SE < 40 Elo。
3. 残差表无 |残差| > 2σ 的系统性克制对（有则如实记录为非传递性证据，不隐瞒）。

## Progress
- [2026-08-16 14:49] 发射（本地后台）。预计 ~30-60 min。
- [2026-08-16 15:25] 21/21 场完成（~2.5 min/场），联合拟合收敛；补做分差可加标度拟合。

## Results

| 锚点 | Elo (sign) | 分差标度 (bc_cnn=0) |
|---|---|---|
| e700 | 1012.5 ± 10.6 | **+1072 ± 250** |
| bc_cnn（钉定 1000/0） | 1000.0 | 0 |
| reuse11_600 | 966.9 ± 10.3 | −397 ± 250 |
| vit240 | 928.8 ± 10.3 | −533 ± 253 |
| rf600 | 894.9 ± 10.3 | −2136 ± 226 |
| ppo44_240 | 830.8 ± 10.7 | −2639 ± 237 |
| ppo44_600 | 828.3 ± 10.7 | −2496 ± 226 |

- 判据 1 ✅：全部既有显著两两结论复现（e700>reuse11>rf600>ppo44_600；vit240>ppo44_240；
  ppo44_240≳ppo44_600 点估计在分差标度上反号但 null，其余全部同号）。
- 判据 2 ✅：Elo SE 全部 ~10.5 < 40。
- 判据 3 ✅（带记录）：最大 sign 残差 0.078≈2.2σ（21 格出 1 个属预期）；分差可加性残差
  >800 有 3 对（bc_cnn/vit240 +804、ppo44_240/e700 +895、ppo44_240/vit240 −864）——
  轻度非传递性存在，量级 <1000 分，不推翻整体标尺。
- 每副配对分差典型 SD ≈ 6434 → sign 标度严重压缩（全谱仅 184 Elo）：巨强差距也只赢
  ~60% 副数。**仪表盘采用双标度：Elo 管排序、分差标度管幅度**（后者与历史竞技场 CI 直接可比）。

## Conclusion
1. 校准成功，锚点表冻结生效（anchors.json + points_scale.json）。
2. **重大副产品发现：bc_cnn（教师模仿 BC，从未与自对弈谱系交过手）位居全池第 2**，
   显著强于 reuse11-600k/vit-240k/rf-600k（后三者 ≤ −397 起），唯一显著在其上的是冠军
   E-700k（+1072±250, z≈4.3）。解读：纯自对弈从零烧 70 万局 + 熵退火，才把强度推到
   「简单模仿平庸教师」之上 ~1000 分——教师先验的价值被长期低估，值得设计
   BC-warm-start + 自对弈 RL 的组合实验（潜在 exp14）。
3. 非传递性以 ~800-900 分量级存在于个别风格对之间，Elo 单值可用但残差监控必须保留。

## Artifacts
| Path | Size | Description |
|---|---|---|
| experiments/elo_league/anchors.json | — | 锚点分数表（冻结资产） |
| experiments/elo_league/matches/*.json | — | 21 场原始对局 |
