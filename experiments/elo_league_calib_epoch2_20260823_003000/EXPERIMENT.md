# Elo 锚点池纪元 2 重校（精确立直暗杠规则）

- **Date**: 预注册 2026-08-23 00:30 PDT；链式启动于 exp22-r2 收尾链（EXP22R2_CLOSE_DONE）之后  **Status**: done
- **Git**: 分支 `engine/riichi-ankan-exact` 2929629（启动前 ff-merge 进 master）；引擎指纹 纪元1 `d24241ea4b1f2577` → 纪元2 `f0bb01e3e75b31ed`
- **Env**: 本地 RTX 4080 + 24 vCPU，conda rlhf_mahjong

## Purpose & Hypothesis
引擎变更（`_can_ankan` RCR 3.12(2) 从"相邻牌即拒绝"近似改为枚举听牌拆分的精确判定）触发
docs/design_elo_league.md 纪元规则：历史对局作废，整池重校。
假设：该规则只影响"立直中摸到第四张"的极窄情形（600 局冠军自对弈实测：13 次，旧放行 7 → 新放行 13，
≈ 每 100 局多 1 个合法动作，0.014% 决策），因此 8 锚点的纪元 2 评分应与纪元 1
（`experiments/elo_league_calib_20260816_144910` + bcrl14_600 晋升记录）在各自 SE（≈10 Elo）内一致，
排序不变。若出现 >2σ 偏移，说明规则改动的影响不止于动作空间（需排查）。

## Method
`scripts/run_elo_league.py calibrate`：8 锚点全循环 28 对 × 200 副复式，sign-MLE，bc_cnn 钉 1000。
纪元 1 的 `experiments/elo_league/` 整目录改名保留为 `experiments/elo_league_epoch1/`（含 matches、history.jsonl），
新 `experiments/elo_league/anchors.json` 带 `engine.fingerprint = f0bb01e3…`。
启动顺序（链式脚本）：等 exp22-r2 收尾链在纪元 1 引擎下完成 → `git merge --ff-only engine/riichi-ankan-exact` →
跑全套 tests → 改名旧池 → calibrate。

## Config
`--deals 200 --seed0 20260823 --parallel 20`；ANCHOR_POOL 8 员（含 bcrl14_600，纪元 1 校准时为 7 员 + 后补晋升）。

## Success Criteria
1. 全部 8 锚点纪元 2 评分与纪元 1 评分之差 |Δ| < 2·sqrt(SE1²+SE2²)；排序不变。
2. 全部 SE < 40 Elo；残差无 |残差| > 2σ 系统性克制对。
3. 新 anchors.json 带纪元 2 指纹，`rate` 在 master 工作树下不再触发 ENGINE EPOCH MISMATCH。

## Progress
- [2026-08-23 00:30] 预注册；链式脚本挂在 exp22-r2 收尾完成之后（预计 03:30–04:30 PDT 启动，约 70 min，28 场 × ~2.5 min）。

## Progress（续）
- [02:44] exp22-r2 收尾完成（纪元 1 引擎，但竞技场分差已是 A−B 量纲）；[02:46] ff-merge 1753c8e，172 tests 通过，
  旧池改名 `elo_league_epoch1/`，开始重校；[03:07] 28/28 完成（每对 ~40 s，比纪元 1 的 2.5 min/对快 3.7×——引擎优化 + 本机空闲）。
- 注意：本次"纪元 2"同时包含两项变更：精确立直暗杠 + **竞技场分差 A−50000 → A−B**（sign-MLE 的胜负判定随之改变）。

## Results（纪元 2 引擎指纹 d22a7f374b33a4ad，seed0 20260823，200 副/对）
| 锚点 | 纪元 1 | 纪元 2 | Δ | z | 判定 |
|---|---|---|---|---|---|
| bcrl14_600 | 1117.7 | **1066.0 ± 10.2** | −51.7 | −2.8 | ✗ 超 2σ |
| e700 | 1012.5 | 1024.2 ± 9.8 | +11.7 | +0.8 | ✓ |
| bc_cnn（钉） | 1000.0 | 1000.0 | 0 | — | — |
| reuse11_600 | 966.9 | 936.8 ± 9.5 | −30.1 | −2.1 | ✗ 略超 2σ |
| vit240 | 928.8 | 914.7 ± 9.6 | −14.1 | −1.0 | ✓ |
| rf600 | 894.9 | 892.8 ± 9.6 | −2.1 | −0.1 | ✓ |
| ppo44_240 | 830.8 | 843.7 ± 9.9 | +12.9 | +0.9 | ✓ |
| ppo44_600 | 828.3 | 842.4 ± 9.9 | +14.1 | +1.0 | ✓ |
排序：完全不变。SE 全部 < 11。

## Conclusion
- 排序不变、6/8 锚点在 2σ 内，**但 bcrl14_600（−52，z −2.8）和 reuse11_600（−30，z −2.1）超出**。两者都不是纪元 1
  全循环赛的原始成员：bcrl14 是 O(3) 晋升（3 场邻近 + 1 场远端，旧量纲）后入池，其 1117.7 本就带晋升协议的传递误差；
  reuse11 在纪元 1 残差表里就是最大偏离者（−0.042）。所以偏移更可能来自**旧量纲 + 稀疏晋升赛**的测量误差，
  而非暗杠规则（实测只动 0.014% 决策）——但本次无法把两者分离（两项变更同纪元）。
- 纪元 1 的 bcrl14 1117.7"教师参照线"被下修到 1066：纯血冠军与教师系的差距比原以为的小。
- 结论性判定：成功标准 1 部分未达（2 个锚点超 2σ），2、3 达成。纪元 2 池随即被纪元 3（雀魂规则 + 赤宝牌）取代，
  这套分数只作为纪元 1→3 的桥。

## Next Steps
- 冠军归属（exp17-C vs exp22r2）改在纪元 3 池上同池再评（`elo_league_calib_epoch3_20260823_023000` 之后）。
- 以后晋升赛（O(3)）在新量纲下重新校验一次 bcrl14 的晋升记录，避免再次出现 −50 级的修正。

## Artifacts
| Path | Description |
|---|---|
| experiments/elo_league/anchors.json | 纪元 2 锚点表 |
| experiments/elo_league_epoch1/ | 纪元 1 全量存档（anchors、history、matches） |
