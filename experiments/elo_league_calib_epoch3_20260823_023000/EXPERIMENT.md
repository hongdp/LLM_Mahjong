# Elo 锚点池纪元 3 重校（雀魂单局规则 + 赤宝牌）

- **Date**: 预注册 2026-08-23 02:30 PDT；链式启动于纪元 2 重校（`elo_league_calib_epoch2_20260823_003000`）完成之后  **Status**: done
- **Git**: 分支 `engine/majsoul-rules` 5ba7289（aeeafa5 规则 + 5ba7289 赤宝牌），启动前 ff-merge 进 master
- **Env**: 本地 RTX 4080 + 24 vCPU，conda rlhf_mahjong

## Purpose & Hypothesis
用户指令（02:00）：单局内规则按雀魂改（docs/design_majsoul_rules.md）。引擎变更 ⇒ 纪元规则 ⇒ 整池重校。
这次变更不再是"微小"：赤宝牌进入牌山（每局 3 张），所有历史 checkpoint 都没见过红五（编码器 v1/v3 折叠、
动作头按"有普通五先打普通五、只剩红五才打"加宽），且新增途中流局/流局满贯/双倍役满/去人和。
假设：(1) 规则对所有锚点一视同仁，**排序基本保持**（Spearman ≥ 0.9）；(2) 绝对分差会比纪元 2 变化更大
（红五带来的打点方差 ↑，sign-MLE 的信息量 ↓ → SE 略增）；(3) 没有锚点因动作头加宽而出现非法/退化行为
（竞技场 40 副冒烟已通过；重校中监控 `_EMPTY_LEGAL` 与异常流局率）。

## Method
同纪元 2：`run_elo_league.py calibrate --deals 200 --seed0 20260823 --parallel 20`，8 锚点 28 对。
纪元 2 目录整体改名 `experiments/elo_league_epoch2/`。新 anchors.json 指纹 = 合并后的引擎。
链式顺序：等 `EPOCH2_CALIB_DONE` → `git merge --ff-only engine/majsoul-rules` → 全套 tests → 改名 → calibrate。

## Config
同上；ANCHOR_POOL 8 员不变（全部为纪元 1 训练、折叠红五的模型）。

## Success Criteria
1. 纪元 3 vs 纪元 2 排序 Spearman ≥ 0.9；bc_cnn 仍钉 1000。
2. 全部 SE < 40；无 |残差| > 2σ 系统性克制对。
3. `rate` 在 master 下不触发 ENGINE EPOCH MISMATCH。
4. 新纪元的所有后续训练 run 用 `*_r` / `*_v3r` 架构（374 动作头 + 红五平面）。

## Progress
- [02:30] 预注册；链式脚本挂在纪元 2 重校之后（预计 05:00–06:30 PDT 启动，约 70 min）。

## Progress（续）
- [03:08] 纪元 2 完成后 ff-merge 9a0aa36（含雀魂规则 + 赤宝牌 + HandSet/exp27/exp28 预注册），172 tests 通过；
  纪元 2 池改名 `elo_league_epoch2/`；[03:29] 28/28 完成（~45 s/对）。

## Results（纪元 3 引擎指纹 127462426506c3b4）
| 锚点 | 纪元 2 | 纪元 3 | Δ | z |
|---|---|---|---|---|
| bcrl14_600 | 1066.0 | **1100.8 ± 10.6** | +34.8 | +2.4 |
| e700 | 1024.2 | 1015.0 ± 9.8 | −9.2 | −0.7 |
| bc_cnn | 1000.0 | 1000.0 | 0 | — |
| reuse11_600 | 936.8 | 948.2 ± 9.6 | +11.4 | +0.8 |
| vit240 | 914.7 | 928.4 ± 9.6 | +13.7 | +1.0 |
| rf600 | 892.8 | 887.5 ± 9.7 | −5.3 | −0.4 |
| ppo44_240 | 843.7 | 839.3 ± 10.0 | −4.4 | −0.3 |
| ppo44_600 | 842.4 | 828.1 ± 10.1 | −14.3 | −1.0 |
Spearman = 1.0；SE 全部 < 11。

## Conclusion
- 成功标准 1–3 全部达成：排序不变，赤宝牌 + 雀魂规则没有改变老模型之间的相对强弱（它们都折叠红五，条件对称）。
- bcrl14 在三个纪元间 1117.7 → 1066.0 → 1100.8：其真实分应在 ~1090 ± 25，纪元 1 的晋升记录偏高、纪元 2 偏低；
  以后引用教师参照线用纪元 3 的 1100.8 ± 10.6。
- 纪元 3 = 当前生效池。后续所有评分在此池上；纪元 1/2 的数字只用于相对排序叙述。

## Next Steps
- 在纪元 3 池上再评 exp17-C final 与 exp22r2 final（冠军归属）；再评 exp20 1.2M 作参考。
- 第一批纪元 3 训练（exp27 A/B/C + exp28 A′/A″）发射。

## Artifacts
| Path | Description |
|---|---|
| experiments/elo_league/anchors.json | 纪元 3 锚点表 |
| experiments/elo_league_epoch2/ | 纪元 2 存档 |
