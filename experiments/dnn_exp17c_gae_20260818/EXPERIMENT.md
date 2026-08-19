# exp17-C：GAE 单变量归因（A0 高分与风格突破的功臣检验）

- **Date**: 预注册 2026-08-18  **Status**: launching
- **Git**: 951a6ea  **Env**: mahjong-dnn-c3（us-east1-b）on-demand g2
- **对照**: exp18-cnn = `dnn_exp18_cnn_20260817`（**同协议已完成的共享基线**，唯一差异=本臂加 GAE）；
  参照点 A0=1022.4（GAE+cnn+600k 从零，但含恒定熵 0.03 与 exp11 语境）

## Purpose & Hypothesis
exp11 副产品：A0（普通 GAE critic）从零 600k 得 Elo 1022.4、自发立直 14.9%——两项都超
e700 谱系（1012.5、立直 ~3%）。假说：**GAE（λ=0.95）是功臣**。本臂在 exp18-cnn 协议上
单变量加 GAE：若复现高分+立直风格，冠军配方升级为 GAE+熵台阶；若不复现，A0 的高分
归因存疑（可能是 target_kl/n_effective 等 exp11 语境差异）。

## Method / Config
exp18 共享协议原样（cnn_m 从零 700k、ppo_epochs=1、熵台阶 0:0.03,600000:0.01、seed 42、
adv_clamp 5）+ **唯一新增 `--gae_lambda 0.95`**。exp_dir `dnn_exp17c_gae_20260818`。

## Success Criteria（发射前定死）
1. **主判定**：700k vs exp18-cnn-700k，200 副复式；显著正 ⇒ GAE 归因确认。
2. **风格判定**：eval_style_profile 4000 局，立直率 ≥10%（exp18-cnn 侧同测对照）。
3. 正式评分预期 ≥1020（A0 量级）；训练健康同标准。

## Progress
- [2026-08-18] 预注册，c3 发射。

## Results
| Metric | This run | Baseline (exp18-cnn) | Success criterion |
|---|---|---|---|
| （待运行） | | | |

## Conclusion
（待运行）

## Artifacts
| Path | Description |
|---|---|
| gs://llm-mahjong-experiments/dnn_exp17c_gae_20260818/ | 云端主目录 |
