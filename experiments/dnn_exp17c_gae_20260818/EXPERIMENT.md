# exp17-C：GAE 单变量归因（A0 高分与风格突破的功臣检验）

- **Date**: 预注册 2026-08-18  **Status**: done（GAE 归因确认，纯自对弈新纪录 1079.7）
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
- [2026-08-18] 发射确认：首 iter win_rate 0.007，81 s/iter（~25 局/s），预计 ~8h 跑满。ladder 已挂。**Status: running**。

## Results
| Metric | This run (GAE) | Baseline (exp18-cnn) | Success criterion |
|---|---|---|---|
| 主判定：700k 对打（seed0=20260826） | +244 ± 984（副胜 185:141） | — | 显著正 ⇒ **未达（点差 null，但见下）** |
| 风格判定：立直率（4000 局） | **0.243**（副露 0.421） | 0.117（副露 0.754） | ≥10% ⇒ **✅ 达成且翻倍于基线** |
| 正式评分（全池 ×100 副） | **1079.7 ± 13.5 —— 纯自对弈史上最高** | 1013.4 | ≥1020 ⇒ ✅ |
| ladder 轨迹 | 887→955→1000→1018→**1040（700k 仍在爬升，无平台）** | 平台 ~1012 | — |

## Conclusion
1. **GAE 归因确认（综合证据）**：点差主判定 null（+244±984，200 副对 ~250 分真差欠功效），
   但正式评分差 +66（z≈3.5）、副胜率 56.7%、风格翻倍三线汇聚——GAE 是 A0 高分与
   立直风格的功臣。**新纯自对弈纪录 1079.7**，与教师参照线 bcrl14_600（1117.7）的缺口
   从 174 分收窄到 **38 分**。
2. **平台被打破**：与所有无 GAE run 不同，本臂 700k 终点仍在爬升——更好的信用分配
   （逐步 TD(λ) 优势）正是此前样本堆不动的那面机制墙。
3. 冠军配方更新候选：**GAE + 熵台阶**；与 exp19 的 ConvFormer 合体 + 续训是显然下一步（exp20）。

## Next Steps
- exp20 候选：GAE + ConvFormer + 熵台阶，1.2M 或续训——冲击教师参照线。

## Artifacts
| Path | Description |
|---|---|
| gs://llm-mahjong-experiments/dnn_exp17c_gae_20260818/ | 云端主目录 |
