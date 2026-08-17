# exp16：优势裁剪界消融——大番数胡牌的策略梯度被 ±5σ cap 是否有害

- **Date**: 预注册 2026-08-16 ~23:40 本地  **Status**: pre-registered → launching
- **Git**: c51c37e（--adv_clamp 参数化 commit）
- **Env**: GCP g2-standard-32 on-demand（c5, us-east1-c），L4
- **对照**: exp15 = `dnn_exp15_vitscale_20260816`（同起点同配方，唯一差异 --adv_clamp 5→10）

## Purpose & Hypothesis
用户旧发现：归一化优势 `.clamp(-5,5)`（train_dnn_ppo.py）把役满级和牌（~8σ）的策略梯度
截去 ~40%，系统性偏向小而快的和牌。原定修复（exp11-A2 乘法价值通道）已被证伪（−2029），
问题无人认领 → 本实验以纯超参方式单变量检验：放宽到 ±10σ（覆盖役满，仍保离群护栏）。
假说：大牌路线的学习信号解禁 → 打点分布右移，强度不降（若大牌路线真有价值则升）。

## Method
与 exp15 完全同配方同起点（vit240 resume，champion 配方 ppo_epochs=1 + 熵 0.01），
唯一差异 `--adv_clamp 10`。跑到 700k（与 exp15 的 400k/700k 里程碑同局数可比）。
ladder watch 全程；两臂同为纯自对弈（主线哲学内）。

## Config
exp15 全套 + `--adv_clamp 10 --total_games 700000 --milestones 400000,700000
--exp_dir experiments/dnn_exp16_advclamp_20260816`

## Success Criteria（发射前定死）
1. **主判定**：700k 终点 vs exp15@700k，200 副复式配对；显著正 ⇒ cap 有害确认；
   显著负 ⇒ ±5 是有益的方差控制；null ⇒ cap 在当前规模无关紧要（三种都有结论价值）。
2. **机制指标（方向性）**：700k 对打的 arena 原始记录中，本臂 ≥8000 点（满贯+）和牌
   频率 ≥ exp15 同场（打点分布右移的直接观测）。
3. 训练健康：KL<0.03、熵无坍缩（≥0.4）、无 NaN；ladder 曲线无异常深谷。

## Progress
- [2026-08-16 23:40] 预注册。c5 启动+播种+发射流程同 exp15（含 ssh 重试教训）。

## Results
| Metric | This run | Baseline (exp15) | Success criterion |
|---|---|---|---|
| （待运行） | | | |

## Conclusion
（待运行）

## Artifacts
| Path | Size | Description |
|---|---|---|
| gs://llm-mahjong-experiments/dnn_exp16_advclamp_20260816/ | — | 云端主目录 |
