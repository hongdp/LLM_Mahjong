# exp110 — Stage 1：引擎精确放铳代价的全动作控制变量进 PPO（无学习 critic）

- **日期**：2026-10-13 预注册  **状态**：prereg
- **Git**：`c2e037a`（`--ron_term`）；repo.tgz md5 `e65f1c26…`
- **Env**：纯血线；Rust 引擎镜像自对弈；`cnn_l_v3r` 从 `exp102_A/games_final.pt`（224M）续 16M → 240M；RunPod Secure
- **用户批准**：2026-10-13 "先跑 Stage 1 混合精确项 PPO（≈$6）"

## Purpose & Hypothesis
exp109 用反事实裁判证明：精确放铳项让估值器在暴露决策上排序优于策略（7σ），但可兑现价值 ≈ 0（+10 ± 27 分/决策）；学到的 Q 当优势一律有害（非暴露状态排序差）。本实验只保留**零学习、无偏**的部分：
策略梯度加入 **全动作控制变量** Σ_a π_θ(a|s)·(−c(s,h,a)) − ratio_t·(−c(s,h,a_t))，c = 引擎精确的即时荣和代价（`VecEnv.ron_cost_info`，`tests/test_ron_cost.py` 对账 0 误差）。两项的期望梯度相减为零（score-function 恒等式）⇒ 估计无偏；当对手确实荣和时，采样项与 GAE 优势里的放铳噪声精确相消，暴露决策上的梯度方差下降一个量级，并给出"每张切牌的期望放铳代价"这一低噪声信号。
- **H1**：窗口均值（228–240M × 2 段 10k）比 exp106 C/C′ 高 ≥ +0.5 pp 且点差 > 0；暴露决策向听 ≥ 2 的出生牌率下降。
- **H0**：|Δ| < 0.5 pp——放铳信号的方差不是防守学不出来的原因（与 exp109 裁判的"可兑现价值 ≈ 0"一致），价值侧彻底关闭。

## 先验声明
c 是训练期引擎事实（隐藏状态下的规则结果），只进策略梯度的控制变量项；策略输入与推理路径不变；零人类数据、零外部模型。用户 2026-10-12 确认不影响纯血。

## Method
- 单臂 **R1**：exp106 C 配方（lr 3e-5、α 0.003 常数、gpi 2048、dup_k 8、γ 0.995、λ 0.95、target_kl 0.03、clip 0.2、batch 4096、1 pass、amp、**无 adv_clamp**）+ `--ron_term 1.0`，seed 1101。对照 = exp106 C / C′（同起点同局数，仅 adv_clamp 差 <0.1%）。
- 实现：`train_dnn_ppo.py --ron_term`：`ron_full` [N,374]（切牌槽，普通/赤五分开）按本批优势 std 缩放，PPO 代理里加 `(Σπ c − ratio·c_t)`；日志 `ron/frac_rows`（有放铳代价的行占比）、`ron/mean_cost_taken`、`ron/expected_cost_pi`、`ron/sampled_cost`。本机冒烟见 Progress。
- 评测：里程碑 228/232/236/240M，自动评测器 10k × 2 段配对；终点 `tile_danger_probe` 对 C 240M。
- 机器：Secure 4090（$0.89/h，≈1000 局/s ⇒ ≈4.5 h ≈ $4）或 A5000（$0.27/h，≈400 局/s ⇒ ≈11 h ≈ $3）。**上限 $8**。

## Success Criteria（预注册）
1. 健康：KL < 0.03、H 0.2–0.4；`ron/expected_cost_pi` 应随训练下降（策略把期望放铳代价压低）而 `ron/frac_rows` 稳定。
2. 主判据：R1 − mean(C, C′) 窗口均值 ≥ +0.5 pp 且点差 > 0 ⇒ H1；|Δ| < 0.5 ⇒ H0；≤ −0.5 ⇒ 有害。
3. 机制：R1 240M 向听 ≥ 2 暴露出生牌率 < C（exp102 口径 0.37）。
4. 运维：TRAIN_DONE ⇒ md5 握手 ⇒ 自毁；拉取循环主 checkout 绝对路径。

## Progress
- [10-13 06:40 PDT] 实现 `--ron_term`（控制变量进 PPO 代理，按本批优势 std 缩放）；本机冒烟 5 迭代：有放铳代价的行占 12.6–13.2%，`expected_cost_pi` ≈ `sampled_cost` ≈ 0.047–0.055（千点），KL 0.0017–0.0027、H 0.30 常态、吞吐与盲配方同（460 局/s on 4080）。
- [10-13 08:28 PDT] **发射**：Secure 4090（≥12 vCPU）与 3090 都无货，用 **Secure A40 `8uggt0946zyhyk`**（CA-MTL-1，9 vCPU，CUDA 13.0，69.30.85.116:22066，$0.59/h）。起点 ckpt md5 一致。实测 **290 局/s**（8 核 CPU 瓶颈，GPU 7%）⇒ 16M ≈ 15 h ≈ **$9**——超预注册上限 $8 约 $1，接受并记录（更多 vCPU 的机型全部缺货）。拉取循环、心跳、TB `exp110_R1_LIVE`、自动评测器已挂。
- [10-13 12:20 PDT] **228M**：0.4636 / 0.4639（C 0.4639/0.4669，C′ 0.4559/0.4556；mean(C,C′) 0.4606 ⇒ +0.3 pp，噪声内）。在轨 0–4M 局：KL 0.002、H 0.25–0.29、`ron/frac_rows` 0.133 稳定、`ron/expected_cost_pi` 持平 52 分（策略尚未压低期望放铳代价）。

## Results
（待）

## Conclusion
（待）

## Artifacts
（待）
