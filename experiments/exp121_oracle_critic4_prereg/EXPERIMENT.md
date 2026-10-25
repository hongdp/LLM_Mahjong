# exp121 — Oracle 四座位评论家：给一步反事实一个有分辨力的 bootstrap（预注册，未发射）

- **Date**: 2026-10-25 15:40（预注册）  **Status**: planned（**未发射**；Stage 0 本地门槛先跑，Stage 1 等用户批预算）
- **Git**: 本 worktree（exp120 收口于 3528675）；模型/引擎代码在 runpod worktree 1736dfe（`cnn_l_v3r`、`oracle_features.py`、`cnn_m_ro`、Rust 引擎）
- **Env**: Stage 0 本地 CPU（12 worker）；Stage 1 RunPod **Community** 3090 Ti（纯血线权重，无冠军谱系资产，按 2026-09-01 安全分层允许）
- **提议人**: 用户（2026-10-25 15:05）："做一个公平评判四个座位胜率的 oracle value model，用它做反事实 rollout"

## Purpose & Hypothesis
**上游结论链**
- exp120（10-25）：exp113 C 的部分信息价值头一步前瞻（V1 / Vcf）在平手牌效、安牌平手两个分层上都**不如策略本身**
  （Vcf 0.476 vs π 0.582；现物置顶 0.208 vs 0.737），候选间 V 差 0.01–0.05 ≈ 评论家噪声 → "搜索出目标再蒸馏"缺的是评论家分辨力。
- exp72（09-07）：同牌山反事实 rollout 到局末当 PPO 优势，1M 局中性；但量到**执行动作只在 60–68% 的决策上是同牌山最优**——策略与反事实目标有 1/3 不一致。
- exp67（09-05）：**oracle 评论家（看对手暗手/待张/牌山构成/隐宝）在 bc65 起步的 PPO 里零增益**，且量化了原因：
  即使知道全部隐藏牌，本局回报的可解释方差只从 5.3% 升到 6.0%（最后 5 步 8.4%→9.6%）——**≥90% 的本局结局由尚未发生的摸牌顺序决定**。
- exp68：oracle planes 编码（`_oracle_planes`，8×34：3 家暗手、3 家待张、牌山构成、里宝）与 `cnn_m_ro` 架构已存在；评测路径永远为零，不可能作弊。

**exp67 为什么不直接否掉 exp121**：exp67 问的是"评论家能否解释本局回报"（R²），exp121 问的是"评论家能否**在同一状态的兄弟候选之间排序**"。
摸牌顺序的 90% 方差对同一决策点的所有候选是**共同的**（同一牌山），在 V(s_a) − V(s_b) 里相消；候选间真正的差来自两个 oracle 看得见的量——
放铳（对手待张，oracle 精确知道）和进张（牌山构成，oracle 精确知道）。exp67 的 R² 测量没有回答兄弟差能否被回归出来；exp120 说明部分信息评论家回答不了。

**假设 H1**：4 座位 oracle 评论家对兄弟候选的排序，比部分信息评论家更接近"同牌山精确反事实"的排序，并在 exp120 的两个分层上超过策略
（判据见下）。**反假设 H0**：评论家回归误差本身 ≥ 兄弟差（Δ 几十点 vs 回归残差几百点），oracle 信息改善不了排序 → exp67 的结论在排序层面同样成立，
"搜索作改进算子"方向关闭，纯血线剩余杠杆只剩规模×时间或接受平台。

**纯度**：输入全部是引擎状态（自对弈特权），零人类知识；评论家只在训练/探针侧使用，策略与评测路径不变。与 exp68 同一纯度声明。

## Method

### Stage 0 — 门槛（本地 CPU，$0，发射 Stage 1 的前提）
**精确同牌山反事实到局末**（T=0 下给定真实牌山完全确定，无需任何评论家）= 排序的金标准，也是任何评论家驱动搜索的**收益上限**。
- 工具：`scripts/probe_value_1ply.py` 加 `--cf-to-end`（复用 exp72 `start_branch` 的 `play_game_gen(table=clone)` 模式，rollout 到局末读 `final_rewards[pid]`），
  对 exp120 B 组同一批决策（种子 70000000+，p=0.25；记录 (seed, pid, step) 以便 Stage 1 重放同一批状态）。
- 指标（按 exp120 分层）：
  - `Vend_gain` = mean[ max_alt Vend − Vend(π 所选) ]（后见之明上界，单位千点+顺位奖励）；
  - `π_end_opt` = π 所选是 Vend 最优的比例；
  - 安牌平手分层：Vend(现物) − Vend(π 所选) 的配对均值±SE。
- **门槛**：平手牌效或安牌平手分层上 `Vend_gain ≥ 0.05`（≥50 点/决策；此类决策每半庄约 10 次 → ≈500 点/半庄 ≈ 0.4 顺位点）**且** `π_end_opt ≤ 0.85`。
  不过门槛 → 即使有完美评论家，可挖的空间也不到 50 点/决策 → **不发射 Stage 1，整条搜索方向关闭**（记 FINDINGS）。
- 成本：exp120 B 组 3241 决策 × ≤14 候选 × ≈60 次策略前向/rollout ≈ 2.7M 次 CPU 前向，12 worker ≈ 10–15 min。

### Stage 1 — 数据生成 + 4 座位 oracle 评论家回归（RunPod）
**数据生成**
- 策略：exp113 C 384M，四份同桌，**T=0**（评论家要评的是部署形态下的状态分布；多样性来自牌山与随机场况），`randomize_round`。
- 引擎：**Python 引擎**（oracle planes/`oracle_features` 只有 Python 实现；Rust 侧无全信息导出，加导出需过 parity 全套测试——不值）。
  3090 Ti community 12 worker + `--gpu_infer` ≈ 200 局/s（exp57/58 口径）。
- 样本：每个**轮到切牌**的决策状态（四席都记）+ 他家响应窗口状态（评论家在 Vcf 里会被问到的两类状态都要在分布内）。
  **按 1/4 抽样**（同局状态强相关，全存浪费），每局 ≈ 70×4 决策 → 抽 ≈ 70 样本/局。
- 规模：**400k 局 → ≈28M 样本**。存储：v3r 公开平面 56×34 u8（k/20 网格，SKILLS 2026-09 修法）+ oracle 8×34 u8 + `oracle_features` 319 f16 + 标签 4 f32
  ≈ 2.9 KB/样本 → **≈80 GB**，只能留在 pod 卷上训练，**不回传**（工作站剩 43 GB）；回传只有 ckpt（<50 MB）与指标。
- 标签：该局终局 `final_rewards`（4 座位，REWARD_SCALE 0.001 × 点差 + RANK_BONUS，**四座位和为 0**），按绝对座位→以样本席位为 0 的相对顺序存。
- 时间：400k 局 / 200 局/s ≈ **35 min**。

**评论家**
- 输入：样本席视角的 v3r 公开平面（与策略同编码）+ `_oracle_planes` 8 平面（含**牌山构成**，不含牌山顺序——理由：含顺序 = 学模拟器 = 后见之明目标，
  与 exp72 的单牌山方差同病；不含顺序 = 对摸牌顺序取期望，正是公开信息策略要对齐的量）+ `oracle_features` 319 维投影进 value 路径。
- 架构：`cnn_m_ro` 的 trunk（64ch×3，in_planes = 56+8）去掉 policy 头，value 头输出 **4 维**（相对席位 0..3），**减均值**强制零和；
  参数 ≈1M；也挂一个 `cnn_m_v3r` 盲对照头（同数据、无 oracle 平面）——**同批控制臂**，exp120 的 public critic 是 PPO 价值头不是干净的回归，不能直接当对照。
- 训练：Huber 回归，batch 2048，AdamW 3e-4，cosine，**2 epoch ≈ 27k 步**，按局切分 95/5 留出。3090 Ti ≈ 20–30 min。
- 评论家侧诊断（不作判据）：4 座位留出 R²（预期 oracle 6–10%、盲 4–6%，复现 exp67 量级即正常）；放铳标定：对"下一步被荣和"的状态，oracle 价值是否显著更低。

**用评论家重跑 exp120 B 组（工作站，$0）**
- 重放 Stage 0 的同一批 (seed, pid, step) 决策（T=0 确定性）；Vcf 的 bootstrap 换成 oracle 评论家（**Vcf_o**），盲回归头同法得 **Vcf_b**；
  π、V1、原 Vcf（PPO 价值头）保留。
- 排序对照金标准 = Stage 0 的 `Vend`。

## Config（预注册）
| 项 | 值 |
|---|---|
| 策略 ckpt | `experiments/exp113_C/games_384000000.pt`（cnn_l_v3r） |
| Stage 0 决策集 | exp120 B 组种子 70000000+299，p=0.25，≈3,241 决策（重放） |
| 数据生成 | 400k 局，T=0，Python 引擎，12 worker + gpu_infer，1/4 抽样，≈28M 样本，≈80 GB pod 卷 |
| 评论家 | `cnn_m_ro` trunk + 4 维零和 value 头（oracle 臂）；`cnn_m_v3r` trunk 同头（盲臂） |
| 训练 | Huber，batch 2048，AdamW 3e-4 cosine，2 epoch，按局留出 5% |
| 输出 | `experiments/probes/exp121_stage0_cfend.json`、`exp121_critic_{oracle,blind}.pt`、`exp121_rerun_exp120B.json` |

## Success Criteria（发射前定）
**Stage 0 门槛**（上文）：`Vend_gain ≥ 0.05` 且 `π_end_opt ≤ 0.85` 于至少一个平手分层；否则不发射。

**Stage 1 判据**（n：平手牌效 ≈340、立直安牌平手 ≈259，与 exp120 同）
1. **排序分辩力（主判据）**：平手牌效分层上，Vcf_o 的 top-1 与金标准 `Vend` 最优一致率 ≥ **Vcf_b + 15 pp**（oracle 信息有用）
   **且** ≥ **π + 10 pp**（评论家搜索比策略更接近金标准）。
2. **exp120 判据复用**：Vcf_o 的"向听+受入最优"命中率 ≥ π + 15 pp，**或** 立直安牌平手现物置顶 ≥ π + 20 pp（exp120 原判据，同口径可比）。
3. **配对差**：安牌平手分层 Vcf_o(现物) − Vcf_o(π 所选) 与 Vend 同号且 |z| ≥ 2。
- **H1**：1 成立 + 2 或 3 至少一项 → 评论家分辨力到位 → 下一步 exp122：蒸馏训练（Vcf_o 作目标的策略改进算子，纯血），另开预注册。
- **H0**：1 不成立（Vcf_o ≈ Vcf_b）→ 兄弟差在回归误差之下，exp67 结论延伸到排序层面；**搜索方向关闭**，评论家工件保留。
- 中间（1 成立但 2/3 都不成立）：评论家对齐金标准但金标准本身不偏向牌效/现物 → 说明"小 EV 差"的真相是现有策略已近最优（exp72 复现），同样不进训练。

## 成本估算
| 项 | 计算 | 金额 |
|---|---|---|
| Stage 0 | 本地 CPU 10–15 min | $0 |
| 数据生成 | 3090 Ti community $0.27/h × ≈0.6 h（35 min + bootstrap） | ≈$0.2 |
| 评论家训练（两臂） | 同 pod ≈0.5–1 h | ≈$0.2 |
| 重放评测 | 工作站 | $0 |
| pod 开销/重试余量 | ×2 | — |
| **合计** | | **≈$0.5，上限 $3**（价格以发射时查询为准；3090 Ti 库存常为 Low，备选 A4000 com $0.17） |
工程量（我的时间）：`--cf-to-end` 半天内；数据生成器 + 4 维头 + 回归脚本 + 重放评测 ≈ 1 天；无 Rust 改动、无训练器改动。
纯血线累计 ≈$501（目标 $200 早已超），本实验增量 <$3，但**Stage 1 发射仍须用户批**。

## 心跳/工件规矩（发射时执行）
- Stage 1 pod：发射相 15 min 死线（数据生成日志无新行即 ALERT）、训练相 30 min STALL；工作站侧监控调 API delete-pod 兜底（SKILLS 09-01 教训）。
- TB 三件套：`experiments/_cloud_mirror/exp121/tensorboard` + 周期 rsync + `--logdir_spec` 加 `exp121_LIVE`。
- 数据集不回传；ckpt + 指标 json rsync 回工作站；GCS 命名空间 `gs://llm-mahjong-experiments/exp121/`。

## Progress
- [10-25 15:40] 预注册写就，未发射。待：① 用户批 Stage 1 预算；② Stage 0 门槛结果。

## Results
（空）

## Conclusion
（空）

## Artifacts
| Path | Size | Description |
|---|---|---|
