# exp15：vit 谱系标度冲顶（冠军配方迁移 + 样本量假说检验）

- **Date**: 预注册 2026-08-16 ~16:20 本地；发射待机器（c3 跑完 A2 自动接棒，或更早拿到新机）  **Status**: pre-registered
- **Git**: 394d592 + 未提交预注册
- **Env**: GCP g2-standard-32 on-demand（L4），runner run_dnn_cloud.sh
- **对照**: vit240 = `dnn_vit_rl_20260815r4/games_240000.pt`（续训起点）；e700 = 现冠军

> **主线声明（2026-08-16 晚，用户决策）**：本实验是**纯自对弈主线（AlphaZero 志向）**的
> 当前代表——起点 vit240 与全部后续训练均零人类/教师知识。exp14（教师 warm-start）仅作
> 参照系：exp15 曲线与 exp14 参照曲线的差距收敛程度，是「自行探索」成色的直接读数。

## Purpose & Hypothesis
双重目的（2026-08-16 用户决策：transformer 谱系更强则押注 transformer）：
1. **用户样本量假说**：纯 RL 前 ~240k 局在交「规则税」，强度曲线远未饱和——若为真，
   续训应持续显著上涨。vit 每局学习效率碾压 cnn（同 240k：−533 vs −2639 分），是最佳载体。
2. **冠军挑战**：把 exp9/12/13 定稿的冠军配方（ppo_epochs=1 + 饱和后熵一步 0.01）迁移到
   vit 谱系并拉到 1.2M 局，挑战 e700。

## Method
- `--resume` 自 vit240（240k，其原配方 ppo_epochs=4/熵 0.03）。
- 配方切换（均为已定稿结论，非本实验变量）：ppo_epochs 4→1（exp9：防 240k 后倒退）；
  entropy_coef 0.03→0.01（exp12/13 台阶律：vit win_rate 自 120k 起已饱和于 ~0.60，
  熵平于 1.26，已处「饱和后」阶段）。
- 全程本地 elo_ladder_watcher watch 模式：每 ≥40k 局对 3 邻近锚点评 Elo，TB 出曲线
  （首个带训练中排位轨迹的 run）。

## Config
vit-r4 全套参数 + `--arch vit_small --resume <seed_vit240.pt> --ppo_epochs 1
--entropy_coef 0.01 --total_games 1200000 --milestones 400000,700000,1200000
--ckpt_every 10 --seed 42`（完整快照 config.json 由 GCS 同步）

## Success Criteria（发射前定死）
1. **主判定 A（冠军战）**：1.2M 终点 vs e700，200 副复式，超 95% CI 为胜。
2. **主判定 B（样本量假说）**：ladder 分差标度曲线 240k→1.2M 净增 ≥ +800 分且终点
   vs vit240 直接竞技场显著正；若 400k 后每 300k 局增幅 <+300 且 CI 含 0，
   判「该规模下样本量假说不成立」（假说可证伪条款）。
3. 训练健康：熵 ≥0.4 地板（exp12 坍缩卫）；无 NaN；吞吐 ≥12 局/s；KL<0.03。

## Progress
- [2026-08-16 ~16:20] 预注册。发射机制：exp1415_queue_watcher（marker 文件防双发，
  修正了 pgrep 自匹配 bug 的教训）。

## Results
| Metric | This run | Baseline | Success criterion |
|---|---|---|---|
| （待运行） | | | |

## Conclusion
（待运行）

## Artifacts
| Path | Size | Description |
|---|---|---|
| gs://llm-mahjong-experiments/dnn_exp15_vitscale_20260816/ | — | 云端主目录 |
