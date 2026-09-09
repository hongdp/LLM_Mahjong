# exp77 纯血线规模 32M 局（Rust 引擎 × 社区 3090，$200 计划阶段 C 重启，2026-09-08）

- **Date**: 2026-09-08  **Status**: running（发射）
- **Git**: 见 Progress；RunPod **社区** RTX 3090（32 vCPU，$0.22/h）单臂；纯血同 exp74-A1（零人类数据、零外部模型，pod 上无冠军资产）
- **Env**: `train_dnn_ppo.py --engine rust`（riichi_rs.VecEnv）+ 向量化 GAE 训练器

## Purpose & Hypothesis
exp75 给出对数线性缩放：vs P3 1M 0.523 → 4M 0.558，vs bc49 0.33 → 0.356（≈+1.5–2 pp/倍增）。当时按 $3.1/百万局算，到 M1（0.40）要 $50–100。
Rust 引擎 + 社区 3090 把成本压到 **≈$0.07/百万局**（exp76 探针），本实验把同配方放大到 **32M 局（8×）**，直接检验缩放律是否延续到 3 个倍增之外。
**H1（缩放律延续）**：vs bc49 从 0.356 升到 ≈0.40–0.42（M1），vs P3 ≈0.60+；**H0（饱和）**：16M 处 vs bc49 ≤ 0.36 ⇒ 早停，记录饱和点。
副目标：验证 Rust 引擎在 10 小时级长跑中的稳定性与社区主机的可靠性（15 min 拉取 + `--resume`）。

## Method
- 配方 = exp74-A1（P3 钉死旗标 + `--shaping --shaping_schedule 0:1.0,400000:0.0`，熵表 0:0.03,600000:0.01，lr 1e-4）+ `--engine rust --games_per_worker 1024 --workers 1`；
  `--total_games 32000000`，milestones 每 4M，`--ckpt_every 400`（≈82 万局/ckpt）。
- 在轨（工作站）：每个 4M 里程碑 ckpt 对 P3 终点与 **bc49** 各 n=600 T=0；同时保留同局数 P3 对照（≤1M 无意义，略）。
- 早停规则：16M ckpt vs bc49（600 对）**≤ 0.36** 且 vs P3 终点 ≤ 0.56 ⇒ 饱和，终止。
- 终评（工作站）：终点 vs P3 / exp27A / **bc49** 双段 20k 对；exp75_X 对照；梯子 Elo；防守探针；风格 @bc49。
- 可靠性：pod 消失 ⇒ 用最近拉回的 ckpt `--resume` 到新 pod（同 exp_dir，TB 不重放）。

## Success Criteria（预注册）
1. vs bc49 ≥ 0.40（M1）→ 缩放律延续，继续 128M；0.37–0.40 → 减速，评估是否值得；≤0.36 → 饱和。
2. 缩放曲线（1M/2M/4M/8M/16M/32M vs P3 与 vs bc49）写入 FINDINGS/LEADERBOARD。
- 预算：32M 局 ≈ 10–12 h × $0.22 ≈ **$2.5，上限 $6**（含可能的重发）。

## Progress
