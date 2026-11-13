# 冠军模型卡与部署手册

> **本文件是「现役最强模型如何配置、如何跑、要多少资源」的唯一权威说明。**
> 冠军易主、新纪元重校、部署形态变更 ⇒ 必须在**同一 PR** 里更新本文件（清单见 [§7 版本更新流程](#7-版本更新流程新冠军加冕后必须做的事)）。
> 榜单原始数字在 [experiments/LEADERBOARD.md](../experiments/LEADERBOARD.md)，实验原委在
> [experiments/INDEX.md](../experiments/INDEX.md) / [experiments/FINDINGS.md](../experiments/FINDINGS.md)。

## 1. 现役冠军速览（2026-11-12）

**部署冠军两个并列：exp124 @ 256M（纯血谱系，我们评测里最强）与 bc49（人类先验谱系，BC 线冠军、所有读数的参照）。**
两者网络结构、观测编码、动作空间完全相同，实战与终审一律 T=0。

| 项 | **exp124 @ 256M** | **bc49** |
|---|---|---|
| 全称 | exp124 A ConvFormer，2.56 亿局里程碑（Stage 2 的 `games_256000000.pt`） | `bc49_convformer_m_v3r_m46_full` |
| 谱系 | 纯血（零人类数据、零外部模型，从零自对弈） | 人类先验（天凤凤凰卓牌谱行为克隆） |
| 加冕 | 2026-11-12（用户决定与 bc49 并列部署） | 2026-08-27 exp49-B；纪元 6 全池重校后仍是部署冠军 |
| 架构 | `convformer_m_v3r_m46` — ConvFormer（d=160 / 6 层 / 5 头）+ Mortal 式 46 槽结构化动作头 | 同左 |
| 参数量 | **2,004,239（2.00M）**；checkpoint **24.2 MB**（含优化器状态） | **2,004,239（2.00M）**；fp32 checkpoint **8.05 MB** |
| 观测编码 | `v3r`：**56 平面 × 34 张 + 29 标量**（完整公开记录 + 赤宝牌平面） | 同左 |
| 动作空间 | `mortal46`（46 槽：34 打牌 + 3 赤五 + 立直/吃×3/碰/杠/和/流局/pass） | 同左 |
| 训练信号 | 纯自对弈 PPO，**2.56 亿局**，单种子；无人类牌谱、无外部模型 | 纯 BC（无 RL）：训练 **18,458 局**凤凰卓牌谱 + holdout 1,000 局（语料共 20.9k 局） |
| **对 bc49 头对头**（T=0 单局复式） | **0.5230 ± 0.0033**（20,000 副牌山，高出五五开 7 个标准误）；同牌山比 192M 检查点高 +1.09 ± 0.43 pp | —（参照） |
| BC 精度 | — | holdout **0.8059**（649,328 决策；打牌 0.771 / 副露 0.743 / 立直 0.795）；出货权重 = epoch 6 |
| 部署刻度 Elo（纪元 6 梯子） | 未上梯子 | **1189.0 ± 7.9**（候选 T=0 vs 13 锚 T=1，n=200/锚，纪元 6 池） |
| 外部参照（真 Mortal 298k） | 192M 检查点：同牌山上对 bc49 的份额比 Mortal 高 +0.75 ± 0.43 pp（2 万牌山）；四方半庄 uma 192M − Mortal −495 ± 978（同档）。256M 未对 Mortal 测 | Mortal 1199.6 ± 8.0（同协议）→ **北极星差距 ~10 ± 11** |
| 防守探针 | 144M 检查点：安慰剂配对弃和探针 0.171（bc49 0.165）；256M 未测 | `defense_iq` **0.184** |
| 人类刻度 | 133M 检查点雀魂 maka 3 局 S / B- / S；256M 未测 | 雀魂 maka 两轮 **S+** |
| **推理温度** | **T = 0（贪心）** | **T = 0（贪心）**——同一模型 T=1 采样实测弱 **484 ± 418 点/副**（exp25） |

**加冕依据与范围**：[§7](#7-版本更新流程新冠军加冕后必须做的事) 的判据（双方 T=0 头对头、n ≥ 1000 单局显著为正）在单局刻度满足。
范围戳：单训练种子、本项目复式协议与引擎、单局刻度；成对半庄裁决（≥ 1,200 场）和纪元 6 梯子评级没有在 exp124 检查点上跑过。
训练仍在继续（Stage 2 目标 384M 局），384M 判决出来后本卡随之更新。

**为什么不是 exp46-I**：exp46-I gen10 的梯子分 1198.0 名义更高，但两者**双 T=0 头对头 1000 局
胜份 0.5005 ± 0.0158**（完全打平），且 I 是在 bc49 上继续 RL 的产物。按「打平不换冠军」规则，
人类先验谱系的部署形态仍用 bc49。纯血（AlphaZero 式，零人类知识）谱系的冠军现在是上表的 exp124 @ 256M，前任 exp27-A（1064.5）见 [§8](#8-冠军版本历史)。

## 2. 拿到 checkpoint

权重**不入 git**（`experiments/` 整体 gitignore），权威副本在 GCS：

```bash
# exp124 @ 256M（24.2 MB；md5 c7d993be883ae0b8b4f11508d385c3fb）
mkdir -p experiments/exp124s2_S2
gsutil cp gs://llm-mahjong-experiments/checkpoints/pure_lineage/exp124_convformer_m_v3r_m46_games_256000000.pt experiments/exp124s2_S2/games_256000000.pt
# bc49（8.05 MB）
gsutil cp gs://llm-mahjong-experiments/checkpoints/human_lineage/bc_convformer_m_v3r_m46_best.pt experiments/_anchors_epoch6/bc49.pt
```

| 模型 | 副本 | 路径 | 用途 |
|---|---|---|---|
| exp124 @ 256M | GCS（权威） | `gs://llm-mahjong-experiments/checkpoints/pure_lineage/exp124_convformer_m_v3r_m46_games_256000000.pt` | 分发 |
| exp124 @ 256M | 本地 | `experiments/exp124s2_S2/games_256000000.pt` | 日常跑；也是训练机拉回的原件（`milestones.md5` 已核对） |
| bc49 | GCS（权威） | `gs://llm-mahjong-experiments/checkpoints/human_lineage/bc_convformer_m_v3r_m46_best.pt` | 分发 |
| bc49 | 本地锚池 | `experiments/_anchors_epoch6/bc49.pt` | 梯子锚点 / 日常跑 |
| bc49 | 训练产物 | `experiments/exp49_20260827_205132/B/bc_convformer_m_v3r_m46_best.pt` | 出处存档 |

bucket 为私有项目资源。没有访问权限的读者可按 [§5](#5-从零复现冠军bc-训练配方) 用公开的天凤牌谱自行复现同一模型。
**Mortal 权重（`data/mortal_ext/`）只作参照，不得再分发。**

## 3. 环境与配置

```bash
conda activate rlhf_mahjong          # Python 3.10 + torch 2.12 (cu130)
python -m pytest tests -q            # ~196 项，全绿再上牌桌
```

- **推理只需 3 个包**：`torch`、`numpy`、`mahjong`（点数计算）。没有 GPU 也能跑（见 §4）。
- **没有配置文件**：所有入口都靠命令行 flags（项目规则：不建顶层 `configs/`）。
- **checkpoint 自描述**：blob 里带 `arch` / `encoder_variant`，`load_dnn` 自动实例化正确的网络与编码器 —— 
  所有入口只需要 `--ckpt <路径>`，不要也不用手写架构名。
- **唯一必须显式设置的部署参数是温度**：`--temperature 0`（服务端）/ `--dnn_temperature 0`（竞技场）。

## 4. 四种跑法

下面的命令以 bc49 的路径为例；把路径换成 `experiments/exp124s2_S2/games_256000000.pt` 就是跑 exp124 @ 256M（checkpoint 自描述，别的不用改）。2026-11-12 已用该文件冒烟过 4.1 的桥接校验（`scripts/verify_mjai_bridge.py --games 10`）和 4.2 的竞技场。

### 4.1 雀魂实战（部署形态）
```bash
PYTHONPATH=. python scripts/serve_mjai_bot.py --ckpt experiments/_anchors_epoch6/bc49.pt --temperature 0 --log experiments/majsoul_sessions/bc49_$(date +%Y%m%d_%H%M%S)/mjai_session.jsonl
```
HTTP 服务默认 `127.0.0.1:8765`（**无鉴权，不要暴露到公网**）。完整接入流程见
[tools/majsoul_bridge/README.md](../tools/majsoul_bridge/README.md)（MahjongCopilot 插件、两种模式、留底与计分），
**Windows 打牌机**（三处补丁 + 排障速查）见 [tools/majsoul_bridge/WINDOWS.md](../tools/majsoul_bridge/WINDOWS.md)，
六步速览在 [README.md 的「Live play on Majsoul」](../README.md#live-play-on-majsoul-windows-client-machine)（中文：[README.zh.md](../README.zh.md#雀魂实战测试windows-打牌机)）。
使用第三方自动化工具违反雀魂 ToS，有封号风险。

### 4.2 头对头竞技场（复式牌，单局刻度）
```bash
PYTHONPATH=. python scripts/run_arena_dnn.py --dnn_a experiments/_anchors_epoch6/bc49.pt --dnn_b <挑战者.pt> --deals 1000 --seed0 46000001 --dnn_temperature 0 --parallel 20 --out experiments/probes/bc49_vs_challenger.json
```
两边同 seed 同场上下文、A−B 对称配对分差。**跨部署形态比较必须双方都 T=0。**

### 4.3 半庄刻度（裁决刻度）
```bash
PYTHONPATH=. python scripts/run_hanchan_arena.py --a experiments/_anchors_epoch6/bc49.pt --b <挑战者.pt> --games 300 --seed0 46000001 --out experiments/probes/bc49_hanchan.json
```
连庄/本场/流满/uma 全套，约 1.8× 单局放大。终审 n ≥ 300。

### 4.4 锚点梯子评级（榜单数字的来源）
```bash
PYTHONPATH=. python scripts/run_elo_league.py rate --ckpt <候选.pt> --label <标签>_T0 --deals 200 --seed0 <新种子> --temperature 0 --parallel 20
```
需要本地锚池 `experiments/elo_league/anchors.json` + 13 个锚 checkpoint（均不入 git）。
外部读者没有锚池时用 4.2 / 4.3 与本模型直接对打即可。
引擎指纹不符会拒跑（纪元守卫）——这是设计，不要用 `--allow_engine_mismatch` 绕过后拿去比较。

### 4.5 检视台（看牌局与逐步概率）
```bash
conda run -n rlhf_mahjong python tools/webui/server.py --port 8642
```

## 5. 从零复现冠军（BC 训练配方）

```bash
python scripts/train_human_bc.py --arch convformer_m_v3r_m46 \
  --limit_games 0 --holdout_games 1000 \
  --max_epochs 30 --patience 3 --min_delta 0.0005 \
  --batch 1024 --lr 3e-4 --workers 10 --seed 0 \
  --out experiments/bc49_repro_$(date +%Y%m%d_%H%M%S)
```
- 数据：`data/tenhou/raw`（凤凰卓 20.9k 局，本地 348 MB / 打包 134 MB；
  `gs://llm-mahjong-experiments/data/tenhou/tenhou_raw_20526.tar.zst`），采集脚本 `scripts/harvest_human_bc.py`；
- 实测（本机 RTX 4080 + 24 核，每 epoch 12.03M 训练决策）：**epoch 6 = 出货权重（144 分钟）**，早停在第 10 epoch 触发（patience 3 / min_delta 5e-4），全程 206 分钟；
- 可选加速：`scripts/materialize_bc.py` 物化张量缓存（约 31 GB，epoch 20.5 → 6.9 分钟）；
- **超参不要动**：exp49-A 直测过 `--lr 1e-4` 与 `--lr_schedule cosine`，两者对 11M 模型均 <+0.1pp；
  容量也不是杠杆（11M SEres 三次不敌本 2M ConvFormer）。**当前唯一未饱和的杠杆是数据量**
  （9× 数据 = +3.7pp 精度 = +27~57 Elo，20k 局仍未见饱和）。

### exp124 @ 256M（纯血自对弈 RL 配方）

同一条谱系分三段训练；预注册、逐里程碑读数与全部进度在
[experiments/exp124_convformer_scale_prereg/EXPERIMENT.md](../experiments/exp124_convformer_scale_prereg/EXPERIMENT.md)。

```bash
# 段 1：0 → 64M 局
python3 scripts/train_dnn_ppo.py --engine rust --arch convformer_m_v3r_m46 --lr 1e-4 \
  --batch 4096 --ppo_epochs 1 --clip_eps 0.2 --entropy_coef 0.03 --entropy_schedule 0:0.03 --gamma 0.995 --gae_lambda 0.95 \
  --dup_k 8 --games_per_iter 2048 --games_per_worker 1024 --workers 1 --target_kl 0.03 --value_coef 0.5 --adv_clamp 5.0 \
  --amp_update --tf32 1 --shaping --shaping_schedule 0:1.0,400000:0.0 \
  --warmup_updates 3000 --fast_trunk --compile --bf16_rollout --rollout_values --profile_pack \
  --ckpt_every 400 --seed 1241 --total_games 64000000 --exp_dir <dir>
```
- 段 2：64M → 192M，在段 1 的命令上加 `--resume <段 1 的 latest.pt> --lr 3e-5 --total_games 192000000`，其余不变；
- 段 3（Stage 2）：192M → 256M，段 2 配方不变（lr 3e-5，seed 1441），rollout 换成流式
  （`--stream_rollout --stream_chunk_games 128 --stream_queue 1 --m46_one_forward`）；生效配置快照在
  `experiments/exp124s2_S2/config.json`。流式 rollout 的代码在分支 `claude/perf-rollout-overlap`
  （commit 1de99e5，仅本地、未合入 master）；
- 机器与成本：到 192M ≈ $121；192M → 256M 在 RunPod Secure RTX 3090 + 32 vCPU（$0.50/h）上约 54 小时 ≈ $27，
  流式吞吐 ≈ 380 局/s；
- 评测：每个里程碑对 bc49 做 T=0 单局复式（逐牌山结果在 `experiments/probes/pairdiff_*`）。单种子。

## 6. 资源需求

| 场景 | 硬件 | 实测 | 备注 |
|---|---|---|---|
| **实战推理（一席）** | CPU 1 核，无需 GPU | 单线程 batch-1 **≈ 4.6 ms/决策**（本机 24 核 4080，测量时机器有其他负载 ⇒ 保守上限）；进程 RSS ≈ 1.1 GB（几乎全是 torch 运行时） | 一局一席约 10² 次决策（估算：全桌 ~430 次网络调用/局），半庄合计几秒钟，远低于对局时限；雀魂桥接默认 `--device cpu` |
| **GPU 批推理**（评测/训练内循环） | RTX 4080 16 GB | batch 512 ≈ 7.8 ms → **~65k 决策/s**；显存 0.22 GB | `scripts/train_dnn_ppo.py --gpu_infer` 走同一条路径 |
| **一次梯子评级** | 24 核 CPU | **≈ 1 分钟**（13 锚 × 200 副，`--parallel 20`） | 13 个锚 checkpoint 合计 218 MB |
| **BC 复现训练** | RTX 4080 + 24 核 + 348 MB 数据 | **206 分钟 / 10 epoch**；物化缓存另需 31 GB 磁盘 | 本地就够，不必上云 |
| **纯血从零训练（exp124）** | RunPod RTX 3090 + 32 vCPU（$0.50/h） | 流式 rollout **≈ 380 局/s**；到 256M 局累计 **≈ $148** | 配方见 §5；自对弈吞吐由 vCPU 决定 |
| **继续 RL（可选）** | GCP `g4-standard-48` flex | 1.0M 局 ≈ 85 分钟 ≈ **$3** | 配方与成本见 [docs/gcp_compute_cost_and_quota.md](gcp_compute_cost_and_quota.md)；自对弈吞吐由 vCPU 决定，不是 GPU |

单机部署（模型机=打牌机）完全可行；两机部署用 SSH 隧道转发 8765 端口。

## 7. 版本更新流程（新冠军加冕后必须做的事）

**加冕判据**：候选在**双方 T=0** 的头对头（n ≥ 1000 单局或 n ≥ 300 半庄）上对现役冠军显著为正 —— 
梯子分只用于排名，绝对值对 T=0 候选系统性高估（bc49 的采样税实测 ≈ +29 Elo），单独不构成加冕依据。

判据满足后，**同一个 PR 里**：
1. `experiments/LEADERBOARD.md` —— 新一轮榜单（纪元校准后必更）；
2. **本文件** —— §1 速览表整体替换、§2 checkpoint 路径、§5 复现配方，并在 §8 追加一行历史；
3. `README.md`（英文，权威）与 `README.zh.md` 的「Current status / 当前状态」冠军行；
4. [tools/majsoul_bridge/README.md](../tools/majsoul_bridge/README.md) 的 checkpoint 行（实战 runbook 里出现的默认 ckpt）；
5. 把新 checkpoint 上传到 `gs://llm-mahjong-experiments/checkpoints/<谱系>/`；
6. `experiments/INDEX.md` 一行总账 + 对应 `EXPERIMENT.md` 判决（本来就是硬规则）。

**旧冠军不删**：降级为锚点留在池里（纪元重校时一起重评），历史行保留在 §8。

## 8. 冠军版本历史

| 加冕日 | 模型 | 谱系 | 架构 / 输入 | 当时刻度 | 现状 |
|---|---|---|---|---|---|
| 2026-08-16 | exp12-E 700k | 纯血 | cnn_m / v1 | 997.5（纪元 6 重评） | 锚点 `e700` |
| 2026-08-18 | exp17-C（GAE） | 纯血 | cnn_m / v1 | 1079.7（纪元 2 口径） | 首个雀魂实战部署；已退役 |
| 2026-08-23 | exp27-A | 纯血 | `cnn_m_r` / v1r（首个识赤五） | 1122（T=0，纪元 3） | 纯血谱系前冠军（2026-11-12 由 exp124 接替），1064.5（纪元 6） |
| 2026-08-27 | **bc49** | 人类先验 | `convformer_m_v3r_m46` / v3r | 1191.4（T=1，纪元 5） | **现役部署冠军（并列）**，1189.0（纪元 6） |
| 2026-11-12 | **exp124 @ 256M** | 纯血 | `convformer_m_v3r_m46` / v3r | 对 bc49 0.5230 ± 0.0033（T=0 单局复式，20,000 牌山） | **现役部署冠军（并列）**，纯血谱系冠军 |

并列参照（非冠军）：exp46-I gen10 1198.0（bc49 + RL，双 T=0 头对头打平）、bc51 v3r2 1192.1、
exp52-P3 hrfC 1187.0、真 Mortal 298k 1199.6（外部黑盒参照）。

## 9. English quick reference

**Two deployment champions (2026-11-12): exp124 @ 256M and bc49.** Same network: ConvFormer trunk
(d=160 / 6 layers / 5 heads) with Mortal's 46-slot structured action head, **2,004,239 params**,
encoder `v3r` = **56 planes × 34 tiles + 29 scalars**. **Always run either at temperature 0.**

- **exp124 @ 256M** (pure lineage): self-play PPO from scratch, 256M games, no human data, one seed.
  Head-to-head against bc49 on 20,000 duplicate walls (single deals, both greedy): **0.5230 ± 0.0033**,
  7 standard errors above parity, and +1.09 ± 0.43 pp above the 192M checkpoint on the same walls.
  Not rated on the epoch-6 ladder; the pairwise hanchan verdict has not been run. Checkpoint 24.2 MB
  (optimizer state included). Training continues to 384M.
- **bc49** (`bc49_convformer_m_v3r_m46_full`, human-prior lineage, crowned 2026-08-27): pure behaviour
  cloning on **18,458 Tenhou Houou games** (holdout 1,000), holdout accuracy **0.8059**, 8.05 MB fp32.
  Ratings: **1189.0 ± 7.9** deployment scale (candidate T=0 vs 13 anchors at T=1, n=200/anchor);
  real Mortal 298k = 1199.6 ± 8.0 on the same protocol; `defense_iq` 0.184; Majsoul maka **S+** twice.
  Sampling at T=1 costs ~484 points/deal (exp25).

The commands below use the bc49 path; pass `experiments/exp124s2_S2/games_256000000.pt` to run exp124 @ 256M
(`gsutil cp gs://llm-mahjong-experiments/checkpoints/pure_lineage/exp124_convformer_m_v3r_m46_games_256000000.pt experiments/exp124s2_S2/games_256000000.pt`).
Its training recipe is in section 5.

```bash
# 1. weights (not in git; private bucket)
gsutil cp gs://llm-mahjong-experiments/checkpoints/human_lineage/bc_convformer_m_v3r_m46_best.pt experiments/_anchors_epoch6/bc49.pt

# 2. live play against humans (MJAI/MahjongCopilot bridge; localhost only, no auth)
PYTHONPATH=. python scripts/serve_mjai_bot.py --ckpt experiments/_anchors_epoch6/bc49.pt --temperature 0

# 3. head-to-head vs a challenger (duplicate deals, both sides greedy)
PYTHONPATH=. python scripts/run_arena_dnn.py --dnn_a experiments/_anchors_epoch6/bc49.pt --dnn_b <challenger.pt> --deals 1000 --seed0 46000001 --dnn_temperature 0 --parallel 20 --out experiments/probes/h2h.json

# 4. reproduce bc49 from public Tenhou logs (~3.5 h on one RTX 4080)
python scripts/train_human_bc.py --arch convformer_m_v3r_m46 --limit_games 0 --holdout_games 1000 \
  --max_epochs 30 --patience 3 --min_delta 0.0005 --batch 1024 --lr 3e-4 --workers 10 --seed 0 --out <dir>
```

The checkpoint is self-describing (`arch` + `encoder_variant` in the blob), so every entry point only
needs `--ckpt`. Inference needs `torch`, `numpy` and `mahjong` — **CPU is enough**: ~4.6 ms per decision
single-threaded, ~1.1 GB RSS, no GPU. A GPU (RTX 4080) gives ~65k decisions/s at batch 512 and is only
needed for training or bulk evaluation. Full details, resource table and update procedure: sections 1–8 above (Chinese).
