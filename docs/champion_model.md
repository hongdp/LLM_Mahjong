# 冠军模型卡与部署手册

> **本文件是「现役最强模型如何配置、如何跑、要多少资源」的唯一权威说明。**
> 冠军易主、新纪元重校、部署形态变更 ⇒ 必须在**同一 PR** 里更新本文件（清单见 [§7 版本更新流程](#7-版本更新流程新冠军加冕后必须做的事)）。
> 榜单原始数字在 [experiments/LEADERBOARD.md](../experiments/LEADERBOARD.md)，实验原委在
> [experiments/INDEX.md](../experiments/INDEX.md) / [experiments/FINDINGS.md](../experiments/FINDINGS.md)。

## 1. 现役冠军速览（2026-08-30，纪元 6）

| 项 | 值 |
|---|---|
| **模型名** | **bc49**（全称 `bc49_convformer_m_v3r_m46_full`） |
| 谱系 | 人类先验谱系（天凤凤凰卓牌谱行为克隆） |
| 加冕 | 2026-08-27 exp49-B；纪元 6 全池重校后仍是部署冠军 |
| 架构 | `convformer_m_v3r_m46` — ConvFormer（d=160 / 6 层 / 5 头）+ Mortal 式 46 槽结构化动作头 |
| 参数量 | **2,004,239（2.00M）**；fp32 checkpoint **8.05 MB** |
| 观测编码 | `v3r`：**56 平面 × 34 张 + 29 标量**（完整公开记录 + 赤宝牌平面） |
| 动作空间 | `mortal46`（46 槽：34 打牌 + 3 赤五 + 立直/吃×3/碰/杠/和/流局/pass） |
| 训练信号 | 纯 BC（无 RL）：训练 **18,458 局**凤凰卓牌谱 + holdout 1,000 局（语料共 20.9k 局） |
| BC 精度 | holdout **0.8059**（649,328 决策；打牌 0.771 / 副露 0.743 / 立直 0.795）；出货权重 = epoch 6 |
| **部署刻度 Elo** | **1189.0 ± 7.9**（候选 T=0 vs 13 锚 T=1，n=200/锚，纪元 6 池） |
| 外部参照 | 真 Mortal 298k = 1199.6 ± 8.0（同协议）→ **北极星差距 ~10 ± 11** |
| 防守探针 | `defense_iq` **0.184** |
| 人类刻度 | 雀魂 maka 两轮 **S+** |
| **推理温度** | **T = 0（贪心）**——实战与终审一律；同一模型 T=1 采样实测弱 **484 ± 418 点/副**（exp25） |

**为什么不是 exp46-I**：exp46-I gen10 的梯子分 1198.0 名义更高，但两者**双 T=0 头对头 1000 局
胜份 0.5005 ± 0.0158**（完全打平），且 I 是在 bc49 上继续 RL 的产物。按「打平不换冠军」规则，
部署形态仍用 bc49。纯血（AlphaZero 式，零人类知识）谱系的冠军是另一条线的 exp27-A（1064.5）。

## 2. 拿到 checkpoint

权重**不入 git**（`experiments/` 整体 gitignore），权威副本在 GCS：

```bash
gsutil cp gs://llm-mahjong-experiments/checkpoints/human_lineage/bc_convformer_m_v3r_m46_best.pt experiments/_anchors_epoch6/bc49.pt
```

| 副本 | 路径 | 用途 |
|---|---|---|
| GCS（权威） | `gs://llm-mahjong-experiments/checkpoints/human_lineage/bc_convformer_m_v3r_m46_best.pt` | 分发 |
| 本地锚池 | `experiments/_anchors_epoch6/bc49.pt` | 梯子锚点 / 日常跑 |
| 训练产物 | `experiments/exp49_20260827_205132/B/bc_convformer_m_v3r_m46_best.pt` | 出处存档 |

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

### 4.1 雀魂实战（部署形态）
```bash
PYTHONPATH=. python scripts/serve_mjai_bot.py --ckpt experiments/_anchors_epoch6/bc49.pt --temperature 0 --log experiments/majsoul_sessions/bc49_$(date +%Y%m%d_%H%M%S)/mjai_session.jsonl
```
HTTP 服务默认 `127.0.0.1:8765`（**无鉴权，不要暴露到公网**）；MahjongCopilot 插件安装、Windows 打牌机、
留底与计分脚本见 [tools/majsoul_bridge/README.md](../tools/majsoul_bridge/README.md)。
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

## 6. 资源需求

| 场景 | 硬件 | 实测 | 备注 |
|---|---|---|---|
| **实战推理（一席）** | CPU 1 核，无需 GPU | 单线程 batch-1 **≈ 4.6 ms/决策**（本机 24 核 4080，测量时机器有其他负载 ⇒ 保守上限）；进程 RSS ≈ 1.1 GB（几乎全是 torch 运行时） | 一局一席约 10² 次决策（估算：全桌 ~430 次网络调用/局），半庄合计几秒钟，远低于对局时限；雀魂桥接默认 `--device cpu` |
| **GPU 批推理**（评测/训练内循环） | RTX 4080 16 GB | batch 512 ≈ 7.8 ms → **~65k 决策/s**；显存 0.22 GB | `scripts/train_dnn_ppo.py --gpu_infer` 走同一条路径 |
| **一次梯子评级** | 24 核 CPU | **≈ 1 分钟**（13 锚 × 200 副，`--parallel 20`） | 13 个锚 checkpoint 合计 218 MB |
| **BC 复现训练** | RTX 4080 + 24 核 + 348 MB 数据 | **206 分钟 / 10 epoch**；物化缓存另需 31 GB 磁盘 | 本地就够，不必上云 |
| **继续 RL（可选）** | GCP `g4-standard-48` flex | 1.0M 局 ≈ 85 分钟 ≈ **$3** | 配方与成本见 [docs/gcp_compute_cost_and_quota.md](gcp_compute_cost_and_quota.md)；自对弈吞吐由 vCPU 决定，不是 GPU |

单机部署（模型机=打牌机）完全可行；两机部署用 SSH 隧道转发 8765 端口。

## 7. 版本更新流程（新冠军加冕后必须做的事）

**加冕判据**：候选在**双方 T=0** 的头对头（n ≥ 1000 单局或 n ≥ 300 半庄）上对现役冠军显著为正 —— 
梯子分只用于排名，绝对值对 T=0 候选系统性高估（bc49 的采样税实测 ≈ +29 Elo），单独不构成加冕依据。

判据满足后，**同一个 PR 里**：
1. `experiments/LEADERBOARD.md` —— 新一轮榜单（纪元校准后必更）；
2. **本文件** —— §1 速览表整体替换、§2 checkpoint 路径、§5 复现配方，并在 §8 追加一行历史；
3. `README.md` 与 `README.en.md` 的「当前状态 / Current status」冠军行；
4. [tools/majsoul_bridge/README.md](../tools/majsoul_bridge/README.md) 的 checkpoint 行（实战 runbook 里出现的默认 ckpt）；
5. 把新 checkpoint 上传到 `gs://llm-mahjong-experiments/checkpoints/<谱系>/`；
6. `experiments/INDEX.md` 一行总账 + 对应 `EXPERIMENT.md` 判决（本来就是硬规则）。

**旧冠军不删**：降级为锚点留在池里（纪元重校时一起重评），历史行保留在 §8。

## 8. 冠军版本历史

| 加冕日 | 模型 | 谱系 | 架构 / 输入 | 当时刻度 | 现状 |
|---|---|---|---|---|---|
| 2026-08-16 | exp12-E 700k | 纯血 | cnn_m / v1 | 997.5（纪元 6 重评） | 锚点 `e700` |
| 2026-08-18 | exp17-C（GAE） | 纯血 | cnn_m / v1 | 1079.7（纪元 2 口径） | 首个雀魂实战部署；已退役 |
| 2026-08-23 | exp27-A | 纯血 | `cnn_m_r` / v1r（首个识赤五） | 1122（T=0，纪元 3） | **纯血谱系现役冠军**，1064.5（纪元 6） |
| 2026-08-27 | **bc49** | 人类先验 | `convformer_m_v3r_m46` / v3r | 1191.4（T=1，纪元 5） | **现役部署冠军**，1189.0（纪元 6） |

并列参照（非冠军）：exp46-I gen10 1198.0（bc49 + RL，双 T=0 头对头打平）、bc51 v3r2 1192.1、
exp52-P3 hrfC 1187.0、真 Mortal 298k 1199.6（外部黑盒参照）。

## 9. English quick reference

**Current champion — bc49** (`bc49_convformer_m_v3r_m46_full`, crowned 2026-08-27, epoch-6 board):
ConvFormer trunk (d=160 / 6 layers / 5 heads) with Mortal's 46-slot structured action head,
**2,004,239 params (8.05 MB fp32)**, encoder `v3r` = **56 planes × 34 tiles + 29 scalars**.
Pure behaviour cloning on **18,458 Tenhou Houou games** (holdout 1,000), holdout accuracy **0.8059**.
Ratings: **1189.0 ± 7.9** deployment scale (candidate T=0 vs 13 anchors at T=1, n=200/anchor);
real Mortal 298k = 1199.6 ± 8.0 on the same protocol; `defense_iq` 0.184; Majsoul maka **S+** twice.
**Always run it at temperature 0** — sampling costs ~484 points/deal (exp25).

```bash
# 1. weights (not in git; private bucket)
gsutil cp gs://llm-mahjong-experiments/checkpoints/human_lineage/bc_convformer_m_v3r_m46_best.pt experiments/_anchors_epoch6/bc49.pt

# 2. live play against humans (MJAI/MahjongCopilot bridge; localhost only, no auth)
PYTHONPATH=. python scripts/serve_mjai_bot.py --ckpt experiments/_anchors_epoch6/bc49.pt --temperature 0

# 3. head-to-head vs a challenger (duplicate deals, both sides greedy)
PYTHONPATH=. python scripts/run_arena_dnn.py --dnn_a experiments/_anchors_epoch6/bc49.pt --dnn_b <challenger.pt> --deals 1000 --seed0 46000001 --dnn_temperature 0 --parallel 20 --out experiments/probes/h2h.json

# 4. reproduce the champion from public Tenhou logs (~3.5 h on one RTX 4080)
python scripts/train_human_bc.py --arch convformer_m_v3r_m46 --limit_games 0 --holdout_games 1000 \
  --max_epochs 30 --patience 3 --min_delta 0.0005 --batch 1024 --lr 3e-4 --workers 10 --seed 0 --out <dir>
```

The checkpoint is self-describing (`arch` + `encoder_variant` in the blob), so every entry point only
needs `--ckpt`. Inference needs `torch`, `numpy` and `mahjong` — **CPU is enough**: ~4.6 ms per decision
single-threaded, ~1.1 GB RSS, no GPU. A GPU (RTX 4080) gives ~65k decisions/s at batch 512 and is only
needed for training or bulk evaluation. Full details, resource table and update procedure: sections 1–8 above (Chinese).
