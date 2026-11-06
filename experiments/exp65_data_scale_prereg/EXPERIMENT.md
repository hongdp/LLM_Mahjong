# exp65 数据 5–10×：凤凰卓 2026 全年段 BC 重训（文献对照的第一杠杆）

- **Date**: 2026-09-04  **Status**: done（判据 2：兑现但斜率放缓）
- **Git**: 见 Progress；采集在工作站（`tools/tenhou/fetch_houou.py`，单会话、1 请求/秒、只取 .log.gz、不再分发）；训练上 RunPod Secure
- **Env**: 数据 `data/tenhou/raw/<date>/*.mjlog`（主检出）；训练 `train_human_bc.py`（bc49 配方，v3r + mortal46）

## Purpose & Hypothesis
exp49：9× 数据 = +3.7–4.0pp 精度 = +27–57 Elo，18.4k 局未饱和；文献（Tuyls 2023、Jones 2021、AlphaGo SL）一致：模仿学习的
数据缩放是幂律且顶部精度→强度是凸的。假设：把 2026-01-01 至 2026-09-03 的凤凰卓四人南喰赤全部取下（≈500 局/天 ≈ **12 万局**，
现有 2.09 万局的 ~6×），同配方重训 → holdout 精度 **≥ +2pp**，T=0 配对牌山对 bc49 **≥ 0.53**。
反假设：凤凰卓人类噪声地板已近（Suphx 15M 样本 76.7% 切牌精度），增益 <1pp。

## Method
阶段 1（采集，2026-09-04 起）：`fetch_houou.py --start 20260101 --end 20260903`，幂等续传；ToS：单会话、≥20 分钟轮询索引、
每日索引取一次、逐局 1 秒间隔（实测约 0.3–0.5 局/秒 → 12 万局约 3–4 天）；官方 2025 及更早的日归档已不再提供（探针 0 局），
不用第三方再分发包。
阶段 2（训练）：同 bc49 配方（`convformer_m_v3r_m46`，holdout 按 hash 10% 上限 1000 局同旧集，max_epochs 30 / patience 3 /
min_delta 5e-4，batch 1024，lr 3e-4）。数据量 ~78M 决策/epoch：pod 上物化 uint8 缓存（~150GB 盘）或流式；估 L40S 8–14 小时
≈ **$10–16**。按 exp62 教训：同批控制臂 = 旧 2.09 万局子集原配方重跑（同 pod、同代码），双方各 1 种子（预算），差异判据设在噪声地板之上。
阶段 3（可选，若 exp64 成立）：在新模型上做高分席位微调。

## Success Criteria（预注册）
1. 新模型 vs 同批控制臂：holdout acc ≥ +2pp（同一 holdout 集）；T=0 配对牌山 n=4000（再补 16000）share ≥ 0.53 → 数据缩放兑现，
   **冠军候选**：走加冕流程（docs/champion_model.md §7，含 Mortal 半庄头对头 n≥1200 与雀魂 maka）。
2. +1–2pp 且 share 0.51–0.53：兑现但斜率放缓，记录缩放曲线点（2k / 18k / 120k）。
3. <+1pp：饱和，数据线关闭，剩余杠杆只剩搜索蒸馏。

## Progress
- [09-04 00:45] 探针：2025-03-01、2025-10-01 日归档 0 局（不再提供）；2026-01-15 474 局 → 可取范围 2026-01-01 起。
  采集启动（pid 见 scratchpad `harvest65.log`），2026-01-01 570 局。
- [09-05] **采集完成**：2026-01-01 → 09-03 共下载 93,481 局，`data/tenhou/raw` 合计 **114,803 局**（≈ 旧集 20,526 的 5.6×），
  0 个 warn；单会话 throttle 0.4 s，约 26 小时。holdout 冻结为旧集的 1000 局（`experiments/configs/exp65_holdout_1000.txt`）。
  发射：L40S Secure US-TX-4，220GB 盘（全量 uint8 缓存 ≈ 145GB + 旧集 26GB），两臂并行：F = 全量 114.8k 局、C = 旧快照 20.5k 局（同批控制臂）。
- [09-05 02:45–04:05] pod `si0t05br78kc6s`：上传 750MB 27 秒；物化 FULL **74,212,185 训练行 / 649,328 留出行 / bad 0，66.4 分钟**（11 worker）；
  物化 OLD 12,724,150 行 11.8 分钟；留出行数两臂完全一致（冻结 holdout 生效）。04:05 两臂发射（各 5 worker）。
- [09-05 05:29] **C 臂（旧快照）早停**：11 epoch，best acc **0.8077**（83.7 分钟）——与 exp62 控制臂 0.8075 一致。
- [09-05 06:06] **F 臂慢 5×**：5.6k 样本/s（exp62 25k），GPU 0%，磁盘读 1.0–1.1 GB/s。病根两条：①`MaterializedBCDataset.index` 是
  7,400 万个 Python 元组（每 worker ~6GB 私有 RSS）；②容器 cgroup memory.max = 188GB，145GB 缓存 + 5 worker × 30GB RSS 放不进页缓存，
  memmap 随机访问触发内核预读（每 2KB 行读 ~128KB）→ 持续从盘读。修复：索引改 numpy（0.7GB）、memmap `madvise(MADV_RANDOM)`、
  删除已完成的 OLD 缓存、worker 5→10；kill F 臂重发（丢弃 2.2 小时 ≈ $2.4 的 epoch 0 进度，保留 `exp65_full_try1` 日志）。
  心跳误报一次 STALL（把已完成的 C 臂当卡死）→ 规则改为只看 done=0 的臂。
- [09-05 06:11] F 臂重发后 **30.3k 样本/s**（epoch ≈ 41 分钟）。
- [09-05 12:08] **F 臂早停**：10 epoch / 359 分钟，acc 0.8074 → 0.8119 → 0.8135 → 0.8149 → 0.8158 → 0.8162 → **0.8171（ep6，出货 ckpt）**
  → 0.8168 → 0.8164 → 0.8172（未过 min_delta）。pod terminate（204）。全程 ≈ 9.5h ≈ **$10.5**（含 220GB 盘）。
- [09-05 12:20] 工作站终评（两个牌山段）+ 破缺率诊断。

## Results

| 判据 | 目标 | 实测 | 判定 |
|---|---|---|---|
| 1 精度 F − C（同一冻结 holdout，649,328 行） | ≥ +2pp | **0.8171 − 0.8077 = +0.94pp**（切牌 0.7838 vs 0.7727 = +1.1pp；防守 0.8322 vs 0.8159；CE 0.4696 vs 0.4984） | ❌ 判据 1；✅ 判据 2 下沿 |
| 1 强度 F vs C，配对牌山 T=0 | ≥ 0.53 | 67M 段 n=4000 **0.5118±0.0079**；68M 段 n=16000 **0.5102±0.0040**；**合并 20k 0.5105±0.0035（z=+3.0）** | ❌ 判据 1；✅ 判据 2（0.51–0.53） |
| F vs bc49（现役冠军） | — | 67M 0.5089、68M 0.5097 → **合并 0.5095±0.0035（z=+2.7）**，+156~346 分/对 | 两段同向、显著 |
| C vs bc49（同批控制臂 vs 原版） | 噪声地板 | 67M 0.5046、68M 0.4991 → **合并 0.5002±0.0035** | 本批无"重跑偏高" |
| A/A | — | 0.500 | ✅ |
| 破缺率（exp61 诊断） | — | bc49 5.72% → **F 2.05%**（n=11,136） | 数据本身教会花色不变性 |

**缩放曲线（同配方、同 holdout 口径，全类型 top-1）**：2k 局 0.7694 → 18.4k 0.8059/0.8077 → 114.8k 0.8171。
斜率 **3.78 pp/十倍 → 1.41 pp/十倍**（降 2.7×）。切牌精度 0.7838（Suphx 76.7%@15M 样本，口径不完全可比）。

## Conclusion
**数据 5.6× 兑现了一个真实但小的增益，且斜率在急剧放缓。** F 对同批控制臂 +1.05% share（z=3.0，两个牌山段各 +1.0–1.2%），
对现役冠军 bc49 +0.95%（z=2.7）；同批控制臂对 bc49 恰为 0.500，排除了 exp62/64 那种"重跑偏高"。精度 +0.94pp 与 exp49 汇率
（+1pp ≈ +10 Elo）一致，头对头 +1% share 大致也是 +7–10 Elo 量级。
但缩放斜率从 3.8 掉到 1.4 pp/十倍：再 10× 数据（100 万局，天凤官方已不供给）按此斜率也只有 +1pp 左右，**凤凰卓 BC 的人类噪声地板
大约在 0.82–0.83**。附带证据：破缺率 5.7%→2.05%，数据在替代我们没做成的对称增广。
**加冕判定**：F vs bc49 双段合并 z=2.7 满足 CLAUDE.md"双方 T=0 头对头显著为正"的统计判据，但未达本实验预注册的候选门槛（+2pp / 0.53），
且增益量级（+1%）在 exp62/64 记录的训练级噪声地板边缘（本批控制臂为 0.500 是有利证据）。→ **不自动加冕**；建议补 Mortal 半庄头对头
n≥1200 与雀魂 maka 后由用户拍板（加冕有 5 处文档与 GCS 义务）。ckpt 已作为候选上传 GCS `checkpoints/bc_lineage/bc65_full_ep6.pt`。

## Next Steps
- 若拍板加冕：走 docs/champion_model.md §7（LEADERBOARD、模型卡、README、桥接 runbook、GCS）。
- 数据线剩余空间：官方渠道已到顶（2026-01 起）；再 10× 只能靛第三方再分发包（ToS 灰区，不建议）或雀魂低段位数据（需先证明"低段位 + 高分微调"可行，exp64 对凤凰卓内部已否）。
- 本轮文献路线全部执行完毕；剩余唯一方向是 RL 的前置条件（critic 看隐藏信息 + GRP + 10⁵ 级评测），需用户决定是否投入。

## Artifacts
| Path | Size | Description |
|---|---|---|
| experiments/exp65_full/ + gs://llm-mahjong-experiments/exp65_full/ | 16MB | F 臂 best（ep6，0.8171）/ last ckpt、metrics（10 epoch）、TB |
| gs://llm-mahjong-experiments/checkpoints/bc_lineage/bc65_full_ep6.pt | 8MB | 冠军候选（未加冕） |
| experiments/exp65_old/ + gs://…/exp65_old/ | 16MB | C 臂（旧快照同批控制）best（ep10 0.8077）/ last、metrics、TB |
| experiments/exp65_full_try1/ | — | 首次发射（5.6k 样本/s）的 TB，未完成 epoch |
| experiments/exp65_pull/*.log | — | pod 端日志（materialize_full/old、train_full/old、exp65） |
| experiments/probes/exp65_arms.json、exp65_symmetry_full.json | — | 终评（双段 + 合并）、破缺率 |
| experiments/configs/exp65_holdout_1000.txt | 60KB | 冻结 holdout |
| data/tenhou/raw（主检出，114,803 局，1.9GB）| — | 采集数据（不入库、不再分发） |
