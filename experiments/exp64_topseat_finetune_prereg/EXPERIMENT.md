# exp64 全量预训练 → 高分席位微调：BC 能否超过"示范者平均"

- **Date**: 2026-09-04  **Status**: done（判平）
- **Git**: 见 Progress；RunPod Secure L40S（$1.09/h），数据 = exp62 同一份 20,526 局快照（tenhou_raw.tar.zst），pod 零凭证
- **Env**: `train_human_bc.py --init --seat_min_rate --max_updates`（本实验新增），流式回放（无需物化）

## Purpose & Hypothesis
文献（docs/literature_review_2026-09.md §3）：BC 落在示范者分布的"平滑平均"之下；AlphaStar Unplugged 的同数据配方
"全量预训练 → 高 MMR 子集微调" +105 Elo（84%→89% 胜 very_hard），只过滤不预训练反而 84%→65%。
我们的凤凰卓数据里各席 R 值分布：中位 2160，**R≥2200 占 25.4% 席位**，R≥2150 占 56%。假设：以 bc49 为起点、只在 R≥2200 席位的
决策上微调，T=0 配对头对头对"同步数全席位微调"的控制臂 ≥ +1%，holdout 精度不降。反假设：凤凰卓内部技能差太小
（ILEED 同质 split 零增益），微调只带来 exp62 量级的噪声。

## Method（同批控制臂 + 双种子，按 exp62 教训）
四个 run 同一 pod 并行，全部 `--init bc49.pt`，lr 1e-4 const（原配方 1/3），batch 1024，bf16，early-stop 关（固定步数）：
- **T1/T2（实验）**：`--seat_min_rate 2200`（≈ 3.0M 决策/epoch），3 epoch ≈ 9k 更新，seed 1 / 2；
- **C1/C2（控制）**：全席位，`--max_updates` = T 臂实际更新数（步数匹配），seed 1 / 2。
每 epoch 记录全 holdout acc（1000 局全席位）。终评（工作站）：每个 T 对每个 C 的 n=4000 配对牌山 T=0（4 个交叉对），
汇总为 T 均值 vs C 均值；T/C 各对 bc49 原版 n=4000；A/A 对照。若 T−C 合并 ≥ +1.5σ 再补 n=16000。

## Config
- 数据快照 = exp62 同一 tarball；holdout 同 hash 规则（1000 局，全席位）；R 阈值 2200（top quartile）。
- 预算：物化省掉，流式 13 核 4 个 run 并行 ≈ 2 h ≈ **$2.5，上限 $5**。

## Success Criteria（预注册）
1. **主判据**：T（2 种子均值）vs C（2 种子均值），配对牌山 T=0，合并 n≥16000 对 share **≥ 0.51 且两个种子同号** → 高分席位微调成立，
   进入"数据 10× + 高分微调"组合（exp65 的默认收尾步骤）；0.50–0.51 → 噪声地板内，判平；<0.495 → 有害。
2. 精度：T 的全 holdout acc 不低于 C −0.3pp（微调到子集不能毁全局模仿）。
3. 诚实条款：T 与 C 步数匹配、同 init、同 lr；不做事后选 epoch（取最后一个 epoch 的权重）。

## Progress
- [09-04 01:07] git fb113b1。本机冒烟（300 局、25 更新）：热启动 0 键跳过、席位过滤 320/1200 单元、预算停止生效。
  pod `m9chauvfbxturo` L40S Secure US-TX-4（$1.09/h），bootstrap 2 分钟。**假发射**：repo.tar 不含 gitignore 的
  `experiments/_anchors_epoch6/bc49.pt` → 四臂 3 秒内 FileNotFoundError，心跳把 TRAIN_DONE 误判为完成。补传权重后
  01:10 重发：T 臂席位单元 18,792 / 73,832（25.5%），四臂并行各 3 worker（cgroup 13.6 核）。教训入 SKILLS：pod 归档要显式带锚点。
- [09-04 01:45] C1/C2 完成（9000 更新 = 0.75 个全席位 epoch，33 分钟，4.8k 行/s/臂）：acc **0.8088 / 0.8091**。
  T 臂发现脚本只存 best 不存 last（步数匹配要比最后权重）→ 停掉 T1/T2（保留其 ep0 权重为 T*_ep0），脚本加 last.pt 后重发 T1b/T2b
  （各 6 worker）。远程 kill 一次自匹配杀掉了 ssh 会话（pgrep 命中远端 sh -c 命令行，SKILLS 老坑）。
- [09-04 02:40] T1b/T2b 完成（9000 更新 ≈ 3 个高分席位 epoch）：逐 epoch acc 0.8086 → 0.8071 → 0.8064 → **0.8061**（T1b）、
  0.8086 → 0.8076 → 0.8065 → **0.8065**（T2b）——在子集上继续训练，全 holdout 精度单调下滑。pod terminate（204），≈2.7h ≈ **$3.0**。
- [09-04 03:00] 工作站终评（含 16k 补充）。

## Results

| 判据 | 目标 | 实测 | 判定 |
|---|---|---|---|
| 1 主判据 T vs C（T 取 last，C 取 9000 更新权重），配对牌山 T=0 | ≥0.51 且双种子同号 | T1b/C1 0.5045、T1b/C2 0.4979、T2b/C1 0.4984、T2b/C2 0.4940（各 n=4000）→ **合并 16k 对 0.4987±0.0040** | ❌ 判平（略负） |
| 2 精度 T − C（全 holdout） | ≥ −0.3pp | 0.8061/0.8065 − 0.8088/0.8091 = **−0.27pp** | ⚠️ 擦边；趋势为负（逐 epoch 下滑） |
| T@ep0（3000 更新）vs C | — | 0.5000 / 0.4961（n=4000） | 平 |
| T1b vs bc49 | — | n=4000 0.5134、n=16000 0.5050 → 合并 0.5067±0.0035 | +1.9σ（噪声地板内） |
| C vs bc49（探索性："低 lr 续训 9000 步"） | — | C1：4k 0.5052 / 16k 0.5013 → 0.5021；C2：4k 0.5184 / 16k 0.5000 → **0.5037±0.0035**；C2 vs C1 16k 0.5000 | 平；4k（seed 67M 段）偏高是牌山段效应 |
| A/A | — | 0.500 | ✅ |

## Conclusion
**高分席位微调判平（略负）**：以 R≥2200（前四分之一席位）决策微调 9000 步，头对头对同步数全席位控制臂 0.4987±0.004，
全 holdout 精度随微调 epoch 单调下滑（0.8086→0.8061），说明模型在"更像顶尖 25%"和"更像全体凤凰卓"之间只是换了一个
略差的拟合点——凤凰卓内部技能差（R 2119–2239 四分位距）太小，正如 ILEED 的同质 split 零增益与 AlphaStar Unplugged 的前提
（MMR 跨度 3500→6000+）所预示。AlphaStar 的 +105 Elo 需要示范者之间有真实的技能梯度，凤凰卓没有。
探索性发现：bc49 之后再低 lr 续训 9000 步（C 臂）精度 +0.3pp（0.8059→0.8090）但头对头 0.5021/0.5037，不兑现；
seed 67M 牌山段对所有"重跑/续训"模型系统偏高 +0.5–1.8%（exp62 亦然），**多段种子已是必需**（SKILLS 既有规则的再确认）。

## Next Steps
- 路线 ③ 关闭（凤凰卓无技能梯度）。若未来引入雀魂/天凤特上等低段位数据做 5–10× 扩容，"全量预训练 → 凤凰席位微调"才有意义。
- 数据 10×（exp65）成为唯一在跑的主线；采集完成后同批控制臂 + 双牌山段评测。

## Artifacts
| Path | Size | Description |
|---|---|---|
| experiments/exp64_{C1,C2}/ + gs://llm-mahjong-experiments/exp64_{C1,C2}/ | 8MB ×2 | 控制臂（全席位，9000 更新）best=last ckpt、metrics、TB |
| experiments/exp64_{T1b,T2b}/ + gs://… | 16MB ×2 | 实验臂 best（ep0）与 last（ep3）ckpt、metrics（4 epoch）、TB |
| experiments/exp64_{T1_ep0,T2_ep0}/ | 8MB ×2 | 首次发射 T 臂的 ep0 权重（3000 更新） |
| experiments/exp64_pull/*.log | — | pod 端日志 |
| experiments/probes/exp64_arms_n4000.json, exp64_arms_n16000.json | 1KB | 终评 |
