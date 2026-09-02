# exp57 — RunPod 性价比裁决（扩展曲线 + 实查目录 + 在 pod 实测每核速率）

- **Date**: 2026-08-30 17:45  **Status**: done（19:15 收官，探针 pod 已 terminate）
- **Git**: 3964de8（工作树干净；scaling_bench.py 在会话 scratchpad，正文已抄录本目录 `scaling_bench.py`）
- **Env**: 本机 RTX 4080 16GB + 24 核桌面 CPU；rlhf_mahjong conda env；RunPod 目录 = 2026-08-30 实查

## Purpose & Hypothesis
`docs/runpod_cost_and_ops.md` 的整张性价比表压在「5.8 局/s/核、线性到 48 核」的未验证外推上。
本实验分三步裁决「RunPod 上是否存在比 g4-standard-48 flex（$2.25/h，48 vCPU，实测 279 局/s）更划算的训练选项」：

1. **H1（扩展形状）**：rollout 吞吐随 worker 数的曲线在 24 核内近线性（每核速率衰减 <20%）。
   若在 ~16 核就饱和 → 核数不再是采购的主变量，整张表按曲线重算。
2. **H2（目录）**：以 GraphQL `lowestPrice.minVcpu` 实查为准重建候选表（文档旧表已知至少
   3090 community 的「16 核」与实查 8 核不符）。
3. **H3（每核速率）**：候选宿主的实测每核速率 ≥ g4 的 5.8 局/s/核（文档 §1.2 预注册的 ~$0.25 探针）。

## Method
- **本地扩展基准**（上会话排队，chain3 等机器空闲自动开跑）：bc49 ckpt，deal/hanchan 两模式 ×
  workers ∈ {1,2,4,8,12,16,20,24}，gpu_infer=on，产物 `experiments/probes/exp57_rollout_scaling.json`
  （含每点 GPU 利用率以定瓶颈）。
- **目录实查**：REST v2 list-gpu-types（价格/库存/上限）+ GraphQL lowestPrice（每卡保证 vCPU/RAM，
  分 secure/community、gpuCount 1–4）。快照存本目录 `catalog_snapshot/`。
- **RunPod 探针**：对目录+曲线联合裁决出的最优候选开一台 community/secure pod，跑同一个
  `scaling_bench.py`（git 同 SHA），得实测每核速率与实配核数；用完 terminate（不 stop）。
- 基线对照：g4-standard-48 flex 实测 279 局/s（docs/runpod_cost_and_ops.md §1，cnn_m_r K=32）。
  注意基准模型不同（本实验用 bc49 conv），跨文档比较以「同脚本同 ckpt 的本机 24 核点」为换算桥。

## Config
- scaling_bench.py：GAMES_PER_WORKER_TOTAL=24，games_per_worker=16（向量化 K=16），
  infer_max_batch=128，wait_ms=0，temperature=1.0，seed 70000000。
- 探针 pod 预算：**≤ $0.60**（1 台 ≤1h，候选价 $0.22–0.50/h；超出即中止并记录）。

## Success Criteria
- **裁决产出**：一张以实测曲线（非线性外推）+ 实查价格计算的 局/s/$ 表，明确「换/不换 g4」。
- H3 探针判「候选胜」需：实测 局/s/$ ≥ g4 基线（124 局/s/$，cnn_m_r 口径需按 bc49 换算桥折算）
  的 **1.5×**（低于 1.5× 不值得引入 community 可靠性风险与迁移成本）。
- 副产物（§3.2 误差棒）：确定性回放 12 对逐字节一致 ⇒ 判「二项 SE 正确，2.2σ = 牌山运气 +
  跨锚共享牌山相关性」；不一致 ⇒ 立项查非确定性来源。

## Progress
- [17:30] chain3 已排队（等 spec_probe + mortal 链结束后自动跑本地扩展基准）。
- [17:39] 目录实查完成：REST 46 卡 + GraphQL vCPU 表。要点：3090 community 最低价宿主
  仅 8 vCPU/卡（文档写 16，前提 3 被否）；3090 secure 32 vCPU/卡（×2=$1.00/64 核）；
  3090 Ti community 28 核/$0.27；A4000 community 16 核/$0.17；A5000 断货。
- [17:42] §3.2 确定性回放发射（12 对，seed0=51000000，对照探针批存档逐字节比对）。
- [17:44] §3.2 中间结论:两批温度均 0、无 fallback、无平局、代码/ckpt/池无变动;跨锚共享
  牌山相关 r=+0.2~0.3（bc49↔exp46I，策略相近所致），bc51 与两者不相关 ⇒ 「两锚同向偏」
  非独立复现。
- [18:30] **扩展基准 v1 作废、v2 重发**：v1（scaling_bench.py）每点只跑 3.4–5s，
  ~3.4s 固定启动（spawn 服务器 + CUDA graph + 进程池）淹没信号，表面「每核 7.1→4.8」
  全是摊销假象；且 hanchan 模式崩（向量化 hanchan 生成器把 b 侧席位当第二个模型 id，
  单模型服务器 `net[mid]` 越界——待修，不阻塞本裁决）。v2（scaling_bench2.py，已抄本目录）
  每点跑短（24 局/worker）+ 长（168 局/worker）两遍取**边际吞吐**，差分精确消掉固定开销；
  只跑 deal 模式（与历史跨机数字同单位）。
- [17:50] **§3.2 结案**：12 对确定性回放与探针批存档逐字节一致（24 场 uma/顺位/局数全同，
  不同 GPU 负载下重放）。T=0 对局 = 种子的确定函数 ⇒ 二项 SE 数学正确（无平局时还严格等于
  经验方差）。判决：**2.2σ = 牌山运气 + 选择效应**，n=4000 结论不受影响，n≤400 读数按
  标称 SE 使用。新发现：五锚 Mortal 评分共用 seed0=49900000 同段牌山 ⇒ 相似策略锚
  （bc49↔exp46I r≈+0.2~0.3）读数正相关，池拟合 SE 偏乐观、三角闭合残差不可当独立证据；
  **以后每锚用不重叠种子段**。回放件：determinism_replay.json（已拷本目录）。

- [18:55 事故] **rl-vs-bc worktree 在探针发射窗口被移除**（非本会话操作；`git worktree list`
  已无该项）。分支与全部记录 commit（d3ae85b/a859dda）在共享对象库中**完好**，已 ff 合并到
  `claude/runpod-training-cost-optimization-626ab2` 继续。**丢失的 gitignored 工件**：
  半庄 T=0 锚池文件 + history.jsonl + `mortal_shards/` 逐场原始数据 + exp56 运行目录日志。
  **可完全再生**：T=0 对局是种子的确定函数（今日回放已证逐字节一致），种子段已写进记录
  （探针批 51000000×400、链批 49900000×200/锚）。锚 ckpt 本体在主检出 `_anchors_epoch6/` 无恙；
  探针 stdout 日志已从旧会话 scratchpad 抢救进 `experiments/probes/*.log`；
  v2 曲线 JSON 已从日志重建。教训follow-up：worktree 内的 durable 工件不 durable——
  评测原始数据应落主检出或 GCS。
- [08-31 01:30 补记] **worktree 蒸发的账本再生完成并闭环**：五锚链批 + 池拟合逐字节复现
  （ELO 1465.3±11.9，五残差全同），两个探针批（bc49/exp46I × 400 对）七组聚合校验
  **CLOSURE PASS（7/7 完全一致）**。产物已落主检出 `experiments/elo_league/hanchan/`
  并镜像 `gs://llm-mahjong-experiments/elo_league/hanchan/`。确定性再生从验证走到投产。
- [19:05] **H3 探针在跑**：pod `g1v1470djni9rd`（3090 Ti community，$0.27/h，EPYC 7663 宿主，
  cgroup 保证 23 核 + burst）。坑重踩：Ubuntu24.04 pip 要 `--break-system-packages`（文档坑 3）；
  ZOO 键名是 `handset_xl_cnn_m_r` 不是 `handset_xl`。
  **头条已出：cnn_m_r K=32 @23 workers = 214.5 局/s；@46 workers = 232.9 局/s**
  （对照 g4-48 = 279、本机 24 核 = 204、L4 g2-32 = 104）。

## Results

**H1 扩展曲线（v2 边际吞吐法，bc49，deal 模式，本机 24 核 + 4080，2026-08-30 18:40）**：

| workers | 边际 局/s | 每核 | GPU% |
|---|---|---|---|
| 1 | 28.9 | 28.9 | 6 |
| 2 | 52.1 | 26.0 | 11 |
| 4 | 106.8 | 26.7 | 23 |
| 8 | 185.1 | 23.1 | 31 |
| 12 | 220.0 | 18.3 | 38 |
| 16 | 273.0 | 17.1 | 39 |
| 20 | 284.0 | 14.2 | 36 |
| 24 | 294.2 | 12.3 | 34 |

**H1 判决：非线性，预注册的「衰减 <20%」不成立**。1→4 近线性（~27/核），之后持续衰减：
24 worker 的每核效率只剩单 worker 的 **42%**；w16→24 每加一核边际只有 ~2.6 局/s。
单 run 在本栈（单推理服务器）上逼近 **~300 局/s 的平台**，且 GPU 峰值只有 39%——
瓶颈是**往返延迟/争用结构**，与 SKILLS 2026-08-25 g4 上的观察（两侧都不饱和、加 worker 无效）
同象。**推论：①「48 核才够」前提倒塌，单 run 的核甜点 ≈16；②大于 16 核的整机在为
用不上的核付钱；③堆吞吐的正确形状 = 多个便宜 pod 各跑一臂，不是一台大机**。

**v1 失败教训（方法论）**：秒级基准里 ~3.4s 固定启动能把整条曲线变成摊销假象——
v1 给出「每核 7」，v2 差分后同点位实际 23-29。**任何 <30s 的吞吐点必须用双长度差分**。

**H2 成本表重算（曲线代入 + 实查价，每档取本曲线同核数的吞吐估计）**：

| 选项 | $/h | 保证核 | 估 局/s | 局/s 每 $ | 相对 g4 |
|---|---|---|---|---|---|
| g4-48 flex（基线，实测 cnn_m_r 口径） | 2.25 | 48 | 279 实测 | 124 | 1× |
| **A4000 community** | **0.17** | 16 | ~270 | **~1590** | ~13× |
| **3090 Ti community** | **0.27** | 28 | ~290 | **~1070** | ~9× |
| 3090 community | 0.22 | 8 | ~185 | ~840 | ~7× |
| 3090 secure（EU-CZ-1） | 0.50 | 32 | ~295 | ~590 | ~5× |

（估值假设 community 核 ≈ 本机桌面核；就算慢 2× 也全线 ≥3× 于 g4。**该假设正是 H3 探针要验证的**。）

**H3 探针实测（3090 Ti community，$0.27/h，EPYC 7663 宿主，cgroup 23 核保证，
2026-08-30 19:10，pod g1v1470djni9rd，总花费 ~$0.12）**：

| 项目 | 3090Ti com 23核 | g4-48 flex | 本机24核+4080 | g2-32 L4 |
|---|---|---|---|---|
| cnn_m_r K=32（局/s） | **214.5**（46w 超订: 232.9） | 279 | 204 | 104 |
| handset_xl（局/s） | **62.4** | 160 | 97 | 36 |
| $/h | **0.27** | 2.25 | 0 | 2.6 |
| cnn 局/s 每 $ | **794**（超订 862） | 124 | ∞ | 40 |
| handset 局/s 每 $ | **231** | 71 | ∞ | 14 |

- 每核速率 9.3 局/s/核（cnn_m_r 口径）= **g4 服务器核的 1.6×**——community EPYC 核不慢反快。
- worker 曲线（W×48 局的小样本点，含启动偏置）：2w 22.6 / 4w 44.4 / 8w 85.5 / 16w 138.9 /
  23w 178.8——到 23 核仍在爬，46w 超订阅再 +8.6%，与本机曲线同形状。
- 大模型（handset_xl 20M）也 3.3× 于 g4 每美元；每核 2.7 vs g4 3.3——GPU 未封顶（3090Ti ~1TB/s）。

## Conclusion

**三个假设全部裁决，预注册判据（≥1.5× g4 每美元）以 6.4× 大幅达成：**

1. **H1 否**：吞吐非线性，24 核处每核效率剩 42%，单 run 平台 ~300 局/s（GPU 39% 未饱和，
   往返结构瓶颈）。「48 核整机」为用不上的核付钱；单 run 核甜点 ≈16-24。
2. **H2 表已重算**：旧表两处关键错误（3090 com 实为 8 核/卡；线性外推高估大核数机型）。
3. **H3 过**：3090 Ti community 实测 **cnn_m_r 794 局/s/$（g4 的 6.4×）**，每核速率反超 g4 核 1.6×。

**裁决：RunPod 上存在明确优于 g4-standard-48 flex 的训练选项。**
推荐形态：**单卡 3090 Ti community（23-28 核，$0.27/h）为默认训练机**——
单 run 墙钟 = g4 的 77%（214 vs 279），单位钱吞吐 6.4×；1M 局 rollout 口径 ~$0.50 vs g4 ~$3。
多臂实验开多个 pod（吞吐随 $ 线性堆，不受单 run 平台限制）。
备选：A4000 com（$0.17，16 核）做小实验、3090 secure（$0.50，32 核，EU-CZ-1）做长跑可靠性要求高的 run。

**未验证残留（迁移前必须补）**：①community 长跑可靠性（26h 不断）与 `--resume` 演练；
②训练全链路（PPO 更新步 + ckpt→GCS）在 3090Ti 上的实测占比；③库存波动（3090Ti 仅 Low）。
这些属迁移立项（exp58?），不改变本裁决。

- **Status 更新**: done（2026-08-30 19:15；探针已 terminate，账户实花 ~$0.12）

## Next Steps
- **H3 探针（等用户批准，≤$0.60）**：3090 Ti community（$0.27/h）跑 `pod_bench.sh`
  （bench_rollout_infer 旧协议：cnn_m_r/handset_xl K=32 对照 g4 的 279/160 + 本宿主 worker 曲线）；
  候选不可得退 A4000 com（$0.17）或 3090 secure（$0.50）。**用完 terminate 不 stop**。
- hanchan 向量化评测的单模型 `net[mid]` 越界待修（评测基建债，不阻塞成本裁决）。
- 若探针达标（≥1.5× g4 的 局/s/$，预注册判据），迁移问题再立项：community 可靠性
  （--resume 演练）、ckpt→GCS 通道、多 pod 舰队发射脚本。

## Artifacts
| Path | Size | Description |
|---|---|---|
| experiments/probes/exp57_rollout_scaling.json | 2KB | 本地 v2 曲线（边际吞吐法，从日志重建） |
| exp57_runpod_cost_prereg/scaling_bench2.{py,log} | 9KB | v2 基准脚本与原始日志 |
| exp57_runpod_cost_prereg/scaling_bench_v1_failed.log | 5KB | v1 作废曲线（启动摊销教材） |
| exp57_runpod_cost_prereg/pod_bench_3090ti.log | 2KB | H3 探针原始输出（3090Ti community） |
| exp57_runpod_cost_prereg/pod_bench.sh | 2KB | pod 侧基准脚本（含自停保险） |
| exp57_runpod_cost_prereg/catalog_snapshot/ | 34KB | 2026-08-30 GraphQL 目录快照（单卡+多卡 vCPU） |
| exp57_runpod_cost_prereg/determinism_replay.json | 5KB | §3.2 确定性回放件 |
| experiments/probes/exp56_{I_specialization,baseline_h2h_T0}.log | 8KB | 从旧会话 scratchpad 抢救的探针日志 |
