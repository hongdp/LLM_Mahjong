# exp124 — ConvFormer（注意力主干）纯血自对弈上亿局：结构是不是纯血平台的瓶颈

- **Date**: 2026-10-31 预注册；同日改为 46 槽臂并批准  **Status**: approved → Stage 0 发射中
- **Git**: 分支 `claude/exp124-convformer-m46` **badaad3**（= 02d0cbe + 46 槽 Rust rollout + 快速主干/只编译 block 栈 + rollout 价值复用 + 测试）
- **提议人**: 用户 2026-10-31："有验证过之前的 transformer 等架构在上亿局的训练里的表现吗" → "检查之前的设计，挑一个最 promising 的 transformer 架构进行上亿局训练，先优化估计价格"
  → "所以我们一直以来测的模型都和 bc49 不同 backbone 吗" → **"用 46 槽，做准备，发射前再完整检查一遍有没有任何后面修复的 perception 或者 action bug 需要加进去"**（= 批准，臂改为与 bc49 完全同结构）

## 2026-10-31 修订：臂 = `convformer_m_v3r_m46`（bc49 的结构，逐层一致）
- 动作头由 374 改为 Mortal 式 46 槽结构化头（bc49 = `convformer_m_v3r_m46`）。BC 证据（exp47）里动作头主效应 ≈0，用户选择"与 bc49 完全同结构"以便结论直接读作"bc49 结构的纯血模型能否到 bc49"。
- **代价**：与对照 exp101 cnn_l（374 头）的比较从单变量变成"主干 + 动作头"两个变量；另 rollout 用 bf16（exp113 起标准件，exp101 为 fp32 rollout，approxKL 影响 ~1e-4）。结论按"bc49 结构 vs cnn_l 配方"读，不归因到单个因子。
- **工程**（全部新写，发射前完成并测试）：Rust VecEnv 只产 374 原生掩码 ⇒ 新增 `m46_native.py` 在主机侧向量化导出 46 槽掩码 / 立直与杠的两步协议 / 回译原生动作；吃需要的被鸣牌取 v3r 第 13 平面（上一张弃牌）。
  `collect_rust` 46 槽路径：声明行 + 选牌行同观测，选牌行带奖励（与 `selfplay._choose` 的 DnnStep 对一致），回合结束后按 (seed, slot, seat) 键稳定排序整体改写 episodes。

## 发射前完整检查（2026-10-31，用户要求：有没有后来修过的感知 / 动作 bug 要加进来）
| 检查项 | 结果 |
|---|---|
| 所有本地分支、worktree、未提交改动里有没有 02d0cbe 之外的代码修复 | **无**。02d0cbe 是全部代码的最新端；唯一不在其中的 4 个提交是 08-23 的雀魂桥接（mjai_bridge），与训练无关且已被后续版本覆盖；主 checkout 未提交改动只是 3 处文档路径注释 |
| exp101（a89161f）→ 02d0cbe 之间 Rust 引擎 / 编码器的改动 | 全是**新增**：v3rh / v3rg 编码、oracle / ron_cost 接口、DealFact 字段、slot 字段；`win_result` 重构为 `win_result_red(None)` 等价。**v3r 编码与规则零变化** ⇒ 对照 exp101 曲线同引擎同观测 |
| engine_known_issues 里的动作缺口（加杠在 4 副露被禁、完成形拒和转立直不给） | Python 修于 13095b9（纪元 6）；**Rust 端也已修**（`kan_allowed(new_meld)`、立直门 `shanten <= 0`） |
| 09-05 规则覆盖审计 | 引擎零改动；唯一刻意未修的观测细节："红五当宝牌指示牌"显示为普通五（两引擎一致，不影响算分；改则换引擎指纹、需重校） |
| dbb2e50 的 `genbutsu_34` 被鸣牌修复 | 只影响 v3rg / fold 辅助函数，v3r 不用 |
| ConvFormer46 的 `red_idx` buffer 修复（exp46 发射事故） | 已在代码中 |
| 46 槽映射正确性（本次新写） | 对拍 `MortalActionSpace`（bc49 的空间，Python 引擎动作 XML）：随机 + 门清两种策略 **65,295 个实战决策全部一致**（立直 434 含红五 4、和牌 260、杠 359 含选牌、吃低中高、碰）+ 静态罕见例（九种九牌、双杠选牌、花色边缘吃）。对拍中抓到并修掉 1 个自己的 bug：花色边缘吃的键越界串槽 |
| Rust rollout 自洽 | 每行动作在自身 46 槽掩码内；按存下的观测重算 logprob 误差 ≤2.4e-4（fp32）；价值误差 ≤1.2e-5；立直声明行与选牌行一一对应、声明行奖励为 0 |
| 现有测试 | 55 项全过（m46 对拍、动作空间、Mortal 对齐、Rust/Python 编码器 / 牌桌 / rollout / 随机对局对拍、快速主干） |
| compile | 训练器内 FC_DEBUG 逐调用对照 eager：最大 0.024（= eager bf16），无 NaN |
| rollout 价值复用 vs 打包阶段价值前向 | max\|dV\| 0.016（bf16 在 \|V\|≈2–4 处的 1 ulp）、均值 1e-5 ⇒ 等价 |
| 评测链路 | `run_arena_dnn.py`（Python 引擎、支持两步协议）直接加载新检查点对 bc49 完成对局 |

## 为什么是 ConvFormer（台账检索，三处 checkout 已 grep）
| 设计 | 最大训练量 | 结果 |
|---|---|---|
| vit_small（exp10/18） | 700k 局 | 同配置输 cnn 85 Elo（exp18），exp10 的 +1633 未复现 |
| **ConvFormer**（exp19，d160×6 层×5 头，花色局部卷积 stem + 牌距相对偏置） | 700k / 1.2M（exp20） | **+52 vs cnn（z≈2.8），700k 仍在爬升**；exp20 加 GAE 后 1.2M 零净增 |
| HandSet 集合注意力（exp27-B / exp31） | 2M | 不优于 34 轴 CNN（H1 否定） |
| HRF 手牌×牌河交叉注意力（exp30/36/37） | 1.8M | RL 钉在 850–900，"架构上限"；BC 下却是防守之王 |
| 人类 BC 区间（exp45/49） | 11.5 万局牌谱 | **ConvFormer 2M 全面胜 11M SEres**，bc49 本身就是 ConvFormer |

所有注意力架构的 RL 都停在 ≤2M 局（Python 引擎时代）；≥100M 局的纯血训练全部是 CNN（cnn_m / cnn_l / 384×12）。exp113 证明**容量**不是平台瓶颈（5× 打平），
但 exp49 的读法是"**结构**（注意力 / 局部性先验）才是兑现变量"——这条在纯血上亿局刻度上从未检验过。SKILLS 08-23 用户规则：架构结论必须在调过配方、跑到平台后才下。

**先验（必须先写明）**：纯血平台 ≈0.47 的已知成因是行为 / 目标侧（受威胁远手的持续防守在单局目标下近乎不值钱，exp94/114–119），不是表示能力；
exp20 显示 ConvFormer 在 1.2M 局已不再领先。最可能的结果是 ConvFormer 与 cnn_l 同曲线（H0）。本实验的价值在于把"注意力从未在规模上测过"这条悬案关掉，或找到唯一一个非 CNN 的上升斜率。

## Hypotheses
- **H0**：ConvFormer 与 cnn_l（exp101，同配方同局数）在 112M / 192M 里程碑上差 < +1.0 pp。
- **H1**：192M 处 ConvFormer − cnn_l ≥ +1.0 pp（两段 20k 对，SE ≈0.25 pp ⇒ 4σ），且斜率（112M→192M）≥ cnn_l ⇒ 续到 384M，冲 0.48+。

## Method
- **臂**：~~`convformer_m_v3r`（374 头）~~ → **`convformer_m_v3r_m46`**（2.0M 参数，46 槽结构化头，v3r 编码 = 纯血编码 = bc49 同结构）从零；**对照 = exp101 cnn_l_v3r 现有曲线**（32M 0.4305 / 64M 0.4379 / 112M 0.4497 / 160M 0.4597 / 192M 0.4541，
  20k 对终评口径），不重跑对照。
- **配方 = exp101 逐字**：`--engine rust --entropy_coef 0.03 --entropy_schedule 0:0.03`，lr 1e-4（0→64M）→ 3e-5（64M→终点，`--resume` 新后缀 `_p2`），其余同 exp101 config.json（发射前逐项 diff）。
  唯一新增：`--warmup_updates`（零初始化策略头，exp19 必需件；SKILLS：按更新数比例设，取 ≈3000 次更新 ≈ 首 20 万局）。
- **Stage 0（配方健全性，8M 局 × 2 臂）**：lr 1e-4 / 6e-5，8M 处 n=1000 在轨 vs bc49 与熵 / KL 健康；取健康且读数高者进 Stage 1（另一臂丢弃）。对照：exp101 cnn_l 8M 读数。
- **Stage 1**：0→192M。里程碑 32M / 64M / 112M / 160M / 192M 做 T=0 vs bc49 两段 20k 对（本机 4080，$0）。
- **Stage 2（仅 H1）**：192M→384M。
- 速度工程（发射前完成，均为数学等价，不改训练动态）：①快速主干（已完成，等价性 fp64 7.8e-16）；②价值在 rollout 前向里一并算出，去掉打包阶段的整批价值前向；
  ③compile：**已修好（10-31 16:50）**——全图编译在 bf16 autocast 下是 inductor 数值缺陷（stem+≥2 block 同图误差 0.5–0.8、训练器内 NaN），改为 stem eager + 只编译 block 栈，46 个批量与梯度均与 eager bf16 同精度，训练器内 3 迭代无 NaN。**不做**rollout/更新异步重叠（改变策略滞后，与 exp101 曲线不可比）。

## Config（发射命令，逐项对照 exp101_L/config.json）
```
python3 scripts/train_dnn_ppo.py --engine rust --arch convformer_m_v3r_m46 --lr {A: 1e-4 | B: 6e-5} \
  --batch 4096 --ppo_epochs 1 --clip_eps 0.2 --entropy_coef 0.03 --entropy_schedule 0:0.03 --gamma 0.995 --gae_lambda 0.95 \
  --dup_k 8 --games_per_iter 2048 --games_per_worker 1024 --workers 1 --target_kl 0.03 --value_coef 0.5 --adv_clamp 5.0 \
  --amp_update --tf32 1 --shaping --shaping_schedule 0:1.0,400000:0.0 \
  --warmup_updates 3000 --fast_trunk --compile --bf16_rollout --rollout_values --profile_pack \
  --milestones 4M,8M,16M,24M,…,64M --ckpt_every 400 --seed {A: 1241 | B: 1242} --total_games 64000000 --exp_dir /workspace/exp124_{A|B}
```
与 exp101 p1 的差异：`--arch`（cnn_l_v3r → convformer_m_v3r_m46）、`--warmup_updates 3000`（零初始化策略头，exp19 必需件）、`--fast_trunk --compile --bf16_rollout --rollout_values`（数学等价 / 数值噪声级）、seed、exp101 的 `resume_migrate` 是迁机续训的痕迹。其余逐项相同。

## Success Criteria（发射前定）
1. **健康**：32M ≥ 0.42（cnn_l 0.4305 − 1 pp）；熵 ≥ 0.6 nat；KL 常态。破线 ⇒ 排查后重发或停。
2. **早停**：64M ≤ cnn_l − 1.5 pp（≤ 0.423）⇒ 停（结构在拖累），报告。
3. **主判据（192M）**：ConvFormer − cnn_l ≥ +1.0 pp 且 112M→192M 斜率 ≥ cnn_l ⇒ H1，申请 Stage 2；+0.5～+1.0 ⇒ 弱，报用户；< +0.5 ⇒ H0，关闭注意力线。
4. 延长规则（用户 09-12）：192M 处最后两格仍单调上升 ≥ +0.5 pp 时报用户申请追加，不自行越上限。
5. 所有在轨读数只用于健康 / 早停；判决只认 20k 对口径。

## 预算（基于 bench_throughput_20261031_142931 实测 + exp113 机型表 10-16 报价，发射当天重核价）
ConvFormer 每 2048 局（训练好策略的样本量）在本机 4080：已测 fast 5.3 s；已测 fast+③ 4.5 s；+②（价值复用，估）≈3.9 s。GPU 段换算用 exp113 的 GEMM 训练步实测比（4090 1.53×、L40S 1.34×、H100 3.0× 于本机 4080；RTX 4000 Ada 未测，估 0.5×），CPU 段 0.66 s 不变。

| 方案 | 机型 | 局/s | $/百万局 | 192M | 384M | 192M 墙钟 |
|---|---|---|---|---|---|---|
| 只用快速主干（现状） | 4090 Secure $0.89/h | ≈550 | ≈$0.45 | ≈$86 | ≈$173 | ≈97 h |
| + compile（已测，现在就能跑） | 4090 Secure | ≈650 | ≈$0.38 | ≈$73 | ≈$146 | ≈82 h |
| + compile + 价值复用（推荐） | 4090 Secure | ≈750 | ≈$0.33 | ≈$64 | ≈$127 | ≈72 h |
| + compile + 价值复用 | RTX 4000 Ada $0.28/h | ≈290（未测） | ≈$0.27 | ≈$51 | ≈$103 | ≈183 h |
| 参考：cnn_l（exp113 C 实测） | RTX 4000 Ada | 678 | $0.115 | $22 | $44 | 79 h |

- Stage 0：16M 局 ≈ $5–7。机型实测选机（同 exp113 做法）≈ $3。
- **申请额度（推荐方案）**：Stage 0 + Stage 1 ≈ **$70**，上限 **$90**；Stage 2 另批 ≈ $60（上限 $75）。评测全在本机。

**46 槽臂实测（10-31 晚，本机 4080，随机初始化，exp124 配方，迭代 3–7 中位数）**：
| 配置 | rollout | 打包 | 更新 | 合计 / 2048 局 | 局/s |
|---|---|---|---|---|---|
| 374 头 fast + compile（bench） | 2.2 | 1.13 | 3.3 | 6.68 | 307 |
| 46 槽 fast + compile，无价值复用 | 2.9 | 1.12 | 3.4 | 7.42 | 276 |
| **46 槽 fast + compile + 价值复用（向量化改写后，采用）** | **2.5** | **0.18** | **3.4** | **6.08** | **337** |

按训练好策略的样本量（GPU 与主机侧开销 ×0.67，Rust CPU 0.66 s 不变）折算 ≈4.3 s / 2048 局 ≈ **477 局/s（本机）**。机型换算（GEMM 比，未实测，发射后首批迭代即实测）：
| 机型 | 估 局/s | $/百万局 | 192M | 墙钟 |
|---|---|---|---|---|
| **4090 Secure $0.89/h（采用）** | ≈680 | ≈$0.37 | ≈$70 | ≈79 h |
| RTX 4000 Ada $0.28/h | ≈260 | ≈$0.30 | ≈$58 | ≈205 h |
| L40S $1.09/h | ≈610 | ≈$0.50 | ≈$96 | ≈87 h |
| H100 $2.89/h | ≈1100 | ≈$0.73 | ≈$140 | ≈49 h |
- **10-31 实时价目（list-gpu-types）**：4090 Secure $0.89/h（EU-RO-1 / EUR-IS-1/2 库存 LOW）、**4090 Community $0.34/h**。纯血从零不带冠军权重，按安全分层（用户 09-01）允许上 Community ⇒ 首选 Community 4090（192M ≈ **$27**，同吞吐假设），开机核数 < 16 或首批迭代 < 500 局/s 就换机，Community 不可用时退到 Secure（≈$70）。
- **额度不变**：Stage 0（2 臂各到 8M，≈$6）+ Stage 1（8M→192M，≈$64–70）≈ **$76**，上限 **$90**。单卡（未做数据并行：用户未选，且需新代码）。
- **评测口径（本机 $0，`exp124_ops/eval_loop.sh` + `pairdiff_eval.py`）**：判决里程碑 32/64/112/160/192M = T=0 双份 2v2 对 bc49，两段各 1 万牌山，**用 exp101 自己的种子 101112000 / 101112500**（牌山段偏差 0.5–1.8 pp，SKILLS 09-04，换种子就不能和对照曲线直接比）；其余里程碑 2000 牌山（种子 101112000，只作健康 / 早停 / 选臂）。
- **Stage 0 执行方式**：两臂各一台 4090，`--total_games 64000000` 直接起跑，`--milestones` 含 4M/8M；8M 检查点落盘后本机 n=1000（T=0，vs bc49，牌山 67M 段）+ 熵 / KL 健康读数，**选臂后立刻终止另一台**，胜臂原地续到 64M，再 `--resume games_final.pt --lr 3e-5` 续到 192M（同 exp101 两段式）。首批迭代的实测局/s 若 < 500 ⇒ 换机型重估再报。

## Progress
- [2026-10-31 16:10] 预注册草案；吞吐基准与估价完成（见 bench 目录）。等待用户批准额度；批准后先做速度工程②与等价测试，再选机发射。
- [2026-10-31 16:55] ③compile 修好并实测（见 bench 目录 Progress 16:30/16:50）；预算表按实测更新（推荐方案 ≈$0.33/百万局，额度不变）。
- [2026-10-31 晚] 用户："用 46 槽，做准备，发射前再完整检查一遍有没有任何后面修复的 perception 或者 action bug 需要加进去"。
  完成：发射前检查（上表，无需补任何修复）；46 槽 Rust rollout + 价值复用实现并测试，分支 `claude/exp124-convformer-m46` badaad3，代码 worktree `/home/hongdp/Workspace/LLM_Mahjong_exp124`；
  本机冒烟 4 + 7 + 7 迭代 rc=0，337 局/s（随机初始化）；上云脚本 `exp124_ops/`（deploy / pod_boot 含 pod 上 compile 闸门 / 两段式 pod_run / pull_loop 心跳分相报警 / watchdog / eval_loop 用 exp101 种子）。
  **发射被 Claude Code 权限分类器拦截**（create-pod = 真实交易），RunPod 账号零 pod，$0 花费；等用户放行 create-pod 或手动开机。
- [2026-10-31 18:21 PDT = 11-01 01:21 UTC] 用户放行 create-pod。Community 4090 在 CUDA ≥12.8 上**全部缺货**（list-gpu-types availability NONE，去掉核数过滤也开不出），按预案退到 **Secure 4090 $0.89/h**：
  臂 A pod `vnw55zc69161cq`（exp124-A，US-NC-1，16 vCPU / 62 GB，host CUDA 13.2，ssh root@103.196.86.56 -p 11584），lr 1e-4 seed 1241；
  臂 B pod `5we2zlqzwdjy0c`（exp124-B，US-NC-1，16 vCPU / 62 GB，ssh root@103.196.86.110 -p 38593），lr 6e-5 seed 1242。两台合计 $1.78/h；Stage 0 到 8M 后终止败臂。
  TB：`experiments/_tb_qlearning/exp124_{A,B}_convformer46_{0-64M,p2_64-192M}_LIVE` → `experiments/exp124_{A,B}{,_p2}/tensorboard`（TB 用目录 logdir + 120 s reload，自动发现，无需重启）。
  pod 上 compile 闸门两台都过（max_trunk_diff 0.0315 / 0.0305 < 0.06，max|dV| 0.0078 < 0.05）；cgroup v1 配额 13.6 核，训练器自动取 14 线程。随机初始化稳态 ≈385 局/s（本机 4080 同条件 337）。
- [2026-10-31 18:28–18:33] **两臂都在 ~20 迭代处崩溃（我的 46 槽 rollout bug，不是训练数值问题）**：行键 (seed, slot, seat) 在 dup_k=8 下会撞——
  同一个 VecEnv 槽位一轮 rollout 内可能先后两次拿到同一个种子，`_rewrite_m46` 的计数校验按设计直接 raise（−91 / −51 行"多匹配"），**没有静默错位的数据进训练**。
  本机冒烟只跑了 4–7 迭代所以没撞到。修复：行键改为 (该槽已完成局数, slot, seat)，在 drain 时给每局盖章；回归测试强制 4 槽重放同一种子 6 次（旧代码 −10850 行报错、新代码过），
  Rust / m46 / ConvFormer 相关 46 项测试全过；代码分支 `claude/exp124-convformer-m46` **a8fef66**。处置：先停掉会在 TRAIN_FAILED 后触发 pod 自毁的 pull_loop 与 pod_run（两台 pod 保留），
  失败产物归档到 pod `/workspace/aborted_keycollision/` 与本机 `experiments/exp124_aborted_keycollision_20261031/`，两臂用 a8fef66 **从零重新部署**（重跑 compile 闸门）。损失 ≈15 分钟 × 2 台 ≈ $0.5。
- [2026-10-31 18:33 PDT 重新发射 / 18:40 核对] 闸门 max_trunk_diff 0.0289 / 0.0292，max|dV| 0.0078。两臂越过 65 迭代（崩溃点的 3 倍）无异常，在学：
  A（lr 1e-4）iter 65：H 1.99→1.48、win 14.7%、kl 0.0027、402 局/s；B（lr 6e-5）iter 65：H 2.01→1.64、win 9.9%、kl 0.0004、388 局/s。
  pull_loop / watchdog / eval_loop 均在跑，TB 镜像已有事件。**成本复核**：随机初始化 ≈400 局/s ⇒ 按训练好策略样本量折算 ≈560 局/s ⇒ 192M ≈94 h ≈ **$84**，加 B 臂到 8M ≈$5 ⇒ **≈$89，贴着 $90 上限**。
  缓解：胜臂续跑期间若 Community 4090 有货（$0.34/h），用 latest.pt 迁过去（纯血从零允许 Community），可省 ≈$50；首个 4M 里程碑读实测局/s 再定。
- [2026-11-01 01:17 PST] **工作站重启**（内核 6.8.0-146→147 自动更新），本机侧 pull_loop / watchdog / eval_loop / TensorBoard 全部随之死亡；两台 pod 不受影响继续训练。
  后果：预注册的"8M 选臂后立刻终止另一台"没有执行——两臂一直并行跑到 14:31 PST 我回来巡检时的 **33.2M / 33.7M**（iter 16.2k / 16.5k，440 / 447 局/s，H 0.89 / 1.03，win 85% / 82%，kl 0.004，均健康）。
  多烧了约 12 h × $0.89 ≈ **$11**（另一臂本应在 ~02:00 终止）。8M 读数（重启前已出）：A 0.4412±0.0105、B 0.4390±0.0104——平手，当时也选不出来。
  处置：14:32 重启全部本机循环 + TensorBoard（16/24/32M 检查点已镜像），按 32M 判决里程碑（2×10k 牌山）选臂；写好 `exp124_ops/migrate.sh`（Secure→Community 原地续训）。
  **教训**：本机 setsid 循环不抗重启，长跑监控需 systemd/cron 自愈或至少开机自检（进 SKILLS）。

## Results

## Conclusion

## Artifacts
