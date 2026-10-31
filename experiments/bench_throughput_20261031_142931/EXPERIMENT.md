# bench_throughput — 纯血 PPO 训练系统吞吐剖析（本机 4080，$0）

- **Date**: 2026-10-31 14:29  **Status**: done 16:05（$0）
- **Git**: 本 worktree e77c001；训练器/模型/Rust 引擎取 `claude/continue-task-3da2e7` **02d0cbe**（scratchpad detached worktree `ct_code`，用完移除）
- **Env**: 本地 RTX 4080 16GB，24 核（雀魂服务在跑，workers 12），conda `rlhf_mahjong`
- **提议人**: 用户 2026-10-31："有可能对训练系统做更加激进的优化加速吗？" + "有验证过 transformer 等架构在上亿局训练里的表现吗"

## Purpose & Hypothesis
回答"还能快多少"需要先知道 cnn_l 端到端每轮时间花在哪。exp113 只给了 384×12 的 rollout/打包/更新三段，cnn_l 没有分段。
本基准把 cnn_l 每轮拆成：rollout 内 {Rust observe、H2D、GPU 前向+采样、D2H+tolist、Rust step、drain} 与 打包 / 更新，
并量 bc49 的架构（ConvFormer）在同训练器下的每局成本，供"在上亿局刻度验证注意力架构"的预算用。
无假设检验；不训练出任何保留产物。

## Method
1. 训练器原样：从 `experiments/exp113_C/games_384000000.pt` 续 6 迭代（exp113 C 配方逐字，`--profile_pack`），取第 3–6 轮中位数；再加 `--compile` 一次。
2. rollout 分段：scratchpad 脚本复制 `collect_rust` 的普通路径（`net.act`），每段 cuda 同步计时，K=2048、单迭代 2048 局。
3. ConvFormer：`convformer_m_v3r` 从零（随机初始化）同配方 4 迭代，量每局成本（强度无意义）。
产物写 scratchpad；本目录只留日志与汇总。

## Success Criteria
无（基准）。输出：各段秒数/占比表 + 每美元局数投影。

## Progress
- [14:29] 建目录、detached worktree。
- [14:35] A：cnn_l 原配方 6 迭代 rc=0。
- [14:50] rollout 分段（cnn_l / cnn_m / ConvFormer 原版）。用户追加："检查之前的设计，挑最 promising 的 transformer 架构做上亿局训练，先优化再估价" ⇒ 加 ConvFormer 优化。
- [15:10] `fast_convformer.py`（批量花色卷积 + SDPA 加性偏置 + 可选 compile）；等价性 fp64 logit 7.8e-16、fp32 3e-7。
- [15:30] 训练器补丁（`--fast_trunk/--compile` 对 ConvFormer 走新主干；`TilesTransformer.act` 补 `check` 参数，否则 Rust rollout 直接 TypeError）。
- [15:45] C（fast+compile）训练器内首个 rollout 触发 multinomial NaN 断言；独立探针（批量 1–16384、真实 VecEnv 循环、TF32）均无法复现。compile 版 bf16 logit 与原版差 0.16–0.24（无 compile 版 0.006），数值上可疑 ⇒ **compile 判未解决、不采用**；E（fast 无 compile）rc=0。
- [16:30] 用户问"compile 版怎么了" ⇒ 根因排查（`compile_probes/`）：
  ①训练器内加 FC_DEBUG 逐调用对照 eager，**复现**：第 1 次调用（B=2048）编译版与 eager 隐藏层差已 0.37，第 3 次 2.6，第 90 次（B=925）42/925 行非有限，eager 同输入全有限。
  ②把那批行导出单测：编译版 **fp32 精确**（1.7e-6），**bf16 autocast 下**错 0.72–2.4（eager bf16 0.02）；手写注意力替代 SDPA 照样错（2.4）⇒ 不是 SDPA。
  ③逐算子（global_proj / LN / qkv / FFN / 残差加、注意力核、单个 block、stem+1 block）编译与 eager 完全一致；**stem + ≥2 个 block 放进同一张图**误差跳到 0.53；
  只编译 6 个 block（fp32 输入）精确。误差随批量大小变（B=925 0.72、B=4096 0.02）——与"只在某些批次出 NaN"吻合。判读：torch 2.12/cu130 inductor 在
  autocast 混合精度 stem（bf16 conv/linear 输出与 fp32 嵌入拼接）后接多 block 的融合/布局选择上的数值缺陷；模型与输入无问题（标量最大 750 但隐藏层只到 ±47，非溢出）。
  ④**修法：stem 走 eager，只编译 6 个 block + 末层 LN**。46 个批量（1–8192）无 NaN、最大误差 0.028（eager bf16 0.028）；梯度相对误差 1.02%（eager bf16 0.95%）；
  训练器内 FC_DEBUG 3 迭代 500+ 次调用（含更新）全有限、对 eager 最大 0.024。速度几乎不损：微基准 216k 行/s、55.9k 样本/s（全图编译 212k/56.1k）。
- [16:50] F（fast + 修好的 compile）端到端 rc=0：rollout 2.2 / 打包 1.13 / 更新 3.3 = 6.68 s / 2048 局，307 局/s（随机初始化），比 fast 无 compile（258）快 19%。

## Results
**1. cnn_l 每 2048 局（本机 4080，迭代 3–6 中位数，训练好的策略 ≈12.3 万样本/迭代）**
| 配置 | rollout | 打包（其中价值前向） | 更新 | 合计 | 局/s |
|---|---|---|---|---|---|
| fast（exp113 C 配方原样） | 1.1 s | 0.44（0.30） | 0.8 | 2.41 | 850 |
| fast + compile | 1.0 | 0.32（0.18） | 0.5 | 1.88 | 1089 |

rollout 内部（cnn_l，0.94 s）：Rust step 0.38 / GPU 前向 0.27 / drain 0.15 / observe 0.06 / H2D 0.06 / D2H 0.01。三段严格串行；CPU（Rust）约 0.66 s 与 GPU 约 1.0 s 互不重叠。

**2. 架构微基准（随机权重，bf16）**
| 架构 | 推理 行/s（B=1221） | 训练步 样本/s（B=4096） |
|---|---|---|
| cnn_l fast + compile | 1,101k | 250k |
| cnn_l fast | 775k | 162k |
| ConvFormer 原版 | 142k | 36.4k |
| ConvFormer fast | 186k | 44.4k |
| ConvFormer fast + compile（全图，**数值错误，作废**） | 212k | 56.1k |
| ConvFormer fast + compile（只编译 block，采用） | 216k | 55.9k |

**3. ConvFormer 训练器端到端（随机初始化，18.3 万样本/迭代，比训练好的策略多 49%）**
| 配置 | rollout | 打包（价值前向） | 更新 | 合计 | 局/s |
|---|---|---|---|---|---|
| 原版 | 2.7 | 1.86（1.68） | 5.1 | 9.67 | 212 |
| fast | 2.3 | 1.42（1.22） | 4.2 | 7.94 | 258 |
| fast + compile（只编译 block） | 2.2 | 1.13 | 3.3 | 6.68 | 307 |

按训练好策略的样本量（×0.67 于 GPU 段，CPU 段 0.66 s 不变）折算：fast ≈ **5.3 s / 2048 局 ≈ 386 局/s（本机 4080）**，约为 cnn_l 的 0.45×；
fast + compile ≈ **4.5 s ≈ 458 局/s**（实测，替代下表"compile 修好"一行的 ×0.8 假设：GPU 段实测 ×0.82）。

**4. 余量估算（ConvFormer，本机 4080，每 2048 局）**
| 叠加 | 合计 | 局/s | 是否保持算法不变 |
|---|---|---|---|
| fast（已测） | 5.3 s | 386 | ✅ |
| + 价值在 rollout 前向里顺带算（去掉打包里的整批价值前向） | 4.5 | 455 | ✅ |
| + compile 修好（GPU 段 ×0.8） | 3.7 | 555 | ✅ |
| + rollout 与更新异步重叠（1 迭代策略滞后） | 3.1 | 670 | ❌ 改变训练动态，与现有 cnn_l 曲线不可直接比 |

## Conclusion
- 纯血 cnn_l 训练器还有约 1.3×（compile）+ 价值前向复用 + 异步重叠的余量，合计约 2×，之后瓶颈转到 Rust 引擎 CPU。
- ConvFormer 的成本由 GEMM 主导（更新占 55–60%），优化后每局成本约为 cnn_l 的 2–2.5 倍；最大的单项是更新步，换算力型卡（4090/L40S/H100）比换 CPU 更有效。
- 估价与实验设计见 `experiments/exp124_convformer_scale_prereg/EXPERIMENT.md`。

## Artifacts
| Path | Description |
|---|---|
| `fast_convformer.py` | 快速 ConvFormer 主干（state_dict 不变；compile 只编译 block 栈；`FC_DEBUG=1` 逐调用对照 eager 的调试挂钩） |
| `compile_probes/` | compile 数值缺陷的定位脚本（fc_diag / fc_variants / fc_ops / fc_ops2 / fc_fix / fc_fix2）+ 训练器内复现日志（全图 NaN / 只编 block 正常） |
| `fast_convformer_trainer.patch` | 对 02d0cbe 的训练器 / arch_zoo 补丁（不含新文件 `fast_convformer.py`，它放 `src/agents/dnn/`） |
| `bench_arch.py`, `bench_rollout.py` | 等价性 + 微基准 / rollout 分段计时脚本 |
| `run_*.log`, `rollout_segments.log` | 各配置训练器日志与分段结果（train_log.json 在 scratchpad，未保留） |
