# perf_rollout_overlap — rollout 分段剖析 + rollout/更新异步重叠原型

- **Date**: 2026-11-03 19:41 PST  **Status**: done（代码在 `claude/perf-rollout-overlap` c241c25，未合入；在跑的 exp124/125 未改动）
- **Git**: 代码分支 `claude/perf-rollout-overlap`（worktree `/home/hongdp/Workspace/LLM_Mahjong_perf`，起点 f11278a = exp124/125 训练代码）
- **Env**: 本机 RTX 4080 16GB，24 线程；riichi_rs 0.1.0（已装 conda env rlhf_mahjong）；torch 同训练镜像版本；仅基准/冒烟（本地 GPU 规则）

## Purpose & Hypothesis
用户 11-03 问 rollout 是否优化过。exp124 在 3090 上每迭代 rollout 3.0 s / 打包 0.2 s / 更新 2.8 s，三段完全串行：rollout 时 GPU 大半空闲，更新时 CPU 全空。
H1：①rollout 内部（Rust observe/step/drain、GPU 前向、Python 胶水）各段占比可量出来，找出最大块；②把下一迭代的 rollout 放到后台线程与本迭代更新重叠（演员落后 1 次更新），每迭代墙钟可降 ≥1.4×，且演员/学习者 KL 与 PPO 裁剪统计仍在正常范围。
不在范围：在跑的 exp124/exp125 一律不换训练器（判决变量污染）；学习等价性（落后 1 步不伤 Elo）需另起云 A/B，不在本记录内判。

## Method
1. 剖析：monkeypatch `riichi_rs.VecEnv`（计时 observe/step/drain/done）与网络前向（cuda 同步计时），用 exp124 A 112M latest.pt、与 pod 完全相同的 rollout 配置（K=1024，2048 局/迭代，dup_k 8，bf16，rollout_values，compile）跑 3 次 collect_rust，余量记为 Python 胶水。
2. 基线：本机跑训练器 4–6 迭代（pod 的 COMMON 参数，--total_games 极小，exp_dir 放 probes），读 rollout/pack/update。
3. 原型：训练器加 `--async_rollout`：独立演员网络（每迭代开头从学习者拷权重）在后台线程 + 独立 CUDA stream 收集下一迭代；主线程更新。函数属性全局状态（collect_rust.last_m46）改为随结果返回。
4. 对照：同种子同起点，同步 vs 异步各跑 6 迭代，比较每迭代墙钟、局/s、KL、裁剪率、ratio 均值。

## Success Criteria
- 剖析：各段占比之和与 rollout 墙钟误差 <10%。
- 原型：每迭代墙钟 ≥1.4× 提速（本机），无崩溃；异步首轮 PPO ratio 偏离 1 的幅度（|ratio−1| 均值）与 KL 不超过同步的 3×（落后 1 步的预期代价）；同步模式输出逐位不变（`--async_rollout` 缺省关）。

## Progress
- [2026-11-03 19:50] rollout 分段（exp124 112M ckpt，2048 局/迭代，本机 4080，3 次平均 2.66 s）：GPU 前向 forward_with_value 39% + 立直/杠第二次查询 15% = **54%**；Rust step 17%、drain 6%、observe 3%、局末重写 5%；m46 掩码等 Python 4%；未计胶水 9%。**rollout 不是 CPU 瓶颈，是 GPU 前向**（推翻我 11-03 早些时候"Rust 引擎吃 CPU"的说法；Community 慢是宿主单核慢拖慢了前向的 kernel 派发和 Rust step）。
- [19:55] 前向 vs 批量：≤512 行每次 2.5 ms 平（启动开销），≥1024 行线性 ≈20 万行/s（算力受限）；rollout 每轮 ≈1100 行，处于算力区。
- [20:00] 训练器基线（pod 同参数，12 迭代取后 9）：rollout 2.74 / 打包 0.17 / 更新 2.51 s，**5.43 s/迭代 = 378 局/s**（3090 pod 3.0/0.2/2.8）。
- [20:10] `--m46_one_forward`：ConvFormer46 的 mask 只在输出端 masked_fill，第二次查询直接复用第一次的原始 logits；T=0 贪心轨迹/logprob/价值逐位一致（tests/test_m46_one_forward.py）。rollout 2.66→2.35 s，**4.97 s/迭代 = 412 局/s（+9%）**。
- [20:25] `--async_rollout`（后台线程 + 独立演员 + 独立 CUDA stream）：普通优先级 4.30 s/迭代（476 局/s），后台 rollout 被更新抢 GPU 拖到 4.27 s 成关键路径；stream 设高优先级后 **3.55 s/迭代 = 577 局/s（基线 1.53×）**，rollout 等待 0，关键路径变成更新（GPU 总量 ≈ 前向 1.05 + 更新 2.5 s，已贴 GPU 下限）。代价：approx_kl 0.0025→0.0099（落后 1 次更新带来的离策略量，仍远低于 target_kl 0.03），probe_kl 0.006→0.014，EV/价值损失/熵不变。同步路径与改动前逐位一致（8 迭代 approx_kl/pg/value/EV 差 0）。
- [20:30] 用户问"把一批拆成两批累加"：rollout 内拆成两个 VecEnv 交替（2026-09-09 量过慢 20–30%，当时 CNN 前向受启动开销限制，减半批量就加倍开销）。现在 ConvFormer 前向在 ≥512 行已是算力区，减半不掉效率，但能藏的只有 Rust step ≈0.5 s；而异步重叠已把 rollout 的 CPU 部分全藏在更新后面，GPU 成为瓶颈后再拆没有增益——未实现。
- [20:40–21:10] 用户问"累加到更大 global batch 是否更稳、固定时间收敛更快"：训练器加 `--grad_noise_probe`（θ_old 处用更新同款损失，按 McCandlish 2018 估 |G|² 与 tr Σ，b = 512…16384，>4096 用梯度累加），在 exp124 A 8M/32M/64M/112M 四个检查点各测一次（每迭代 ≈13 万行，64 行/局）：

| 检查点 | 策略 |G_N| 全批 | 策略 |g_4096| | 策略 B_simple | 价值 B_simple（行） | 主干 |g_4096| 价值/策略 | 总 |g_4096|（clip=1.0） |
|---|---|---|---|---|---|---|
| 8M | 0.058 | 0.333 | ∞ / 0.6–3.8M | ≈10k | 8.86 / 0.43 = 21× | 12.4 |
| 32M | 0.059 | 0.345 | ∞ / 4M | ≈7.5k | 6.04 / 0.35 = 17× | 6.7 |
| 64M | 0.078 | 0.444 | ∞ / 0.7–13M | ≈26k | 4.67 / 0.27 = 17× | 8.4 |
| 112M | 0.063 | 0.340 | ∞ / 0.6–2M | ≈40k | 3.84 / 0.32 = 12× | 7.6 |

  策略梯度范数严格按 1/√b 缩放（512→16384 比值 5.66 = √32），全批 13 万行的 0.06 也正好是纯噪声外推值 ⇒ **每迭代的全部数据都低于策略梯度的临界 batch**，策略信号只能靠 Adam 动量跨多步累积。价值梯度临界 batch 7k–40k 行，随训练增长。主干上价值梯度是策略梯度的 12–21 倍；总梯度范数 4–12，`clip_grad_norm 1.0` 每步都在裁，裁剪系数由价值梯度定。

## Results
| 配置（本机 4080，exp124 112M 起，9 迭代均值） | s/迭代 | 局/s | approx_kl | 判据 |
|---|---|---|---|---|
| 基线（pod 同参数） | 5.43 | 378 | 0.0025 | — |
| + `--m46_one_forward` | 4.97 | 412 | 0.0027 | 语义不变 ✅ |
| + `--async_rollout`（普通优先级） | 4.30 | 476 | 0.0100 | — |
| + `--async_rollout`（高优先级 stream） | **3.55** | **577** | 0.0099 | 提速 ≥1.4× ✅；KL ≤3× ❌（3.96×） |

## Conclusion
- **吞吐**：rollout 的大头是 GPU 前向，不是 Rust 引擎。两项改动合计 1.53×（单 `--m46_one_forward` +9%，零语义变化，可直接用于下一个 run）。异步重叠后 GPU 成为唯一瓶颈，下一步提速只能减少 GPU 工作量（更新 2.5 s 是大头）。
- **异步的代价**：演员落后 1 次更新，approx_kl 4×（0.01，仍在信赖域内）；"学习效率不受损"未验证，按预注册不能直接上生产，需云 A/B（同种子同配方从零到 16M，比较局数轴和墙钟轴曲线）。
- **batch 问题（理论量已测，学习曲线未测）**：策略梯度在 b ≤ 13 万行全部处于纯噪声区。在这个区间，Adam 下加大 batch 若不加 lr，每样本进展按 √k 变慢（更稳但更慢）；按 √k 放大 lr，则每样本的信号和噪声都不变——**固定时间收敛不会更快**，更新 FLOPs 相同所以墙钟也不变。理论预期：大 global batch = 更平滑的曲线、同样或更慢的收敛。需云 A/B 实证。
- **计划外发现**：价值梯度在共享主干上是策略梯度的 12–21 倍，且每步被全局 clip 裁剪。exp46（BC 初始化 RL，≤2M 局）里 `--value_detach` 是唯一让 pg_loss 转负的干预，但纯血从零线（exp101/113/124/125）全部没开。按 11-01 范围戳规则，这是纯血规模上的待检假设，可能比 batch 大小更值得测。
- **范围戳**：ConvFormer46 2.0M（exp124 配方），纯血自对弈 8M–112M 局检查点，本机 4080；吞吐数字在 3090/4090 上比例应相近但未实测；噪声尺度为单次 rollout 估计（"∞"表示 G² 估计为负，即信号低于估计噪声）。

## Next Steps
- 云 A/B（待用户批准，各臂纯血从零到 16M，3090 Secure $0.50/h，判读用 exp101 种子 2×10k 对局）：A=exp124 A 现成曲线（同代码同配方，同步路径逐位一致，不重跑）；B=`--async_rollout --m46_one_forward`（学习等价性 + 墙钟）；C=global batch 16384（梯度累加，lr ×2 = √4 缩放）；D=价值梯度平衡（`--value_detach` 或 value_coef 下调到主干梯度持平）。
- `--m46_one_forward` 可直接并入下一个 run（零语义变化）。

## Artifacts
| Path | Size | Description |
|---|---|---|
| 分支 `claude/perf-rollout-overlap`（worktree /home/hongdp/Workspace/LLM_Mahjong_perf） | — | c41df41 one_forward、cc30419 async_rollout、c241c25 grad_noise_probe |
| experiments/probes/perf_overlap_{base,sync,onefwd,async,async_prio}/ | ~25 MB 各 | 吞吐对照训练日志 + train_log.json（start.pt = exp124 A 112.23M） |
| experiments/probes/gns_{8M,32M,64M,112M}/gns.json | 小 | 梯度噪声探针输出 |
| scratchpad perf/{profile_rollout.py,fwd_bench.py,run_cfg.sh,noise_probe.sh} | — | 一次性探针脚本（不入库） |
