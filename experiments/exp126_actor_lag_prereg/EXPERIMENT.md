# exp126 — 演员落后度 A/B：同步 vs 异步（落后一迭代）vs 流式（每步刷新），纯血从零到 16M

- **Date**: 2026-11-03（用户 11-03 批准"批准 A/B"，预算 ≈$15）  **Status**: running（11-03 22:02 / 22:11 PST 发射）
- **Git**: 代码 `claude/perf-rollout-overlap` a9012e0（发射时）（worktree /home/hongdp/Workspace/LLM_Mahjong_perf）；运维脚本本 worktree `experiments/exp126_ops/`
- **Env**: RunPod Secure RTX 3090（≥32 vCPU，$0.50/h）×2；评测本机 4080

## Purpose & Hypothesis
perf_rollout_overlap / perf_stream_rollout（本机）给出：异步重叠 1.53× 吞吐但数据平均落后 ≈48 个优化器步（approx_kl 4×）；流式 K=512/块 128 为 1.31×、落后 10.8 步（approx_kl ≈ 同步的 0.7–1.1×）。本地只能测吞吐和 KL，不能测"每局学到多少"。
问题：在 exp124 配方下，数据落后度在 {≈16（同步）, ≈48（异步）, ≈11（流式）} 之间变化，是否改变每局的学习效率？
- H_async：异步按局数轴不劣于同步（16M 处差 ≥ −1.0 pp）⇒ 墙钟轴净赚 1.5×，可作为默认。
- H_stream：流式按局数轴不劣于同步；若优于同步 ≥ +1.0 pp，说明 PPO 自身的批内落后（≈16 步）有代价。

## Method
- 三臂同一配方 = exp124 第一段（ConvFormer46 `convformer_m_v3r_m46`，lr 1e-4，seed 1241，warmup 3000 步，batch 4096，1 epoch，dup_k 8，2048 局/迭代当量，shaping 0→400k 退火，bf16 rollout，rollout_values，compile），唯一差别是演员-学习者调度：
  - A 同步：**不重跑**，用 exp124 A 第一段现成曲线（同代码路径；后续改动的同步路径逐位不变已校验）。
  - B 异步：`--async_rollout --m46_one_forward`
  - C 流式：`--stream_rollout --m46_one_forward --games_per_worker 512 --stream_chunk_games 128 --stream_queue 1 --stream_block_games 16384 --ckpt_every 6400`
- `--m46_one_forward` 语义不变（贪心轨迹逐位一致），只改 RNG 流；A 臂没开，这是与 A 唯一的非调度差异。
- 流式的 shaping 系数按 16384 局块读取（同步按 2048 局），0→400k 的退火在流式里是 24 级台阶；异步提前一迭代读取。
- 种子序列三臂相同（第 j 迭代 6_000_000 + 9973j + deal，8 副本）。
- 评测：里程碑 4M/8M/12M 各 2000 牌山（exp101 种子 101112000），16M 判决 2×1 万牌山（101112000/101112500），T=0 复式 2v2 对 bc49；A 臂 16M 补测同一判决读数（本机 $0）。

## Success Criteria（发射前定）
- 主判据（16M，2×1 万牌山均值，差的标准误 ≈0.46 pp）：
  - 臂 − A ≥ −1.0 pp ⇒ 不劣，可采用（墙钟收益：B 1.5×、C 1.3×）；
  - 臂 − A ≥ +1.0 pp ⇒ 优于同步；
  - 臂 − A < −1.0 pp ⇒ 该调度伤学习，不采用。
- 次要：墙钟轴（每臂按 pod 实测局/s 折算）对比；approx_kl 与 stream/stale_* 在轨记录。
- **局限**：每臂一个种子；16M 处 run-to-run 方差未测，判读按"差在 ±1 pp 内 = 无法区分"。

## 预算
3090 Secure $0.50/h。B ≈ 309×1.5 ≈ 460 局/s ⇒ 16M ≈ 9.7 h ≈ $4.9；C ≈ 309×1.3 ≈ 400 局/s ⇒ 11 h ≈ $5.6；含开机/门控 ≈ $12，上限 $15。超过 15 h 未到 16M 的臂报告用户。

## 范围戳（判决须带）
ConvFormer46 2.0M、exp124 配方、纯血自对弈 0–16M 局、单卡 3090、T=0 对 bc49 复式；不覆盖 CNN、BC 初始化 RL、更大规模（16M 之后落后度影响可能变化）。

## Progress
- [2026-11-03 21:55 PST] 本机两臂从零冒烟（16k 局）通过。开 Secure 3090 pod `5v4kkrfbur3rol`（EU-CZ-1，32 vCPU，ssh root@213.192.2.89 -p 40061）；第二台当时无货。
- [21:58] 异步臂门控在 pod 上失败：pod 镜像 torch 2.8 的 FX 追踪状态是进程全局的，学习者在主线程编译时，后台线程调用已编译函数直接抛 "FX to symbolically trace a dynamo-optimized function"（本机 torch 2.12 不复现）。修复 a9012e0：后台线程的策略调用遇到该错误时等待重试（检查在执行前抛出，无副作用）+ 异步演员也在主线程预热编译图。两臂门控在 pod torch 2.8 上 rc=0。
- [22:02] 异步臂发射（pod_run.sh async）。本机循环：pull_loop/watchdog/eval_loop；TB 链接 `exp126_async_lag1iter_0-16M_LIVE`、`exp126_stream_K512c128_0-16M_LIVE`，TensorBoard 按 PID 重启。
- [22:07] 第二台 Secure 3090 `jv00v47zyjt4cc`（EU-CZ-1，32 vCPU，ssh root@213.192.2.87 -p 40009）到手，撤掉异步 pod 上的 NEXT_ARM 后备；流式臂部署（编译门控 + 流式门控通过）并于 22:11 发射，本机三循环已挂。
- [22:05] 基线 A 臂 16M 判决读数（本机）：**0.4542 ± 0.0046 / 0.4560 ± 0.0046，均值 0.4551**（此前 2000 牌山单次读数 0.4353 偏低约 2 pp）。判据以此为准。
- 运维：脚本 `experiments/exp126_ops/`（每臂完成标记与拉回握手带臂名，支持同机接力 NEXT_ARM）；会话 cron 6bcd73d6 覆盖 exp124/125/126 巡检与判决记账。

- [2026-11-03 23:25 PST] 在轨（0–2M 局）：approx_kl 同步 A 0.007–0.008（最大 0.020，0 次超 target_kl 0.03）；**异步 0.016（最大 0.062，995 迭代中 49 次超 0.03 被提前截断）**；流式 0.005–0.007（最大 0.017，0 次）。lr 1e-4 早期，落后一迭代使 KL 翻倍并截断约 5% 的更新——若异步按局数落后，这是首要机制候选。吞吐：异步 ≈414 局/s、流式 ≈324 局/s（流式在 pod 上只比同步 ≈309 快 5%，本机快 31%；pod 单核较慢放大了演员每轮的 Python 开销）。按预注册不干预。

- [2026-11-04 01:50 PST] 4M 在轨读数（各 2000 牌山，exp101 种子 101112000，单读数 SE ≈1.05 pp、差 SE ≈1.5 pp）：A 同步 0.4055；**异步 0.3807（−2.5 pp，≈−1.7σ）**；**流式 0.4138（+0.8 pp）**。方向与在轨 KL（异步翻倍并被截断、流式低于同步）一致，非判决读数。

## Results

## Conclusion

## Artifacts
