# exp60 — 价值方法线 v2：演员/学习者解耦 + 历史 ckpt 联赛 + 持久对局库

- **Date**: 2026-09-02  **Status**: paused（第 1 轮本地冒烟收官 09-02 00:20；基建成立、0 代晋级；下一轮见 Conclusion）
- **Git**: 基线 0bc003a；设计 [designs/design_value_actor_learner.md](../designs/design_value_actor_learner.md)（用户提出的架构）
- **Env**: 本地 4080（冒烟：演员 + 学习者同机）；正式 tranche 上 RunPod Secure（带冠军权重，安全分层规则）

## Purpose & Hypothesis
exp59 v1.x 钉死了病根：无探索的离策略 Q 只复现先验（0.49–0.50），纯 TD 重建不了排序（无锚 0.21），
可训参数量非变量（头-only ≈ 全微调）。缺的是**覆盖**与**样本效率**，而串行架构让两者都受采集速度绑死。
v2 假设：**演员持续用"最新策略 + 历史/锚点轮换池"凑桌、四席全落盘，学习者持续从对局库训练并按代际晋级**，
能让价值方法产生先验之外的增益——判据是代际贪心强度曲线（vs bc49）单调不降且至少一代 ≥0.52。

## Method
- `src/agents/dnn/replay_store.py`：分片格式（planes uint8×20 + packbits mask + fp16 标量，
  np.savez_compressed，实测 58 B/步）、写入/读取、增量扫描。
- `scripts/value_actor.py`：循环凑桌（学习席=最新 gen，其余从 anchors+最近 K 代逐席均匀抽，全 T=0，
  学习席单点偏离 p=0.04）→ `collect_parallel(all_seats)` → 写分片 + 元数据；`STOP` 文件优雅退出。
- `scripts/train_dnn_dqn.py --store_dir`：从分片流式装入 RAM 环（近期优先）→ 固定更新数/迭代 →
  每 `promote_every` 次更新做晋级评测（贪心 vs 当前最新 gen，n=1000 复式对，≥0.50−1σ 即晋级 gen+1）。
- 配方沿用 exp59 v1.2/v1.5：bc49 热启动、DQfD 锚（bc49 贪心）margin 1.0→0.3、n-step 10、Double DQN、
  target_every 500、MC 烧机 20k 局、reward_scale 0.05。
- 对照：exp59 全部臂（0.49–0.50）。

## Success Criteria（预注册）
1. **基建**：分片读写往返逐位一致（单元测试）；演员吞吐 ≥ 串行 rollout 的 90%；学习者更新不被采集阻塞。
2. **学习（冒烟，本地 ≥3 代）**：晋级机制跑通，代际曲线（每代 vs bc49，n=1000）无单调下降。
3. **学习（正式）**：每代 vs bc49 n=4000 单局 T=0；任一代 ≥ **0.52**（+2.5σ）⇒ 「价值方法产生先验外增益」成立；
   连续三代不涨（差 <1σ）⇒ 停，写诊断（覆盖/信噪比/锚），不追加预算。
4. 预算：本地冒烟 0 元；正式 tranche ≤ $5（Secure 3090）。

## Progress
- [09-02 00:30] 设计定稿；格式实测（21 个离散值、98.6% 0/1 → 58 B/步）；实现中。
- [09-01 23:05] 基建落地并本地冒烟：分片往返测试过（tests/test_replay_store.py）；演员 + 学习者同机跑通，
  首个晋级评测执行（vs bc49 0.4642±0.020 → held，机制正确）。
- [09-01 23:10] **用户指令：演员纯 CPU、学习者独占 GPU 大 batch**。演员 20 CPU worker = 24–25 局/s
  （GPU 共享版 36–40）；学习者 batch 2048，GPU 利用率 95%（共享时 30–40%）。
- [09-01 23:20] 指标分权：演员记 `actor/*`（games/steps per s、learner_pts_vs_pool、gen_playing）+ `style/*`
  （x=产出局数）；学习者记 `dqn/*`（td/q/target、samples_per_s、updates_per_s、ingest_steps_per_s、
  replay_ratio_eff、replay_size、margin_coef；x=摄入局数）。TB 6007 改 symlink 目录自动发现。
- [09-01 23:25] **晋级保险**（用户批准）：候选同时对固定锚 bc49 评测（`dqn/promo_share_vs_bc49`），
  连续两次 <0.48 冻结晋级并报警；接受门槛从 0.5−1σ 收到点估计 ≥0.5——防"对上一代不劣于"链的
  非传递性漂移（v1.6-A −6.7σ 的那种滑坡）。学习者 v3 重发（run 目录 exp60_learner_v3）。

- [09-01 23:30] 学习者 v4 重发（MC 期改按更新数：v3 因一次性摄入 22k 局积压跳过了 MC 期，出现一帧 y≈+1.0）。
- [09-02 00:20] **用户指令：本轮跑完即收**。v4 跑 48 分钟 / 15,000 次更新 / 摄入 90.6k 局，四次晋级评测全部
  held（vs bc49 0.4708 / 0.4875 / 0.4817 / 0.4667，n=600 对），**未产生 gen_0001**。停演员（STOP）与学习者，
  对局库与池子最终同步到 GCS，镜像循环关闭。云上无本项目机器在计费。

## 第 2 轮预注册（2026-09-02，用户指令：系统化设计 + 实测定设计 + 跑一次）
设计文档 [designs/design_distributed_value_training.md](../designs/design_distributed_value_training.md)。
本地已判：GPU 学习者 batch 512 即饱和（bf16 +50%）；解码非瓶颈；**CPU 演员 8 worker 即饱和 ~25 局/s，
CPU pod 方案否决**（54 vs 180 局/s/$）；拓扑 = GPU 演员 : GPU 学习者 ≈ 1:1。
**云测（Secure，带冠军权重）**：拓扑 T-B = 两台 3090 secure（$0.50/h×2，各 32 核），数据层 = GCS
（演员直推分片、学习者 rsync 拉），测：①pod↔GCS 单分片往返与端到端 lag（Q5）；②3090 演员/学习者吞吐；
若第二台拿不到 → T-C 4090×2 单机（$1.48/h，本地盘数据层）。**预算：探针 ~$1 + 第 2 轮训练 ≤ $4，合计 ≤ $5。**
第 2 轮判据同 §Success Criteria 2/3（代际曲线不降；任一代 vs bc49 n=4000 ≥0.52）。收尾：全部 terminate、
对局库/池子/日志同步 GCS。

## Results（第 1 轮，本地冒烟，2026-09-01 23:05 → 09-02 00:20）

| 判据 | 目标 | 实测 | 判定 |
|---|---|---|---|
| 1a 分片往返 | 逐位一致 | tests/test_replay_store.py 2/2 过（含变异守卫） | ✅ |
| 1b 演员吞吐 | ≥ 串行 90% | **CPU 20 worker 23–25 局/s**（串行 GPU 共享 71 局/s 的 35%；GPU 共享演员 36–40） | ❌ 纯 CPU 演员是产出瓶颈 |
| 1c 学习者不被采集阻塞 | — | GPU 94–95%，5.7 更新/s × 2048 = 11.7k 样本/s，有效回放比 ~8 | ✅ |
| 2 代际曲线 | 晋级机制跑通、不单调降 | 机制跑通；4 次评测 0.467–0.488 全 held，**0 代晋级** | ⚠️ 机制 ✅，学习 ✗ |
| 终评 | — | 学习者最新权重（77k 局处 candidate）vs bc49 **n=4000：0.4894±0.0079，−217 分/对**；A/A 0.500 | 平（−1.3σ） |

对局库：**180 分片 / 92,160 局 / 5.52M transition / 377MB（68 B/步含 mask+标量）**，
GCS `gs://llm-mahjong-experiments/exp60_store/`；池子仅 anchors + candidate.pt（`exp60_pool/`）。

## Conclusion（第 1 轮）

**基建成立，学习未起步。** 演员/学习者/对局库/晋级评测/绝对刻度保险全部跑通（本机 0 元）；但 15k 次更新、
90k 局内 Q 的贪心仍停在 bc49 水位下沿（0.467–0.489），与 exp59 全系一致——架构解决的是"能不能高效、可持续地
学"，没有解决"目标里有没有排序信息"。这一轮暴露的资源配比问题：纯 CPU 演员产出只有 GPU 共享版的 2/3，
学习者反而"等饭"（回放比 8）。

**下一轮该做的（按优先级）**：①演员拿回 GPU 推理服务器或加演员（云上 CPU pod）；②在同一份对局库上并行开学习者
消融：n-step {5,10,20,MC}、target_every {250,500,2000}、`--history_frac 0.2`、margin m {0.05,0.01}；
③**配对牌山差分目标**（信噪比杠杆，v3 议题）——这是唯一直接攻击"排序信息不足"的方案；④晋级评测 n 从 600 对
提到 2000 对（±0.011）以免噪声误晋级。

## Artifacts
| Path | Size | Description |
|---|---|---|
| experiments/exp60_store/ + gs://llm-mahjong-experiments/exp60_store/ | 377MB | 180 分片对局库（四席全收、来源标签） |
| experiments/exp60_pool/ | 8MB | anchors.json + candidate.pt（77k 局处学习者权重） |
| experiments/exp60_learner_v{2,3,4}/, exp60_smoke_learner*/ | — | 学习者 run 目录（train_log/TB/ckpt） |
| experiments/exp60_store/tensorboard_actor/ | — | 演员侧 TB（actor/*, style/*） |
| experiments/probes/exp60_v4_candidate77k_vs_bc49_n4000.json | 1KB | 终评 |
| scripts/value_actor.py, src/agents/dnn/replay_store.py, tests/test_replay_store.py | — | 基建代码 |
