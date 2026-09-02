# exp60 — 价值方法线 v2：演员/学习者解耦 + 历史 ckpt 联赛 + 持久对局库

- **Date**: 2026-09-02  **Status**: running（基建实现 + 本地冒烟阶段）
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

## Results
（待）

## Conclusion
（待）

## Artifacts
| Path | Size | Description |
|---|---|---|
