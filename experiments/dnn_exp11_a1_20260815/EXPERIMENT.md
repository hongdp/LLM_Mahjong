# exp11-A1：价值-距离剖面注入 critic（特权特征，策略不可见）

- **Date**: 预注册 2026-08-15；发射 2026-08-16 04:17 UTC；训练完成 09:59 UTC；判定 2026-08-16 ~16:30 本地  **Status**: done（主判定 null）
- **Git**: eb91d8f（实现 commit；发射时如有增量另记）
- **Env**: GCP g2-standard-32 spot（32 vCPU + L4），DLVM cu129，torch 2.13.0+cu129，mahjong==2.0.0
- **对照**: A0 = `dnn_exp11_a0_20260815r2`（同配方普通 critic + GAE，同种子同里程碑）

## Purpose & Hypothesis
用户提议的第一形态：把「手牌距价值≥X 的役种族的向听式距离」（X∈{2000,4000,5000,8000}，4 维单调剖面）
作为**只进 value head** 的特权特征。假设：critic 借注入的役种-价值结构给出更好的 V/GAE 优势 →
策略在不直接看特征的情况下被间接引导（信息平价保持，`test_a1_logits_independent_of_cfeats` 锁定）。
若 A1≈A0，说明剖面信息 critic 自己能从棋盘学到（或用不上）；若 A1>A0，证明外部役种知识有增量。

## Method
- 唯一改动 vs A0：`--critic_feats profile`（value head 输入拼接 4 维剖面；rollout worker 逐步计算并传输）。
- 从零训练（不 warm start），种子/牌种子/里程碑/超参与 A0 完全一致 → 三臂受控比较 A0/A1/A2。
- 特征实现：`yaku_features.value_distance_profile`（单调性由 50 随机手测试锁定）。

## Config
`--total_games 600000 --games_per_iter 2048 --dup_k 8 --workers 30 --lr 1e-4 --entropy_coef 0.03
--value_coef 0.5 --clip_eps 0.2 --ppo_epochs 1 --gae_lambda 0.95 --target_kl 0.03 --batch 8192
--drop_zero_return --train_device cuda --ckpt_every 25 --milestones 20000,80000,240000,600000
--seed 42 --critic_feats profile`（与 A0 的 diff 仅最后一项；完整快照见本目录 config.json，发射后由 GCS 同步）

## Success Criteria（发射前定死）
1. **主判定**：600k 终点 vs A0-600k 的 200-deal 复式配对竞技场；|配对分差| 超出 95% CI 才算非 null。
   预期功效：200 deals SE≈430 → 可检出 ~850+ 的真差。
2. **机制指标**（次要，不能替代主判定）：explained_var 轨迹 ≥ A0 同期（A0 在 30k 局处 ~0.14）；
   value_loss/Var 比值不高于 A0。
3. 训练健康：无 NaN、KL<0.03、吞吐 ≥ 20 局/s（特征计算开销 <25%）。

## Progress
- [2026-08-15 ~15:30] 预注册。实现 eb91d8f，51 单测全过；A1/A2 各 512 局冒烟通过。
  发射机制：exp11_queue_watcher（10 分钟轮询 GCS games_final.pt + VM TERMINATED 双确认后自动发射）。
- [2026-08-15 ~15:50] us-central1-a spot 一日内两次双机同时抢占（14:02/15:01 本地）→ 迁移
  us-central1-b 新机 mahjong-dnn-b1/b2；前置 run（vit RL / A0 基线）以 r3 后缀从零重跑。
  运维参数变更（不影响训练数学）：--ckpt_every 25→10（抢占损失窗 ~33→~13 分钟）；runner 新增
  2 分钟 TB 轻量同步通道。本臂将在 b1 跑完 dnn_vit_rl_20260815r3 后由看守进程自动发射。
- [2026-08-16 04:17 UTC] r3 再遭抢占后转 on-demand 两区拓扑（c 系机），看守进程在 vit-r4
  （mahjong-dnn-c2/c3 域）完成后自动发射本臂于 mahjong-dnn-c3（us-east1-b）。
- [2026-08-16 09:59 UTC] **600k 局跑满自动关机**。终点自对弈 win_rate 0.777、entropy 1.01、
  293 iter；里程碑 20k/80k/240k/600k + final 全部落 GCS。
- [2026-08-16 ~14:45 本地] 准备主判据竞技场（vs A0-r4-600k，200 副）时本地网络故障
  （HTTPS 全断，GCS 不可达），checkpoint 下载阻塞；已挂网络恢复哨兵，恢复即跑。

## Results
| Metric | This run (A1) | Baseline (A0-r4) | Success criterion |
|---|---|---|---|
| 主判定：600k vs A0-600k，200 副复式（seed0=20260817） | **+494 ± 1008，wins 159:150** | — | 超 95% CI ⇒ **未达（null）** |
| explained_var @30k/80k/240k/600k | 0.125/0.072/0.070/0.076 | 0.138/0.081/0.070/0.081 | ≥A0 同期 ⇒ **未达（≈或略低）** |
| 训练健康 | 600k 跑满、无 NaN、KL 正常、终点自对弈 win_rate 0.777 | 0.811 | ✅ |

结果 JSON：`experiments/exp11_arena_A1_vs_A0.json`。

## Conclusion
**双 null，假说的否定分支成立**：价值-距离剖面作为 critic 特权特征，既没有改善价值预测
（explained_var 与 A0 全程重合），也没有转化为竞技场强度。解读=预注册中写明的
「剖面信息 critic 自己能从棋盘学到（或用不上）」。外部役种-价值结构知识以*这种注入方式*
无增量；A2（hazard 分解头，结构性更强的注入）是同一问题的下一个更强检验，正在跑。

## Next Steps
- A2 跑完后做三臂横向总结（A0/A1/A2 + 风格档案），一并出 exp11 总报告。
- A1/A0 checkpoint 已入本地 `_cloud_ckpts/`，将由 Elo ladder 评级入历史库存档。

## Artifacts
| Path | Size | Description |
|---|---|---|
| gs://llm-mahjong-experiments/dnn_exp11_a1_20260815/ | — | 云端主目录（10 分钟增量同步） |
