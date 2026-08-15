# exp11-A1：价值-距离剖面注入 critic（特权特征，策略不可见）

- **Date**: 预注册 2026-08-15；发射待 mahjong-dnn-vit 跑完 vit RL（看守进程自动接棒）  **Status**: pre-registered
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
- [2026-08-15 ~15:30] 预注册。实现 eb91d8f，51 单测全过；A2 冒烟通过（512 局端到端），A1 冒烟进行中。
  发射机制：exp11_queue_watcher（10 分钟轮询 GCS games_final.pt + VM TERMINATED 双确认后自动发射）。

## Results
| Metric | This run | Baseline (A0) | Success criterion |
|---|---|---|---|
| （待运行） | | | |

## Conclusion
（待运行）

## Next Steps
（待运行）

## Artifacts
| Path | Size | Description |
|---|---|---|
| gs://llm-mahjong-experiments/dnn_exp11_a1_20260815/ | — | 云端主目录（10 分钟增量同步） |
