# v2_engine_ppo_run_20260802_054914

- **Date**: 2026-08-01 22:49 (05:49 UTC 08-02)  **Status**: stopped by user at epoch 24/50 (2026-08-02 ~11:30 local) — style migration already pronounced (riichi halved, melds +75% fleet-wide); arena evaluation prioritized over completing 50 epochs. All checkpoints + logs preserved to GCS. Final checkpoint = checkpoint_epoch_24.
- **Arm**: A — PPO algorithm ablation. Single variable vs the concurrently running PBRS restart run `v2_engine_pbrs_run_20260802_054918` (REINFORCE): the RL update rule. Everything else identical — PBRS potential reward, same SFT adapter (from `v2_engine_full_run_20260802_005918`, loaded verbatim, sft_epochs=0), seed 42, 50 epochs × 12 games, lr 1e-6, temp 0.9/top_p 0.95.
- **Algorithm**: PPO clipped surrogate (eps 0.2), 3 inner passes per rollout batch with approx-KL early stop at 0.03. Behavior logprobs recorded at rollout from RAW pre-warp logits (`output_logits=True`); sequences rebuilt from stored token ids (no retokenization drift). At old==new the gradient equals the REINFORCE gradient (unit-tested). No critic: advantages remain buffer-normalized MC return-to-go clipped ±5 (GRPO-flavored).
- **Env**: VM `mahjong-a100-e` (a2-highgpu-1g, A100-SXM4-40GB), zone us-east1-b, same image family/pinned venv as prior runs.
- **Input artifacts**: SFT adapter `checkpoints_sft_warmup_mahjong` pulled from `gs://llm-mahjong-experiments/v2_engine_full_run_20260802_005918/`; `data/sft_mahjong.jsonl` sha256 `b3eefd6d…becf6` verified on VM.
- **Results sink**: `gs://llm-mahjong-experiments/v2_engine_ppo_run_20260802_054914/`; auto-shutdown on exit.

- **Infra rev2→3 (see rev3 note)** (shared by all three concurrent runs): torch 2.12.1+cu129, fast-path kernels active, batched parallel rollout parallel_games=4 (6-9× measured decode scaling). Non-semantic.

- **Infra rev3** (all three arms, measured in perf_tuning_east_20260802): bf16_lora (unquantized base, +55% decode), 12 games/epoch at parallel_games=12 (~3x data per epoch at ~equal wall-clock vs 4 games), update batch_size 4, [SETTLEMENT] breakdown logging. Success criteria unchanged (scale-free). Predecessor 0410xx runs aborted in epoch 1 with no completed epochs.

## Purpose & Hypothesis
1. PPO's sample reuse (up to 3 passes) extracts more improvement per rollout batch: rl/avg_episode_reward trend ≥ the REINFORCE arm's over the same 50 epochs.
2. Clipping prevents destructive updates: format compliance stays ≥0.95 with no collapse epochs, and rl/approx_kl stays bounded (early-stop rarely at pass 1).
3. Pipeline validity: rl/clip_frac > 0 (clipping actually engages) while training remains stable.

## Success Criteria (pre-registered)
1. **Format**: rl/format_compliance ≥ 0.95 for ≥45 of 50 epochs; abort guard never triggers.
2. **Learning signal**: mean rl/avg_episode_reward over final 10 epochs > mean over first 10 epochs.
3. **Game quality**: ≥10% of rollout games end in a win.
4. **PPO health**: median rl/ppo_passes ≥ 2 (reuse actually happens) AND no epoch with approx_kl > 3× target after pass 1.
5. **Checkpoint rule**: best = highest avg episode reward checkpoint (top-3 retention).


> **评测注记（2026-08-02，run 中追加，预注册标准不变）**：同权重自对弈下，
> rl/avg_episode_reward 的结算成分四家严格归零，均值主体是 PBRS 起手常数
> （≈+6.5）——标准 #2 对策略强度近乎不敏感（用户洞察）。强度判定以标准 #3
> （和牌率）、行为探针（放铳率/听牌率/打点）与完赛后的锚定对战（复式
> checkpoint 竞技场，见 TASKS #11）为准。

## Progress
- [22:49 (05:49 UTC 08-02)] Launched. Pre-flight: local PPO integration smoke passed (1 epoch/1 game on RTX 4080), 34 unit tests green incl. PPO loss math.

## Results
(pending)

## Conclusion
(pending)

## Artifacts
config_launch.json · tensorboard/ (adds rl/approx_kl, rl/clip_frac, rl/ppo_passes) · checkpoint_epoch_N/ · mahjong_epoch_N_rollouts.txt · gpu_info.txt / pip_freeze.txt / TRAIN_EXIT · GCS mirror incl. train_nohup.log


## Final Results（2026-08-02 竞技场裁决）
- 竞技场（复式32副×双向,vs SFT 锚,原始点数）: **+331 ± 1320（24:19）** — 不显著
- 完整三臂分析: docs/report_rev3_threearm_20260802.md

## Conclusion
~600 局 RL 未产生统计可辨强度变化;PBRS 密集塑形主导了行为迁移(立直↓副露↑)。
成功标准: #1 格式 ✅/边缘(见侵蚀记录) | #2 奖励趋势(已注记为强度盲指标) | #3 和牌局 ✅ 远超10% | #4 checkpoint 规则 ✅。
后继: rev4 settlement-vs-potential 对决(docs/report_rev3_threearm_20260802.md 提案)。
