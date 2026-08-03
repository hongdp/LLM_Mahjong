# v2_engine_pbrs_run_20260802_054918

- **Date**: 2026-08-01 22:49 (05:49 UTC 08-02)  **Status**: stopped by user at epoch 26/50 (2026-08-02 ~11:30 local) — style migration already pronounced (riichi halved, melds +75% fleet-wide); arena evaluation prioritized over completing 50 epochs. All checkpoints + logs preserved to GCS. Final checkpoint = checkpoint_epoch_26.
- **Infra rev2→3** (non-semantic, all three concurrent runs share it): torch 2.12.1+cu129, Qwen3.5 fast-path kernels ACTIVE (flash-linear-attention 0.5.2 + causal-conv1d 1.6.2 source-built), batched parallel rollout `parallel_games=4` (scheduler unit-tested; measured aggregate decode scaling 6-9× on A100). Game semantics identical to the sequential path.
- **Design change vs v2_engine_full_run**: single variable — RL step shaping switched from `MahjongStepReward` (absolute scores, farmable, inconsistent with settlement) to **`MahjongPotentialReward`** (energy-consistent PBRS; see docs/reward_energy_pbrs.md). User-approved mid-flight switch: the predecessor run `v2_engine_full_run_20260802_005918` was stopped at SFT-complete/RL-not-started, and its SFT adapter is loaded verbatim (`peft_model_path` + `sft_epochs=0`), so the SFT stage is bit-identical by construction.
- **Reward math**: Φ(h) = −2.0·shanten + 0.05·|ukeire|; F_i = γψ_i − ψ_{i−1}; terminal energy := 0. Discounted shaping telescopes to −Φ(initial hand) ⇒ shaped return = settlement return + deal constant: cannot be farmed, optimal policy invariant (Ng et al. 1999). Format −10 / ghost-tile −5 stay as constraint terms. 6 unit tests green (`tests/test_potential_reward.py`).
- **Git**: local commit pending (rewards/task/trainer + configs/v2_pbrs_run.json); code rsynced to VM and import-verified before launch.
- **Env**: identical to v2_engine_full_run_20260802_005918 — VM mahjong-a100 (A100-SXM4-40GB, us-central1-b), same venv/pinned packages, seed 42.
- **Input artifacts**:
  - SFT adapter: `experiments/v2_engine_full_run_20260802_005918/checkpoints_sft_warmup_mahjong` (3 epochs × 2000 samples, final epoch avg loss ≈0.105, trained this same day on the A100)
  - `data/sft_mahjong.jsonl` sha256 `b3eefd6d…becf6` (already verified on VM)
- **Results sink**: `gs://llm-mahjong-experiments/v2_engine_pbrs_run_20260802_054918/`; VM auto-shutdown on exit.

- **Restart provenance**: identical design to  (aborted mid-epoch-1 for the infra fix — no results existed). Same SFT adapter, config, seed.

- **Infra rev3** (all three arms, measured in perf_tuning_east_20260802): bf16_lora (unquantized base, +55% decode), 12 games/epoch at parallel_games=12 (~3x data per epoch at ~equal wall-clock vs 4 games), update batch_size 4, [SETTLEMENT] breakdown logging. Success criteria unchanged (scale-free). Predecessor 0410xx runs aborted in epoch 1 with no completed epochs.

## Purpose & Hypothesis
1. (inherited) Template-aligned faithful-CoT SFT gives ≥95% action-format compliance that RL does not erode.
2. (revised for PBRS) With shaping that telescopes to a deal constant, the RL objective is the settlement itself: rl/avg_episode_reward now measures true game outcome + constant, so an upward trend is evidence of actual mahjong improvement, not shaping farming.
3. (inherited) Win actions survive into rollouts: some games end in ron/tsumo.
4. (new, qualitative) Because intermediate reward can no longer be farmed by "many locally-optimal discards", trajectories that lose points should receive clearly negative advantages → earlier emergence of defensive behavior is plausible; probe with the fold/deal-in analysis.

## Method
No SFT warm-up (adapter loaded). RL: 50 epochs × 12 self-play games, lr 1e-6,
gamma 0.99 (buffer AND potential shaping), advantage normalized + clipped ±5,
sampling temperature 0.9/top_p 0.95, batch_size 4, min_format_rate 0.3 abort.
Config snapshot in config_launch.json (`configs/v2_pbrs_run.json`).
Comparison baseline: none same-reward; the aborted step-reward run provides
the SFT stage only. This run becomes the PBRS baseline.

## Success Criteria (pre-registered before launch)
1. **Format**: rl/format_compliance ≥ 0.95 for ≥45 of 50 epochs; abort guard never triggers.
2. **Learning signal**: mean rl/avg_episode_reward over final 10 epochs > mean over first 10 epochs. (Scale differs from the step-reward design; the trend criterion is scale-free.)
3. **Game quality**: ≥10% of rollout games end in a win (ron/tsumo).
4. **Checkpoint rule**: "best" = highest avg episode reward checkpoint (top-3 retention), not the last epoch.


> **评测注记（2026-08-02，run 中追加，预注册标准不变）**：同权重自对弈下，
> rl/avg_episode_reward 的结算成分四家严格归零，均值主体是 PBRS 起手常数
> （≈+6.5）——标准 #2 对策略强度近乎不敏感（用户洞察）。强度判定以标准 #3
> （和牌率）、行为探针（放铳率/听牌率/打点）与完赛后的锚定对战（复式
> checkpoint 竞技场，见 TASKS #11）为准。

## Progress
- [2026-08-01 22:49 (05:49 UTC 08-02)] Launched (RL-only, PBRS shaping). Predecessor stopped cleanly after its SFT checkpoint was written; VM kept up; monitor state reset.

## Results
| Metric | This run | Criterion |
|---|---|---|
| (pending) | | |

## Conclusion
(pending)

## Next Steps
- Defense probes (fold-rate / deal-in-rate after opponent riichi) from per-epoch rollout logs.
- Optional next iteration: state-value baseline to cancel the −ψ_{t−1} variance term in return-to-go.
- Expert iteration: mine top-reward episodes into the next SFT corpus.

## Artifacts
| Path | Description |
|---|---|
| config_launch.json | effective config snapshot |
| tensorboard/ | rl/loss, rl/avg_episode_reward, rl/format_compliance, rl/avg_advantage |
| checkpoint_epoch_N/ | LoRA adapters, top-3 by reward + latest |
| mahjong_epoch_N_rollouts.txt | rollout transcripts (probe mining source) |
| gpu_info.txt / pip_freeze.txt / TRAIN_EXIT | provenance |
| gs://llm-mahjong-experiments/v2_engine_pbrs_run_20260802_054918/ | full mirror incl. train_nohup.log |


## Final Results（2026-08-02 竞技场裁决）
- 竞技场（复式32副×双向,vs SFT 锚,原始点数）: **+1038 ± 1876（27:18）** — 方向正但不显著
- 完整三臂分析: docs/report_exp1_shaping_arms_20260802.md

## Conclusion
~600 局 RL 未产生统计可辨强度变化;PBRS 密集塑形主导了行为迁移(立直↓副露↑)。
成功标准: #1 格式 ✅ 全程100% | #2 奖励趋势(已注记为强度盲指标) | #3 和牌局 ✅ 远超10% | #4 checkpoint 规则 ✅。
后继: exp2 settlement-vs-PBRS 对决(docs/report_exp1_shaping_arms_20260802.md 提案)。
