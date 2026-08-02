# v2_engine_full_run_20260802_005918

- **Date**: 2026-08-01 17:59 (00:59 UTC 08-02, VM clock)  **Status**: stopped by user at SFT-complete / RL-not-started (~19:00 local) — superseded by the PBRS-reward run, which loads this run's SFT adapter verbatim (single-variable reward-model comparison was judged more valuable than finishing 50 RL epochs on the farmable step reward; user decision mid-flight)
- **Inherits**: full experiment design + all four pre-registered success criteria from `v2_engine_full_run_20260801_165312` (aborted mid-SFT locally, no checkpoints — this is the same experiment migrated to GCP, per docs/handoff_gcp_phase1.md). Hyperparameters byte-identical (`configs/v2_full_run.json`), seed 42.
- **Git**: `14aa98e` (code identical to `f132ba1` engine-v2 tree; 14aa98e only adds docs + gitignore rules)
- **Env (GCP Phase 1)**: VM `mahjong-a100` (a2-highgpu-1g: 1×A100-SXM4-40GB, 12 vCPU, 85GB RAM), zone us-central1-b, image family common-cu129-ubuntu-2204-nvidia-580 (driver 580.173.02), Python 3.10.12 venv, pinned packages mirrored from local rlhf_mahjong env (torch 2.12.0+cu130, transformers 5.8.1, peft 0.19.1, bitsandbytes 0.49.2 — full list in `pip_freeze.txt`)
- **Input artifacts**: `data/sft_mahjong.jsonl` — 16601 samples; sha256 verified on VM after rsync: `b3eefd6d144e662b6ed4239cfbdb62197a2c4a941264ae360ab5a250615becf6` ✅
- **Results sink**: `gs://llm-mahjong-experiments/v2_engine_full_run_20260802_005918/` (uploaded by run_training.sh on completion or failure, then VM auto-shutdown)
- **Cost plan**: on-demand a2-highgpu-1g @ ~$3.67/h, projected ~13h wall-clock ≈ ~$48. Spot rejected: PREEMPTIBLE_CPUS quota = 0 and trainer has no mid-run resume (checkpoints only after SFT completes / per RL epoch).

## Purpose & Hypothesis
Identical to `v2_engine_full_run_20260801_165312` (inherited verbatim):
1. Template-aligned faithful-CoT SFT gives ≥95% action-format compliance that RL does not erode.
2. With real point settlement distributed to all four trajectories (incl. deal-in payer, placement bonus ±2/±0.5), 50 epochs of advantage-weighted RL produces a measurable upward trend in mean episode reward over self-play — i.e. the reward plumbing produces learnable signal, not noise.
3. Win actions (riichi/ron/tsumo taught by SFT) survive into rollouts: some games end in wins, not 100% exhaustive draws.

## Method
SFT warm-up (3 epochs × 2000 samples, lr 1e-4) then advantage-weighted NLL RL
(50 epochs × 4 self-play games, lr 1e-6, gamma 0.99, advantage normalized +
clipped ±5, sampling temperature 0.9/top_p 0.95). Baseline for comparison:
`format_check_20260801_150549` (old engine, Qwen2.5-0.5B, format 99.5%, no
valid strategy signal). No same-engine RL baseline exists yet — this run IS
the baseline for future iterations.

Only deltas vs the aborted local run: hardware (RTX 4080 16GB → A100 40GB),
OS/driver (see Env), and results persistence to GCS. Model, data, config,
seed, code all unchanged.

## Config
Snapshot in `config_launch.json`. Key: batch_size 2 (unchanged from local by
design — same-experiment migration, not a re-tune), min_format_rate 0.3 abort
guard, top-3+latest checkpoint retention by avg episode reward.

## Success Criteria (pre-registered, inherited unchanged)
1. **Format**: rl/format_compliance ≥ 0.95 for ≥45 of 50 epochs; never triggers the 3-consecutive-epoch abort.
2. **Learning signal**: mean of rl/avg_episode_reward over final 10 epochs > mean over first 10 epochs (TensorBoard).
3. **Game quality**: ≥10% of rollout games across the run end in a win (ron/tsumo) rather than exhaustive draw.
4. **Checkpoint rule**: "best" = epoch checkpoint with highest avg episode reward (trainer's top-3 retention); final comparison uses that, not the last epoch.

## Progress
- [2026-08-01 17:59 (00:59 UTC 08-02, VM clock)] Launched on GCP VM. Pre-flight: dataset sha256 verified on VM; torch sees A100; pinned env installed clean.
- [2026-08-01 ~18:30] SFT healthy: ~35 steps/min, epoch-2 avg loss 0.105 (smoke baseline was 0.46 at 1×200).
- [2026-08-01 ~19:00] SFT 3/3 complete, adapter saved. User decision: stop before RL and switch the RL phase to the energy-consistent PBRS reward (docs/reward_energy_pbrs.md) as a new experiment reusing this SFT adapter. This run therefore contributes the SFT artifact; no RL epochs were trained under the step reward.

## Results
SFT only: 3 epochs × 2000 samples completed on A100, final epoch avg loss ≈0.105.
Deliverable artifact: `checkpoints_sft_warmup_mahjong` (LoRA adapter), consumed by
the successor PBRS run. No RL metrics exist for this run.

## Conclusion
Partially superseded rather than failed: hypotheses moved unchanged to the PBRS
successor; the step-reward RL arm was retired before start because its shaping
is farmable and inconsistent with settlement (the PBRS analysis, triggered by
user review, showed +2/step accumulates ~10× the settlement scale).

## Next Steps (queued before launch, inherited)
- Defense probes: measure fold-rate and deal-in-rate after opponent riichi from per-epoch rollout logs (`mahjong_epoch_N_rollouts.txt`).
- Next data iteration: teacher defense vocabulary (safe-tile reasoning vs riichi) + scenario curriculum (start states with an opponent already riichi).
- Expert iteration loop: mine top-reward episodes from this run into the next SFT corpus.
- Infra backlog: proper mid-run resume (optimizer + epoch state) so future long runs can use Spot A100 (~47% cheaper; currently blocked by PREEMPTIBLE_CPUS=0 quota anyway).

## Artifacts
| Path | Size | Description |
|---|---|---|
| config_launch.json | 1KB | effective config snapshot |
| tensorboard/ | — | sft/step_loss, rl/loss, rl/avg_episode_reward, rl/format_compliance |
| checkpoint_epoch_N/ | ~50MB each | LoRA adapters, top-3 by reward + latest |
| checkpoints_sft_warmup_mahjong/ | ~50MB | post-SFT adapter (reusable for future RL-only runs) |
| mahjong_epoch_N_rollouts.txt | — | full rollout transcripts per epoch (probe mining source) |
| gpu_info.txt / pip_freeze.txt | — | hardware + exact package provenance |
| TRAIN_EXIT | — | trainer exit code + UTC finish time |
| gs://llm-mahjong-experiments/v2_engine_full_run_20260802_005918/ | — | full mirror incl. train_nohup.log |
