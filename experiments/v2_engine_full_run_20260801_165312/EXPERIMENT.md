# v2_engine_full_run_20260801_165312

- **Date**: 2026-08-01 16:53  **Status**: aborted (paused by user at ~17:15, mid SFT epoch 1, no checkpoints produced — migrating this exact experiment design to a GCP VM; see docs/handoff_gcp_phase1.md)
- **Git**: `f132ba1` (the engine-v2 tree this run executes was committed as f132ba1 shortly after launch; identical content — launch recorded "b00105e + uncommitted", zero code changes between launch and commit)
- **Env**: Qwen/Qwen3.5-2B (QLoRA nf4, LoRA r16 explicit target_modules), RTX 4080 16GB, transformers 5.8.1, peft/bitsandbytes per requirements.txt, seed 42
- **Input artifacts**: `data/sft_mahjong.jsonl` — 16601 samples, 300 games, faithful CoT, line-shuffled seed 42; sha256 `b3eefd6d144e662b6ed4239cfbdb62197a2c4a941264ae360ab5a250615becf6`

## Purpose & Hypothesis
First full training run on the rewritten (rule-faithful) engine. Hypotheses:
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

## Config
Snapshot in `config_launch.json`. Key: batch_size 2 (248k-vocab logits OOM at 4),
min_format_rate 0.3 abort guard, top-3+latest checkpoint retention by avg
episode reward.

## Success Criteria (defined before launch)
1. **Format**: rl/format_compliance ≥ 0.95 for ≥45 of 50 epochs; never triggers the 3-consecutive-epoch abort.
2. **Learning signal**: mean of rl/avg_episode_reward over final 10 epochs > mean over first 10 epochs (TensorBoard).
3. **Game quality**: ≥10% of rollout games across the run end in a win (ron/tsumo) rather than exhaustive draw.
4. **Checkpoint rule**: "best" = epoch checkpoint with highest avg episode reward (trainer's top-3 retention); final comparison uses that, not the last epoch.

## Progress
- [2026-08-01 16:53] Launched. Prior validation: 16 unit tests green; 50-game random stress (points zero-sum, 136-tile conservation); smoke run v2_smoke_20260801_164239 → SFT loss 0.46 (1 epoch/200 samples), rollout format 100% (86/86), no OOM.

## Results
| Metric | This run | Baseline | Success criterion |
|---|---|---|---|
| (pending) | | | |

## Conclusion
(pending)

## Next Steps (queued before launch)
- Defense probes: measure fold-rate and deal-in-rate after opponent riichi from per-epoch rollout logs (`mahjong_epoch_N_rollouts.txt`).
- Next data iteration: teacher defense vocabulary (safe-tile reasoning vs riichi) + scenario curriculum (start states with an opponent already riichi).
- Expert iteration loop: mine top-reward episodes from this run into the next SFT corpus.

## Artifacts
| Path | Size | Description |
|---|---|---|
| config_launch.json | 1KB | effective config snapshot |
| tensorboard/ | — | sft/step_loss, rl/loss, rl/avg_episode_reward, rl/format_compliance |
| checkpoint_epoch_N/ | ~50MB each | LoRA adapters, top-3 by reward + latest |
| checkpoints_sft_warmup_mahjong/ | ~50MB | post-SFT adapter (reusable for future RL-only runs) |
| mahjong_epoch_N_rollouts.txt | — | full rollout transcripts per epoch (probe mining source) |
