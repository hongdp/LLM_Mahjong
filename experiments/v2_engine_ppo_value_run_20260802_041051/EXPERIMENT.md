# v2_engine_ppo_value_run_20260802_041051

- **Date**: 2026-08-01 21:10 (04:10 UTC 08-02)  **Status**: running
- **Arm**: B — PPO + value bundle. Deliberately multi-variable vs arm A (`v2_engine_ppo_run`): adds the full value iteration on top of PPO. Deltas: (1) reward_model=potential_value — PBRS energy gains +0.3·(#dora held); (2) prompt template gains a computed `自家宝牌:` line (value_facts=true); (3) fresh SFT corpus regenerated with the value-aware teacher (dora-keeping tie-break, faithful CoT mentions it) — hence a FULL SFT warm-up (3×2000) precedes RL: the old adapter is template-incompatible by design.
- **Reward math note**: the dora term is a potential term — telescoping to −Φ(s₀) still exact (unit-tested), so consistency/anti-farming guarantees carry over; it only redirects credit toward keeping value tiles.
- **Algorithm**: PPO identical to arm A (eps 0.2, 3 passes, target_kl 0.03).
- **Env**: VM `mahjong-a100-w` (a2-highgpu-1g, A100-SXM4-40GB), zone us-west1-b, same image/pinned venv.
- **Input artifacts**: `data/sft_mahjong_value.jsonl` — value-aware regeneration, 300 games, seed 42, line-shuffled seed 42; sha256 `4133fa33284c2be7f07b76477ef21f9b3fbefc7e5905e96928e48b9eca47a76f` (verified on VM after sync).
- **Results sink**: `gs://llm-mahjong-experiments/v2_engine_ppo_value_run_20260802_041051/`; auto-shutdown on exit.

- **Infra rev2** (shared by all three concurrent runs): torch 2.12.1+cu129, fast-path kernels active, batched parallel rollout parallel_games=4 (6-9× measured decode scaling). Non-semantic.

## Purpose & Hypothesis
1. Value facts + dora-aware teacher give the model usable value signal: rollouts show measurably higher dora retention than arms without it (probe: dora count in winning hands / discarded-dora rate).
2. With PBRS+dora energy, exploration reaches higher-value wins: average winning-hand settlement (points transferred per win) exceeds arm A's.
3. Format compliance unaffected by the template change (new SFT corpus grounds it): ≥0.95.

## Success Criteria (pre-registered)
1. **Format**: rl/format_compliance ≥ 0.95 for ≥45 of 50 epochs; abort never triggers.
2. **Learning signal**: mean rl/avg_episode_reward final-10 > first-10.
3. **Game quality**: ≥10% of games end in a win.
4. **Value signal** (vs arm A, post-hoc probe on rollout logs): mean |terminal settlement| of winning trajectories strictly greater than arm A's, OR mean dora-in-hand at win > arm A's. (Directional; underpowered at 200 games — treated as evidence, not proof.)
5. **Checkpoint rule**: best = highest avg episode reward checkpoint.

## Progress
- [21:10 (04:10 UTC 08-02)] Launched (SFT 3×2000 on value-aware corpus, then 50-epoch PPO RL).

## Results
(pending)

## Conclusion
(pending)

## Artifacts
config_launch.json · checkpoints_sft_warmup_mahjong/ (value-template adapter) · tensorboard/ · checkpoint_epoch_N/ · mahjong_epoch_N_rollouts.txt · provenance files · GCS mirror
