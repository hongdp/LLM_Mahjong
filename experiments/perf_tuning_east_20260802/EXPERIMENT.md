# perf_tuning_east_20260802

- **Date**: 2026-08-01 ~22:00 (started)  **Status**: running  **Type**: performance tuning session (not an ML experiment — no learning claims; throwaway 1-epoch runs on mahjong-a100-e)
- **Context**: user paused all three training runs (central + B VMs stopped; partials + the value-template SFT adapter preserved to GCS) to tune RL throughput on one VM before relaunching.
- **Purpose**: pick the rollout/update configuration for the relaunch — maximize learning-per-wallclock-hour without changing algorithm semantics.

## Measurements so far (A100-SXM4-40GB, Qwen3.5-2B + adapter, fast-path kernels on)

Synthetic decode sweep (900-tok prompt, 64-tok gen, sampling t0.9/p0.95):

| B | nf4 aggregate | bf16 aggregate | bf16 peak mem |
|---|---|---|---|
| 1 | 11.5 tok/s | ~15.5 | — |
| 12 | 113 | 174 | 4.8GB |
| 16 | 149 | 231 | 5.0GB |
| 24 | 216 | **335** | 5.4GB |

- Scaling still near-linear at B=24 (per-seq 9.0 nf4 / 13.9 bf16) — no saturation knee found yet.
- bf16 ≈ +55% over nf4 at every B → new trainer mode `--bf16_lora` (LoRA on unquantized bf16 base; mutually exclusive with --use_qlora).
- `_batch_generate` max_batch raised 8 → 24 (follows parallel_games).

## Real-game E2E result (`tune_bf16_b24_20260802_050057`)
- 24 games, bf16_lora, parallel 24, PPO (3 passes, batch 4): **~40 min total** for rollout + update.
- 96 trajectories, 1864 transitions, format 99.9% (1863/1864) — batching + bf16 + nf4-trained adapter on bf16 base: no quality loss.
- PPO: 3 full passes, no KL early stop, batch_size 4 no OOM.
- Per-game cost 100s vs pre-optimization 525s = **5.2×**.

## Pending decisions for relaunch (user)
- games/epoch 4 → 12–24 (design change; success criteria are scale-free; all arms change together)
- precision nf4 → bf16_lora (design change: numerics; SFT adapters were trained on nf4 base — either accept the small base-precision shift when loading, or redo SFT in bf16 ~30min)
- update batch_size 2 → 4 (gradient-noise change; A100 memory allows)

## Results
See tables above. Headline: sequential nf4 11.5 tok/s → batched bf16 (B=24) 335 tok/s synthetic; real E2E per-game cost 525s → 100s (5.2×).

## Conclusion
Adopt for relaunch (infra rev3): bf16_lora + parallel_games=num_episodes + update batch_size 4 + [SETTLEMENT] logging + git-pull code sync. Games/epoch decision (4/12/24 at ~8h/~17h/~33h per 50-epoch run) escalated to user. Remaining backlog: KV-persistent rollout (v3.1), state-value baseline.
