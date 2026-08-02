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

## In progress
- Real-game E2E: `configs/tune_bf16_b24.json` — 24 games, parallel 24, PPO capture, update batch_size 4. Measures true rollout wall-time (incl. interrupts + tail effect) and PPO update time at ~2k samples.

## Pending decisions for relaunch (user)
- games/epoch 4 → 12–24 (design change; success criteria are scale-free; all arms change together)
- precision nf4 → bf16_lora (design change: numerics; SFT adapters were trained on nf4 base — either accept the small base-precision shift when loading, or redo SFT in bf16 ~30min)
- update batch_size 2 → 4 (gradient-noise change; A100 memory allows)

## Results
(pending)

## Conclusion
(pending)
