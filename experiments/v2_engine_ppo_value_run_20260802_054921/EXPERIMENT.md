# v2_engine_ppo_value_run_20260802_054921

- **Date**: 2026-08-01 22:49 (05:49 UTC 08-02)  **Status**: stopped by user at epoch 25/50 (2026-08-02 ~11:30 local) — style migration already pronounced (riichi halved, melds +75% fleet-wide); arena evaluation prioritized over completing 50 epochs. All checkpoints + logs preserved to GCS. Final checkpoint = checkpoint_epoch_25.
- **Arm**: B — PPO + value bundle. Deliberately multi-variable vs arm A (`v2_engine_ppo_run`): adds the full value iteration on top of PPO. Deltas: (1) reward_model=potential_value — PBRS energy gains +0.3·(#dora held); (2) prompt template gains a computed `自家宝牌:` line (value_facts=true); (3) fresh SFT corpus regenerated with the value-aware teacher (dora-keeping tie-break, faithful CoT mentions it) — SFT reused from predecessor 041051 (value-template adapter, 3×2000, final loss 0.0877) — loaded verbatim, sft_epochs=0.
- **Reward math note**: the dora term is a potential term — telescoping to −Φ(s₀) still exact (unit-tested), so consistency/anti-farming guarantees carry over; it only redirects credit toward keeping value tiles.
- **Algorithm**: PPO identical to arm A (eps 0.2, 3 passes, target_kl 0.03).
- **Env**: VM `mahjong-a100-w` (a2-highgpu-1g, A100-SXM4-40GB), zone us-west1-b, same image/pinned venv.
- **Input artifacts**: `data/sft_mahjong_value.jsonl` — value-aware regeneration, 300 games, seed 42, line-shuffled seed 42; sha256 `4133fa33284c2be7f07b76477ef21f9b3fbefc7e5905e96928e48b9eca47a76f` (verified on VM after sync).
- **Results sink**: `gs://llm-mahjong-experiments/v2_engine_ppo_value_run_20260802_054921/`; auto-shutdown on exit.

- **Infra rev2→3 (see rev3 note)** (shared by all three concurrent runs): torch 2.12.1+cu129, fast-path kernels active, batched parallel rollout parallel_games=4 (6-9× measured decode scaling). Non-semantic.

- **Infra rev3** (all three arms, measured in perf_tuning_east_20260802): bf16_lora (unquantized base, +55% decode), 12 games/epoch at parallel_games=12 (~3x data per epoch at ~equal wall-clock vs 4 games), update batch_size 4, [SETTLEMENT] breakdown logging. Success criteria unchanged (scale-free). Predecessor 0410xx runs aborted in epoch 1 with no completed epochs.

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


> **评测注记（2026-08-02，run 中追加，预注册标准不变）**：同权重自对弈下，
> rl/avg_episode_reward 的结算成分四家严格归零，均值主体是 PBRS 起手常数
> （≈+6.5）——标准 #2 对策略强度近乎不敏感（用户洞察）。强度判定以标准 #3
> （和牌率）、行为探针（放铳率/听牌率/打点）与完赛后的锚定对战（复式
> checkpoint 竞技场，见 TASKS #11）为准。

## Progress
- [2026-08-02 ~10:30 UTC, ep10] 轻度格式侵蚀观察：100%→98.6%/10ep。失败形态=短而完整的 think 后直接 EOS（非截断，中位 12-34 字符 vs 正常 58），高发于「宣告立直」类结论句尾。approx_kl 稳定 ~0.001（无过度更新）。不干预（距 95% 标准余量大，-10 压制在场）；此为 ref-KL 锚的首个实证依据（TASKS 背景队列 #10）。
- [2026-08-02 ~12:40 UTC, ep20] 侵蚀低点缓降：格式低点 98.6(ep10)→98.4(ep15)→97.9(ep20)，≈0.07%/epoch；奖励低点首次转负（−1.50，罚分拖累 ≈−7）。线性外推 ep50 ≈95.8%，标准 #1（≥45/50 epoch ≥95%）大概率仍可达成。BASE(REINFORCE 单遍)始终 100% ⇒ 支持「PPO 复用放大生成漂移」假说。维持不干预阈值（<95% 告警）；rev4 首选处置 = ref-KL 锚。
- [22:49 (05:49 UTC 08-02)] Launched (RL-only; value-template adapter reused from 041051).

## Results
(pending)

## Conclusion
(pending)

## Artifacts
config_launch.json · checkpoints_sft_warmup_mahjong/ (value-template adapter) · tensorboard/ · checkpoint_epoch_N/ · mahjong_epoch_N_rollouts.txt · provenance files · GCS mirror
