# dnn_exp28_Ap_mixT_20260823（第一批纪元 3 训练）

- **Date**: 2026-08-23 04:15 PDT 发射  **Status**: done
- **Git**: master 291be49（纪元 3 引擎指纹 127462426506c3b4）
- **Env**: 本机 24 核 + RTX 4080（免费臂）
- **预注册**: experiments/exp28_entropy_temp_prereg/EXPERIMENT.md（目的 / 方法 / 成功标准均在预注册；本文件只记本 run 的命令、进度与结果）

## Command
`train_dnn_ppo.py --arch cnn_m_r --entropy_schedule 0:0.03,400000:0.01,700000:0.003 --rollout_temps 0.7,1.0,1.3 --games_per_iter 2048 --dup_k 8 --workers 30 --gpu_infer --infer_max_batch 512 --infer_wait_ms 4  --gae_lambda 0.95 --adv_clamp 5.0 --ppo_epochs 1 --target_kl 0.03 --clip_eps 0.2 --lr 1e-4 --gamma 0.995  --entropy_coef 0.03 --total_games 1000000 --milestones 80000,240000,400000,700000,1000000 --seed 42 --ckpt_every 10 --games_per_worker 32`（云：--workers 30；本机：--workers 24）

## Progress
- [04:15] 发射。

## Results
- 本机 103 局/s，161 min。终态熵 0.517。
- Elo T=1 1042.0 / T=0 1094.0；贪心 vs A −787 ± 1321 n.s.；差距 +316。
- 结论：H2 不成立——混合温度（0.7/1.0/1.3 + 行为 logprob 修正）在低熵之上无增益，T=1 反而更低。
  机制正确性已由行为 logprob 修复保证；作为探索手段的价值待目标熵/条件熵路线再评。

## Artifacts
| Path | Description |
|---|---|
| gs://llm-mahjong-experiments/dnn_exp28_Ap_mixT_20260823/ | 主目录（ckpt、train_log.json、tensorboard） |
