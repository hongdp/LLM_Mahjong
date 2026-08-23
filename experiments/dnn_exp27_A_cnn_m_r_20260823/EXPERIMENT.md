# dnn_exp27_A_cnn_m_r_20260823（第一批纪元 3 训练）

- **Date**: 2026-08-23 04:15 PDT 发射  **Status**: done
- **Git**: master 291be49（纪元 3 引擎指纹 127462426506c3b4）
- **Env**: GCP g2-standard-32（L4）on-demand，跑完即删 VM
- **预注册**: experiments/exp27_handset_prereg/EXPERIMENT.md（目的 / 方法 / 成功标准均在预注册；本文件只记本 run 的命令、进度与结果）

## Command
`train_dnn_ppo.py --arch cnn_m_r --entropy_schedule 0:0.03,600000:0.01 --games_per_iter 2048 --dup_k 8 --workers 30 --gpu_infer --infer_max_batch 512 --infer_wait_ms 4  --gae_lambda 0.95 --adv_clamp 5.0 --ppo_epochs 1 --target_kl 0.03 --clip_eps 0.2 --lr 1e-4 --gamma 0.995  --entropy_coef 0.03 --total_games 1000000 --milestones 80000,240000,400000,700000,1000000 --seed 42 --ckpt_every 10 --games_per_worker 32`（云：--workers 30；本机：--workers 24）

## Progress
- [04:15] 发射。

## Results
- 吞吐 196 局/s（g4-48 flex，K=32），1.0M 局 85 min，≈$3.2。
- Elo T=1（纪元 3 池）**1069.2 ± 13.4**（exp22-r2 冠军 1068.2；exp17-C 1035.1）；贪心复式 vs exp22-r2 500 副 −110 ± 1406（持平）。
- 拆分探针 77.5%（一向听长块 65.9%）；defense_iq 0.011，曝露放铳 16.8%。
- 风格（镜像 2000 局）：和牌 20.3% / 放铳 14.4% / 立直 26.1% / 副露 33.3% / 和牌巡目 12.0。
- 结论：纪元 3 纯血基线从零 1.0M 局即追平旧谱系 2.1M 局的冠军强度；风格更门清、放铳略低。

## Artifacts
| Path | Description |
|---|---|
| gs://llm-mahjong-experiments/dnn_exp27_A_cnn_m_r_20260823/ | 主目录（ckpt、train_log.json、tensorboard） |
