# dnn_exp27_B_handset_xl_20260823（第一批纪元 3 训练）

- **Date**: 2026-08-23 04:15 PDT 发射  **Status**: done
- **Git**: master 291be49（纪元 3 引擎指纹 127462426506c3b4）
- **Env**: GCP g2-standard-32（L4）on-demand，跑完即删 VM
- **预注册**: experiments/exp27_handset_prereg/EXPERIMENT.md（目的 / 方法 / 成功标准均在预注册；本文件只记本 run 的命令、进度与结果）

## Command
`train_dnn_ppo.py --arch handset_xl_cnn_m_r --warmup_updates 150 --entropy_schedule 0:0.03,600000:0.01 --games_per_iter 2048 --dup_k 8 --workers 30 --gpu_infer --infer_max_batch 512 --infer_wait_ms 4  --gae_lambda 0.95 --adv_clamp 5.0 --ppo_epochs 1 --target_kl 0.03 --clip_eps 0.2 --lr 1e-4 --gamma 0.995  --entropy_coef 0.03 --total_games 1000000 --milestones 80000,240000,400000,700000,1000000 --seed 42 --ckpt_every 10 --games_per_worker 32`（云：--workers 30；本机：--workers 24）

## Progress
- [04:15] 发射。

## Results
- 修复 rank 偏置原子加争用后 46 局/s（GPU 封顶），1.0M 局 ≈ 6 h。
- Elo T=1 **1058.0 ± 13.3** / T=0 **1096.1 ± 13.7**（A：1069.2 / 1121.8）；贪心 vs exp22-r2 −96 ± 1307。
- 拆分探针 76.3%（一向听长块 65.6%；A 65.9%）——**H1 否定**：实例集合在拆分上没有优势。
- defense_iq 0.010；对锚点贪心 和 24.7% / 铳 12.7% / 和牌巡目 11.7；终态熵 0.603。
- 结论：等配方下集合注意力 ≤ CNN；消融臂 D 不发射。若未来重试：低熵/目标熵配方 + 更大 d。

## Artifacts
| Path | Description |
|---|---|
| gs://llm-mahjong-experiments/dnn_exp27_B_handset_xl_20260823/ | 主目录（ckpt、train_log.json、tensorboard） |
