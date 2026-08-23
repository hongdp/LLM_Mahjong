# dnn_exp27_C_cnn_xl_r_20260823（第一批纪元 3 训练）

- **Date**: 2026-08-23 04:15 PDT 发射  **Status**: done
- **Git**: master 291be49（纪元 3 引擎指纹 127462426506c3b4）
- **Env**: GCP g2-standard-32（L4）on-demand，跑完即删 VM
- **预注册**: experiments/exp27_handset_prereg/EXPERIMENT.md（目的 / 方法 / 成功标准均在预注册；本文件只记本 run 的命令、进度与结果）

## Command
`train_dnn_ppo.py --arch cnn_xl_r --warmup_updates 150 --entropy_schedule 0:0.03,600000:0.01 --games_per_iter 2048 --dup_k 8 --workers 30 --gpu_infer --infer_max_batch 512 --infer_wait_ms 4  --gae_lambda 0.95 --adv_clamp 5.0 --ppo_epochs 1 --target_kl 0.03 --clip_eps 0.2 --lr 1e-4 --gamma 0.995  --entropy_coef 0.03 --total_games 1000000 --milestones 80000,240000,400000,700000,1000000 --seed 42 --ckpt_every 10 --games_per_worker 32`（云：--workers 30；本机：--workers 24）

## Progress
- [04:15] 发射。

## Results
- 吞吐 155 局/s，1.0M 局 ≈ 110 min。
- Elo T=1 **862.0 ± 12.9**；贪心 vs exp22-r2 −10410 ± 1247（137:648）。
- 风格：和牌 10.1% / 流局 59.9% / 立直 1.2% / 副露 96.4% / 和牌巡目 16.0；拆分探针 43%。
- 早期信号已预告：熵平台 1.2 且 250k 后回升、decisive 只有 A 的一半。结论：6.6M 宽卷积在冠军配方（LR 1e-4 / batch 8192 / 熵 0.03）下
  根本没学起来（几乎全副露、不立直）——这是"配方卡住"而非"架构不行"的案例；规模臂 × 低熵/目标熵配方进下一批。

## Artifacts
| Path | Description |
|---|---|
| gs://llm-mahjong-experiments/dnn_exp27_C_cnn_xl_r_20260823/ | 主目录（ckpt、train_log.json、tensorboard） |
