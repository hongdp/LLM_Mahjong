# exp2_pbrs_20260807_065729

- **Date**: 2026-08-07 启动  **Status**: training complete（2026-08-08 10:27 UTC, exit 0）；竞技场判据执行中（exp2_arena_20260808_1050）
- **Series**: exp2_settlement_vs_pbrs, Arm P（对照臂）。主臂：exp2_settlement（另一台 VM，同日启动）。
- **单变量设计**: 两臂唯一差异 = `reward_model`（P: `potential` = 能量一致 PBRS，S: `settlement`）。其余完全一致：PPO(ε0.2/3passes/target_kl 0.03) + ref-KL 锚 k3 coef 0.05 + γ=0.995 + 50×12 + batch 4 + lr 1e-6 + seed 42 + 同一 SFT 锚点。
- **为什么重跑 PBRS**: exp1 的 PBRS 数据出自 pre-audit 引擎 + infra rev2 + 无 ref-KL + γ0.99，与 Arm S 不可比。本臂在完全相同的新条件下重建 PBRS 基线，使 S vs P 成为干净的单变量比较。exp1 的发现（稠密通道支配 → 风格迁移、无强度增益）在新引擎/新锚点下是否重现，本身就是有价值的重复实验。
- **Reward（Arm P）**: `MahjongPotentialReward(gamma=0.995)` — Φ(h)=−2.0·shanten+0.05·ukeire；F_i=γψ_i−ψ_{i−1}，终局 ψ:=0；折扣和 telescopes 到 −Φ(s₀)（策略不变，不可 farm）。格式 −10 / 幽灵牌 −5 约束项同 S。
- **引擎**: post-audit RCR 2016 重写（344b938）。与 exp1 不可直接比较；与 Arm S 完全可比。
- **Infra**: rev3 + DWS flex-start 临时机（同 Arm S 记录）。
- **Git**: ee0c60c
- **Env**: flex-start a2-highgpu-1g（A100 40GB），torch 2.12.1+cu129，fla 0.5.2，cc1d 预编译包。
- **Input artifacts**: SFT 锚点 `v2_engine_full_run_20260802_005918/checkpoints_sft_warmup_mahjong`（GCS；兼作 ref-KL 冻结参考）。RL-only。
- **Results sink**: `gs://llm-mahjong-experiments/exp2_pbrs_20260807_065729/`；EXIT-trap 上传后 VM 自毁。
- **Cost plan**: flex-start $2.020/h × ~14–17h ≈ $30–35；max-run-duration=36h 兜底。

## Purpose & Hypothesis
1. **H1（对照）**: 相同条件下 PBRS 臂与 settlement 臂的竞技场强度差 — exp1 推断"稠密通道抢占学习方向"若成立，S 应 ≥ P。
2. **H2**: ref-KL 锚钉住格式 ≥0.99（同 S 臂）。
3. **H3（重现性）**: exp1 风格迁移（立直↓副露↑）在新引擎 + ref-KL 下是否重现；若 ref-KL 同时抑制了风格迁移，说明 exp1 的迁移部分是无锚 PPO 漂移，而非 PBRS 本身。
4. **度量声明**: Arm P 的 rl/avg_episode_reward ≈ 结算(零和) + PBRS 常数项 E[−Φ₀]≈+5.6，趋势主要反映塑形通道；**强度判定只认竞技场**。

## Method
同 Arm S（见 exp2_settlement EXPERIMENT.md Method 节），唯一差异 reward_model=potential。

## Success Criteria（预注册，启动前）
1. **格式**: rl/format_compliance ≥ 0.95 达 ≥45/50 epochs，不触发中止；期望 ≥0.99。
2. **主判据**: 竞技场 S vs P（64 副复式 × 双方向，配对差分 95% CI）— 与 S 臂共享同一判据。
3. **副判据**: P vs SFT 锚点 64 副。
4. **风格探针**: 立直/副露/和牌/放铳率 vs SFT 锚点；与 exp1 PBRS 臂对照（H3）。
5. **Checkpoint 规则**: top-3 照旧；竞技场用最终 epoch checkpoint（与 S 臂对齐）。

## Progress
- [2026-08-07 06:57 UTC] Launched on flex-start VM mahjong-flex-p（us-east1-b）。
- [2026-08-08 10:27 UTC] 50/50 epochs 完成，exit 0；GCS 归档（50 份 rollout 日志核对无缺）后自动关机，实例已删除。
- [2026-08-08 ~10:55 UTC] 竞技场三连启动（exp2_arena_20260808_1050）。

## Results
(pending)

## Conclusion
(pending)

## Next Steps
- 同 exp2_settlement：竞技场 ×3 → 报告 → 依结果走 reward 默认切换或 critic 路线。

## Artifacts
| Path | Description |
|---|---|
| config_launch.json / config.json | 生效配置快照 |
| tensorboard/ | rl/* 全量（含 rl/ref_kl） |
| checkpoint_epoch_N/ | LoRA adapters, top-3 + latest |
| mahjong_epoch_N_rollouts.txt | 全量 rollout 转录 |
| gpu_info.txt / pip_freeze.txt / TRAIN_EXIT | provenance |
| gs://llm-mahjong-experiments/exp2_pbrs_20260807_065729/ | 全镜像含 train_nohup.log |
