# exp2_smoke_20260806_232343

- **Date**: 2026-08-06  **Status**: complete（基建冒烟，非正式实验；本文件为完成后补记）
- **目的**: 发射 exp2_settlement_vs_pbrs 前，在本地 4080 端到端验证新代码路径：`MahjongSettlementOnly` reward + PPO + **ref-KL 锚（k3, coef 0.05, 冻结 SFT 参考 adapter）** + **γ=0.995 配置化**（buffer + reward model 双侧）+ 并行 rollout + checkpoint 保存。
- **配置**: configs/exp2_smoke.json — 1 epoch × 4 局并行，batch_size 1（首次 batch 2 在 PPO 更新阶段 OOM：bf16 [B,T,248k] logits 超 16GB；A100 40GB 无此约束，云配置保持 batch 4），锚点用本地 v2_smoke adapter。
- **结果**: 退出码 0。rl/ref_kl=0.0016（ref 前向真实执行）、rl/ppo_passes=3、rl/approx_kl=0.0019、format 100%（352/352）、rl/avg_episode_reward=−0.625（4 局全流局 → 结算零和归零，仅约束项小负；符合 settlement 模式预期）、checkpoint_epoch_1 落盘。
- **结论**: exp2 全代码路径可发射。冒烟产物不复用（教训沿袭 config_test_run）。
- **教训**: ①16GB 卡跑 bf16 PPO 更新需 batch 1；②`conda run` 缓冲 stdout，实时监控要看 exp dir 内文件而非管道输出；③后台命令 `... | grep` 会把管道退出码顶替训练退出码（同 `| tail` 教训）——判定成败必须显式 echo `$?`。
