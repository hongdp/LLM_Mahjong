# exp4_critic（预注册；run dir 由 VM 生成后挂靠）

- **Date**: 2026-08-09 预注册  **Status**: pre-registered, pending smoke + launch
- **类型**: 主线实验。证据链：方差分解（起手 Φ₀ 仅解释结算方差 2%）+ exp2 竞技场 CI ±1500-1800 → 信用分配方差是瓶颈，中盘状态价值是缺失项。用户批准的加速方案 #5。
- **设计**: `ValueHead`（2 层 MLP，fp32）读策略主干**最后一个 prompt token 的隐状态**（detach，价值损失不回传主干/LoRA）；V 以 MSE 拟合 return-to-go；优势改为 `normalize(G − V(s))`。PPO minibatch 内联训练（value_coef 0.5，value_lr 1e-4 独立参数组）。每 epoch 更新前一次 no_grad 价值前向出全样本 V(s)。
- **对照臂 = exp2_settlement_20260807_070806（零新增成本）**: 除 `use_critic` 外配置逐键相同（settlement reward、PPO ε0.2/3passes/KL0.03、ref_kl 0.05、γ0.995、50×12、batch 4、seed 42、同 SFT 锚点、同引擎、infra rev4+ref缓存——缓存不改变梯度）。
- **新机制**: `--checkpoint_every 10` 永久保留 ep10/20/30/40/50 → 赛后阶梯竞技场（#5.2 锚定竞技场的后验等价实现，零训练侵入）。
- **Success Criteria（预注册，启动前）**:
  1. **价值学习成立（门槛）**: rl/explained_variance 末 10 epochs 均值 > 0.2（若 ≤0.05：隐状态看不穿中盘价值 → 结论转向 v3 富上下文，实验仍有效）。
  2. **方差降幅（机制判据）**: rl/adv_std_ratio（残差std/原始std）末 10 均值 < 0.9（≥10% 降幅）。
  3. **强度（主判据）**: 竞技场 critic-ep50 vs exp2-S-ep50，64 副复式 × 双方向；95% CI 不含 0 → critic 判胜/负；含 0 → 无差异。
  4. **阶梯赛**: ep10/20/30/40/50 各 vs SFT 锚点 32 副 → 强度轨迹（本项目首次训练过程强度曲线）。
  5. 格式 ≥0.95 达 45/50（沿用）。
- **Env**: flex-start a2-highgpu-1g；预计 ~22h（ref 缓存 −15% vs exp2 的 27h）≈ $45。
- **Results sink**: `gs://llm-mahjong-experiments/exp4_critic_*/`。

## Progress
- [2026-08-09] 实现完成（src/core/value_head.py + trainer 集成，6+47 测试绿）；4080 冒烟中。

## Results
(pending)

## Conclusion
(pending)
