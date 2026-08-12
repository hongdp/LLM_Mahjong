# exp4_critic_20260810_062927

- **Date**: 2026-08-09 预注册  **Status**: complete（训练 50/50 exit 0；竞技场 6 场 exit 0，2026-08-11）。第二次执行（run-1 因宿主机事件+DELETE 全损，见 exp4_critic_20260809_103720 事故记录）。mahjong-flex-c4b，code 51c0e48（STOP 终止 + 10min 增量同步已验证生效）
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

## Results（训练侧；竞技场 pending）
| 预注册判据 | 结果 | 判定 |
|---|---|---|
| 1. explained_variance 末10 > 0.2（门槛） | **0.0202**（分段均值 0.022/0.033/0.026/0.026/0.020，全程平坦无上升趋势；单 epoch 峰值 0.136 为波动） | ❌ 不达标 |
| 2. adv_std_ratio 末10 < 0.9 | **0.9898**（方差仅降 1%） | ❌ 不达标 |
| 5. 格式 ≥0.95 达 45/50 | 99.96%（末10） | ✅ |
| 3. 竞技场主判据 critic vs exp2-S | +634±1231（z=1.01, p=0.31） | 无差异 |
| 4. 阶梯轨迹 vs 锚点 | ep10 +600 / ep20 +1566 / ep30 +278 / **ep40 +2950 (p=0.0027, Bonferroni 后仍显著)** / ep50 +2147 (p=0.021, 校正后不成立) | ✅ 首个显著强度增益 |

其它动力学：rl/value_loss 全程平在 13.6（训练中 in-sample）；rl/ref_kl 0.0066（极低——优势与 exp2-S 几乎相同，策略未被 critic 推离）；avg_episode_reward ≈ −0.22（零和，按设计非信号）。

**关键疑点（已立即追查，非事后猜测）**：训练中 value MSE 13.6 对应 Var(G)≈16-28，隐含 in-sample R² 约 0.15-0.5；而在**下一 epoch 新样本**上测得的 explained_variance 只有 0.02。这种 in-sample 好、out-of-sample 塌的落差，指向两种完全不同的诊断——(A) 特征本身没有可泛化的价值信号（需 v3 富上下文），(B) 价值头在 2048 维特征上过拟合（正则化/降 lr 可救）。已写 `scripts/probe_value_decodability.py`（新 rollout + **按对局切分**的 ridge 拟合；步级切分会因同局共享终局结算而泄漏标签）在竞技场同机并行执行，用 ridge 测试集 R² 判定：<0.10 → 诊断 A。

## Conclusion
(pending)

## Conclusion
**critic 失败，但实验产出了更重要的东西。**

1. 机制判据双双不达标（EV 0.020 vs 门槛 0.2；方差降 1% vs 门槛 10%）→ 按预注册走「诚实负分支」。
2. 主判据 critic vs exp2-S 为 null（+634±1231），与 critic 只改动 1% 优势、ref_kl 仅 0.0066 一致：两个模型实质同分布。
3. **阶梯竞技场（本 run 首次引入）测到项目首个显著强度增益：ep40 vs SFT 锚点 +2950 点，p=0.0027，6 场 Bonferroni 校正后仍成立。** 归因应为 settlement+PPO+ref-KL 配置本身，非 critic。
4. 加权趋势检验斜率 +44.3 点/epoch 但 p=0.14 —— **只主张 ep40 这一点显著，不主张强度随训练单调上升**。
5. 传递性核查（exp4 vs S、S vs 锚、exp4 vs 锚）差 1.14σ，噪声可解释。
6. **方法论收获（最有价值）**：exp2-S 当时只测 ep50 得 null——「只看终点」会漏掉训练中途峰值。阶梯竞技场应成为所有 RL run 的标准收尾。

完整分析：docs/report_exp4_critic_20260811.md
