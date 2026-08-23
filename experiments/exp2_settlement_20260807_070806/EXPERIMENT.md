# exp2_settlement_20260807_070806

- **Date**: 2026-08-07 启动  **Status**: complete（竞技场三场全 null，2026-08-09 判定；见 Conclusion）
- **Series**: exp2_settlement_vs_pbrs, Arm S（本臂）。对照臂：exp2_pbrs（另一台 VM，同日启动）。
- **单变量设计**: 两臂唯一差异 = `reward_model`（S: `settlement`，P: `potential`）。其余完全一致：PPO(ε0.2/3passes/target_kl 0.03) + **ref-KL 锚 k3, coef 0.05**（exp1 新发现的 PPO 格式侵蚀 100%→97.5% 的对策）+ γ=0.995 + 50 epochs × 12 局并行 rollout + batch 4 + lr 1e-6 + seed 42 + 同一 SFT 锚点。
- **动机（exp1 报告 docs/report_exp1_shaping_arms_20260802.md）**: PBRS 的策略不变性是渐近保证；有限样本下稠密塑形通道支配学习方向 → 全舰队风格迁移（立直减半、副露 +75%）而竞技场强度零显著增益。本实验检验：去掉稠密通道、只留诚实的稀疏结算信号，PPO 能否学到竞技场可测的强度。
- **Reward（Arm S）**: `MahjongSettlementOnly` — 中间步 0，仅保留格式 −10 / 幽灵牌 −5 约束项；终局结算（真实点数变动 + 顺位 ±2/±0.5）由 finalize 分配到四条轨迹。
- **γ=0.995 理由**: 稀疏终局信号需要更长的信用分配视野（平均 ~15 巡 × 4 步噪声，0.99^60≈0.55 过度衰减首巡；0.995^60≈0.74）。
- **引擎**: post-audit RCR 2016 重写（用户 commit 344b938：振听分离、岭上补牌、四杠、食替、多家荣和、立直供托、场况役、里宝、包牌）。**与 exp1 舰队（pre-audit 引擎）不可直接比较** — 规则可比性边界，两臂之间可比。
- **Infra**: rev3 — bf16 LoRA、fla+causal-conv1d 快速路径、12 局批式并行 rollout、原始 logits 行为对数概率。**首次 DWS flex-start 供给**（$2.020/h，−45%，DELETE-on-termination 临时机；docs/gcp_compute_cost_and_quota.md）。代码/锚点/内核包均从 GCS 自举。
- **Git**: ee0c60c
- **Env**: flex-start a2-highgpu-1g（A100 40GB），image common-cu129-ubuntu-2204-nvidia-580，torch 2.12.1+cu129，fla 0.5.2，cc1d 预编译包。
- **Input artifacts**: SFT 锚点 `v2_engine_full_run_20260802_005918/checkpoints_sft_warmup_mahjong`（GCS 拉取，adapter 同时作为 ref-KL 冻结参考）。RL-only（sft_epochs=0），不需要 SFT 语料。
- **Results sink**: `gs://llm-mahjong-experiments/exp2_settlement_20260807_070806/`；EXIT-trap 上传后 VM 自毁（flex DELETE）。
- **Cost plan**: flex-start $2.020/h × 预计 14–17h（rollout ~10min/epoch + PPO 更新含 ref 前向 ~5min/epoch）≈ $30–35。max-run-duration=36h 兜底。

## Purpose & Hypothesis
1. **H1（主）**: 仅结算奖励下，PPO 仍能获得竞技场可测强度增益 — 或至少不弱于 PBRS 臂。判据是竞技场，不是 self-play 奖励曲线。
2. **H2**: ref-KL 锚（coef 0.05）把 rl/format_compliance 钉在 ≥0.99，消除 exp1 PPO 臂的漂移（BASE/REINFORCE 无漂移、PPO 漂到 97.5% 的对比）。
3. **H3（风格）**: 无稠密塑形 → 不出现 exp1 的风格迁移（立直率、副露率相对 SFT 锚点变化 < exp1 幅度的一半）。
4. **诚实度量声明**: self-play 结算零和 ⇒ Arm S 的 rl/avg_episode_reward 期望 ≈ 0（仅约束项贡献负偏），**不构成学习信号判据**；本臂学习信号只能由竞技场判定。

## Method
RL-only PPO：50 epochs × 12 并行自对弈局；行为对数概率取自 raw 预 warp logits；PPO ≤3 passes，approx-KL 0.03 早停；每个 minibatch 额外一次 no_grad ref 前向（adapter "ref" = 冻结 SFT 锚点），loss += 0.05·k3(new‖ref)。温度 0.9/top_p 0.95。协变量基线关闭（方差分解显示上限 2%，两臂一致关闭保持单变量）。

## Success Criteria（预注册，启动前）
1. **格式**: rl/format_compliance ≥ 0.95 达 ≥45/50 epochs，且不触发 3-epoch 中止；ref-KL 目标下期望 ≥0.99。
2. **主判据（强度）**: 竞技场 S vs P，64 副复式对局牌 × 双方向，配对差分 95% CI 不含 0 → 判定胜方；含 0 → 无差异结论（也是有效结论）。
3. **副判据**: S vs SFT 锚点、P vs SFT 锚点各 64 副（exp1 为 null，重现或推翻均记录）。
4. **风格探针**: 立直率/副露率/和牌率/放铳率 per 臂 vs SFT 锚点基线（H3 判据）。
5. **Checkpoint 规则**: top-3 保留照旧；**竞技场用最终 epoch checkpoint**（Arm S 的 avg_episode_reward 零和无意义，"best-by-reward" 不适用）。

## Progress
- [2026-08-07 07:08 UTC] Launched on flex-start VM mahjong-flex-s（us-central1-b）。flex 队列几乎即时给容量。
- [2026-08-08 09:33 UTC] 50/50 epochs 完成，exit 0；GCS 全量归档后 VM 自动关机，实例已删除。
- [2026-08-08 ~10:55 UTC] 竞技场三连在 mahjong-flex-a 启动（exp2_arena_20260808_1050）。

## Results（训练侧；竞技场 pending）
| 预注册判据 | 结果 |
|---|---|
| #1 格式 ≥0.95 达 45/50 | ✅ **50/50 epochs**，min 99.89%，last10 均值 100%；H2（ref-KL 钉 ≥0.99）成立 |
| #4 风格探针 | 初值（粗计，末 5 epochs 60 局）：riichi 453 / meld 590 / 和牌局占比 ~72%；精算入总报告 |
| #2/#3 竞技场 | pending（exp2_arena_20260808_1050） |

训练动力学：rl/approx_kl ~0.0007 恒低（PPO 3 passes 从未早停）；rl/clip_frac ~0.3%；**rl/ref_kl 末 10 epochs 均值 0.023（max 0.030）** —— 对照 P 臂 0.102（5×），稀疏结算下策略贴着参考走。avg_episode_reward 全程 ≈0（−0.30→−0.13，零和 + 少量约束罚），与预注册的「非学习信号」声明一致。

## Conclusion
竞技场判定（预注册判据）：S vs P 无显著差异（−183±1591），S vs 锚点无显著差异（−27±1480）。
H1 部分成立（不弱于 PBRS）但无绝对强度增益；H2 成立（格式 100%）；H3 证伪——S 臂在稀疏信号下
反而长出极端副露风格（6.02 副露/局 vs P 臂 1.90），且 token 级 ref-KL 极低（0.023）说明该锚
不约束行为分布。完整分析见 docs/report_exp2_settlement_vs_pbrs_20260809.md。结论：奖励设计
不是当前瓶颈，转投 critic value head。

## Next Steps
- 竞技场 ×3（S vs P 主判据；各 vs 锚点）— flex 机自毁后另起短时 GPU 跑。
- 若 S 显著胜 P：settlement-only 成为默认 reward，PBRS 降级为 curriculum 工具。
- 若均 null：印证方差瓶颈 → 转投 critic value head（TASKS.md 首位）。

## Artifacts
| Path | Description |
|---|---|
| config_launch.json / config.json | 生效配置快照 |
| tensorboard/ | rl/loss, rl/avg_episode_reward, rl/format_compliance, rl/approx_kl, rl/clip_frac, rl/ppo_passes, **rl/ref_kl** |
| checkpoint_epoch_N/ | LoRA adapters, top-3 + latest |
| mahjong_epoch_N_rollouts.txt | 全量 rollout 转录（含 [SETTLEMENT] 行） |
| gpu_info.txt / pip_freeze.txt / TRAIN_EXIT | provenance |
| gs://llm-mahjong-experiments/exp2_settlement_20260807_070806/ | 全镜像含 train_nohup.log |
