# exp28 — 熵系数 × 采样温度消融：学习者尖锐、数据多样（预注册）

- **Date**: 预注册 2026-08-23 03:20 PDT  **Status**: planned（与 exp27 同批发射，共享基线臂 A）
- **Git**: 分支 `engine/majsoul-rules`：行为策略 logprob（`net.act` / 服务端 `log softmax(logits/T)`）、
  `--rollout_temps`（每局每座位从列表抽 T，dup_k 副本各异）；`tests/test_behaviour_logprob.py`
- **Env**: GCP g2-standard-32（L4），统一 GPU rollout；纪元 3 引擎；架构 `cnn_m_r`

## Purpose & Hypothesis
冠军谱系的熵系数是绑定的（0.03 段策略熵卡在 ~1.0 nat；切 0.01 立刻掉到 0.77，1.4M 时 0.62 仍在降），
而 exp25 证明同一 checkpoint 贪心比 T=1 采样强一个代际量级——剩余随机性在实战是纯损失，在训练里是
"带噪声的自己"生成的数据。用户问题：能否降熵系数、靠采样温度（混合温度）保持探索？
- H1（低熵有益）：熵计划 0:0.03 → 400k:0.01 → 700k:0.003 的 A″，T=0 Elo ≥ A，且"贪心−采样差距"收窄。
- H2（混合温度补探索）：A′ = A″ + `--rollout_temps 0.7,1.0,1.3`（行为 logprob 修正，比率 π/b），
  T=0 Elo ≥ A″（多样性不靠熵奖励也能保住，甚至更好）。
- H0（防守）：熵/温度都不改变 defense_iq——防守动作本就被充分采样（fold≈0.20 与手牌无关），缺的是信用。
同时修复了一个潜在偏差：此前 T≠1 时记录的是 π(T=1) 的 logprob，PPO 比率分母错误；本次起记录行为策略 b 的 logprob。

## Method
三臂同配方（exp17-C 配方 + v1r + 374 头，从零，1.0M 局）：
| 臂 | 熵计划 | rollout 温度 | 备注 |
|---|---|---|---|
| A（共享 exp27 基线） | 0:0.03 → 600k:0.01 | 1.0 | 现配方 |
| A″ | 0:0.03 → 400k:0.01 → 700k:0.003 | 1.0 | 仅低熵 |
| A′ | 同 A″ | {0.7, 1.0, 1.3} 每座位抽 | 低熵 + 混合温度 |
评估：每臂 T=1 与 T=0 各评一次 Elo（纪元 3 锚点；"贪心−采样差距" = 两者之差）、defense_iq、风格、
A′/A″ vs A 复式牌 1000 副（T=0）。

## Config
`--arch cnn_m_r --total_games 1000000 --entropy_schedule "0:0.03,400000:0.01,700000:0.003" [--rollout_temps 0.7,1.0,1.3] --gpu_infer --infer_max_batch 128`

## Success Criteria
1. H1：A″ T=0 Elo ≥ A T=0 Elo − 10，且 A″ 的贪心−采样差距 < A 的一半。
2. H2：A′ T=0 Elo ≥ A″ + 15（CI 不重叠）⇒ 混合温度进入配方；若 A′ < A″ − 15 ⇒ 重要性权重方差代价过大，弃。
3. H0：三臂 defense_iq 差 < 0.01（预期）。
4. 健康：A′ 的 approx_kl 与 clip 比例无异常（比率 π/b 在 T 0.7–1.3 内应温和）。

## Progress
- [03:20] 预注册；行为 logprob + 混合温度本地冒烟（GPU 基建）通过。

## Results
| 臂 | Elo T=1 | Elo T=0 | 差距 | defense_iq | 熵(终) |
|---|---|---|---|---|---|

## Conclusion
（待）

## Next Steps
（待）
