# exp67 Oracle critic：让 critic 看见隐藏信息，PPO 从 bc65 起步能否第一次越过 BC 水位

- **Date**: 2026-09-05  **Status**: running（实现 + 本机冒烟阶段）
- **Git**: 见 Progress；训练上 RunPod Secure（冠军谱系 bc65）；评测工作站双牌山段
- **Env**: `train_dnn_ppo.py --critic_feats oracle --arch convformer_m_v3r_m46_oc`（本实验新增）；引擎指纹不变（规则审计见 docs/engine_known_issues.md 2026-09-05）

## Purpose & Hypothesis
文献对照（docs/literature_review_2026-09.md §2/§4）：所有在麻将里越过模仿水位的系统都让 critic 看见隐藏状态
（Suphx oracle guiding、RVR 相对价值网 +4.9% 胜率、PerfectDou 完美信息 critic），并用 GRP/配对复式压方差。
本项目 exp46/59/60 的 RL 全部"能学不毁值不超越"，exp66 量出病根：单局回报 σ≈3,700 分，动作价值差几十到几百分，
(σ/Δ)² ≈ 10²–10³。**假设**：把对手暗手、牌山构成、隐藏宝牌/里宝喂给 value head（策略不看），critic 的解释方差大幅上升、
优势估计噪声下降，PPO（bc65 init + KL 锚）在同样局数下对同批"盲 critic"控制臂显著为正，并首次对 bc65 本体为正。
**反假设**：方差瓶颈不在 critic 而在策略梯度本身（每决策 1 个样本），oracle critic 只让 value loss 好看，强度仍平。

## Method
1. **oracle 特征（critic 专用，319 维）**：三家对手暗手计数、赤五、听牌待张、向听；牌山 34 类计数与长度；下一张杠宝牌指示牌、
   首张里宝牌指示牌（`src/agents/dnn/oracle_features.py`）。策略 logits 路径与 bc65 逐位一致（测试保证）。
2. **网络**：`convformer_m_v3r_m46_oc` = 同 trunk/头 + value 路径拼接 oracle 投影（policy 键从 bc65 加载，value 头重新初始化）。
3. **训练配方**（沿用 exp46I 的最强 RL 配方，唯一变量 = critic 信息）：PPO，`--init bc65 --bc_anchor bc65 --bc_kl_coef 0.3
   --value_detach --entropy_coef 0 --dup_k 8`（配对牌山组基线 = 控制变量）、`--value_warmup`（先只训 critic）、league 对手池含 bc65/bc49。
   - **O 臂**：`--critic_feats oracle`；**C 臂**：`--critic_feats none`（同批、同种子、同局数）。
4. **诊断指标**（TB）：value loss、critic 解释方差 EV = 1 − Var(R−V)/Var(R)（两臂对比，O 应显著高）、优势方差、KL 到锚、熵。
5. **终评**（工作站）：O vs C、O vs bc65、C vs bc65，各两个牌山段（67M n=4000 + 68M n=16000，若 z>2 再补到 50k/段）；A/A。

## Config（预注册）
- 局数：与 exp46I 同量级 1.0M 局/臂（RunPod L40S 13 核演员+学习者同 pod ≈ 65 局/s → 两臂并行 ≈ 8.5h ≈ **$9–10**，上限 $15）。
  先本机冒烟（2k 局）验证管线与 EV 指标。
- value_warmup：前 20% 更新只训 critic。

## Success Criteria（预注册）
1. **诊断**：O 臂 critic EV ≥ C 臂 + 0.2（否则 oracle 特征没被用上，后续结论无效）。
2. **主判据**：O vs C 双段合并 ≥ 0.51（z ≥ 2.5 在 20k 对 SE 0.0035 下 = 0.509）**且** O vs bc65 ≥ 0.51 → 首次 RL 越过 BC 水位，
   下一步扩局数 + 演员多 pod（exp60 设计）。
3. 0.50–0.51：critic 方差不是瓶颈；记录 EV 增益与强度零增益的组合 → 病根转向策略梯度样本数（每决策 1 样本），
   剩余方案只有 exp60 式大规模演员或搜索（需 oracle critic 本身作为价值网 —— 本实验产物可复用）。
4. < 0.495：RL 毁值，检查锚系数。

## Progress

## Results

## Conclusion

## Next Steps

## Artifacts
