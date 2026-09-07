# exp72 决策级反事实 rollout 优势（纯血线 $200 计划阶段 B-1，2026-09-07）

- **Date**: 2026-09-07  **Status**: running（实现 + 冒烟 + 发射）
- **Git**: 见 Progress；RunPod Secure L40S 单 pod 双臂；纯血：零人类数据、零外部模型；先知信息（真牌山/暗手）只用于构造优势，不进策略输入
- **Env**: `train_dnn_ppo.py --cf_p/--cf_k/--cf_all/--cf_slots/--cf_scale`（本实验新增）；`parallel_rollout._worker_vectorized` 分支 rollout

## Purpose & Hypothesis
纯血线剩余差距被诊断为**归因信噪比**（终局回报 σ≈5,000 分 vs 单步决策价值差几百分）与**目标**（exp70 在测）。本实验测前者：
在采样到的学习者弃牌决策上克隆牌桌（真牌山、真暗手），把 K 个替代弃牌各自用当前策略打到局末，**同牌山配对差**
A_cf = R(执行动作) − mean R(替代) 替换该步的 GAE 优势（基线独立于执行动作 ⇒ 无偏）。设计见 `designs/design_offpolicy_data_and_counterfactual.md` §3。
- **C1（防守靶向）**：只在"场上有对手立直"的弃牌决策上做，p=1.0，替代 = 1 张现物（对立直家安全）+ 1 张随机合法弃牌。
- **C2（通用方差降低）**：所有弃牌决策 p=0.02，替代 2 张随机（有立直时含现物）。
**假设 H1**：C1 或 C2 同预算（100 万局）T=0 对 P3 ≥ 0.53；C1 的暴露席放铳率 ≤ 0.17、defense_iq ≥ 0.08（防守首次在单局目标下涌现）。
**反假设 H0**：≤ 0.50 且防守指标不变 ⇒ 单局目标下反事实评估诚实地算出"弃张不划算"（与 exp46/63 推论一致），信噪比不是瓶颈，剩余=目标/样本效率。
**副产物**：`cf_fold_gain_mean`（现物续打 − 执行续打，同牌山）= 弃张价值的直接无偏估计，随训练的轨迹本身就是结论的一部分。

## Method
- 配方 = P3 钉死旗标（`--ppo_epochs 1 --adv_clamp 5.0 --target_kl 0.03 --batch 4096 --lr 1e-4 --entropy_schedule 0:0.03,600000:0.01 --gamma 0.995 --gae_lambda 0.95 --dup_k 8 --games_per_iter 2048`），cnn_m_r，随机初始化，100 万局。
- 分支：`play_game_gen(table=deepcopy)` 从决策点续打，首动作强制为替代弃牌，其余席位同温度；分支不产 episode，只回该席回报（含终局结算）。
  分支占独立槽位（`--cf_slots 64`），槽满则跳过并计数（cf_skipped，随机选择效应与决策内容无关）。
- 本机测量（P3@30 万局权重，128 局）：立直 0.04 席/局（纯血模型 T=1 下几乎不立直）→ C1 可评估决策 ≈0.4–0.5/局，rollout 开销 **+15%**；
  C2 预计 ≈1.2 决策/局 × 2 分支 × ~42 步 ⇒ rollout ≈2.2×。双臂同 pod 预计 8–9 h。
- 对照：P3（同配方、同预算，已知三次独立读数 P3/G3/exp27A，噪声地板 ±0.5–0.9%）。
- 终评（工作站）：C1/C2 vs P3、vs exp27A 双段 20k 对；防守探针 1,600 局（含 P3 同批）；风格向量；cf_fold_gain 轨迹图。

## Success Criteria（预注册）
1. **主判据**：C1 或 C2 vs P3 合并 share ≥ **0.53**（z≥8）→ 杠杆成立，进入阶段 C 组合；0.50–0.53 中性；<0.50 判负。
2. **防守**（C1）：暴露席放铳 ≤ 0.17（P3 0.195）且 defense_iq ≥ 0.08（P3 0.024）。
3. **诊断**：cf_fold_gain_mean 的训练轨迹：若始终 >0（弃张更好）而策略不学 ⇒ 优势尺度/归一化问题；若 →0 ⇒ 策略已内化；若 <0 ⇒ 单局目标下弃张确实不划算。
4. 在轨：每个 ckpt 对同局数 P3 ckpt T=0 n=600（`exp72_track_loop`），任一臂 <0.35 持续三个点 ⇒ 早停该臂。
- 预算：双臂同 pod ≈ 8–9 h ≈ **$9–10，上限 $12**（$200 计划阶段 B）。

## Progress
- [09-07 00:40 PDT] 实现：`parallel_rollout` 分支 rollout（deepcopy 0.46 ms/次，槽位预算，forced first action，回合内新分支不接收回复）、
  `collect_parallel` 按席附加 `cf_adv`/`cf_fold_gain`（用组基线前的原始回报）、训练器旗标/优势替换/日志（cf_n, cf_adv_mean, cf_exec_best_frac, cf_fold_n, cf_fold_gain_mean, cf_skipped）。
  `tests/test_cf_rollout.py` 2/2（分支有限优势；T=0 下父局轨迹不受分支影响 ≥90%）。本机冒烟 512 局 ×2（cf_p 0/1）：rollout 4.6–4.9 s → 5.3–5.6 s。

## Results

## Conclusion

## Next Steps

## Artifacts
