# exp70 半庄顺位奖励从零（纯血线设定变更，用户 2026-09-06 批准）

- **Date**: 2026-09-06  **Status**: running（实现 + 冒烟）
- **Git**: 见 Progress；RunPod Secure L40S；纯血无核心资产、零人类数据（W 不用 exp55-D 的学习残差）
- **Env**: `train_dnn_ppo.py --hanchan_pure --hanchan_credit rank --arch cnn_m_rh`（本实验新增）；引擎字节不变（半庄层是驱动侧）

## Purpose & Hypothesis
exp68/69 证明从零 RL 的防守缺失不是感知问题：看得见暗手、预测得出待张都不多弃一张牌。诊断是"单局终局点差"设定下弃张不划算
（放铳几千分 ≪ 运气 σ≈5,000，四家同水平的均衡 = 都不防）。人类先验模型的防守来自人类数据内化的**多局顺位代价**。
本实验把 episode 改为整个半庄（E1–S4，连庄/本场/供托/飞），奖励 = 终局顺位 uma（±15k/±5k）+ 点差，每局按**解析的顺位势函数**
（当前名次 uma + 自身点差，`RankUmaCredit`，无学习、无人类数据）做精确 telescoping 的逐局归因。
**假设**：顺位刻度下放铳的代价被放大（尤其领先时），防守首次在纯血线涌现——H 臂暴露席放铳率 ≤ 单局配方产物 − 0.03、defense_iq ≥ 0.08，
半庄头对头对 P3（单局冠军配方）显著为正。**反假设**：horizon 变长 + 方差更大 → 同预算下学得更慢，强度反而低于单局配方；
或顺位压力只改变风险偏好（领先时保守）而不改进弃张选择。

## Method
- 生成器：`play_hanchan_gen`（exp46-D/55-D 既有）+ 新 `RankUmaCredit`（纯解析 W）；四席全为学习者（`--hanchan_pure`，dup_k 复式同 match 种子 = 组基线）。
- 编码：`cnn_m_rh` = cnn_m_r + 3 个场况标量（局数/4、本场/8、all-last），平面不变；单局评测时这些标量由 `randomize_round` 表提供，语义一致。
- 配方 = 钉死的冠军旗标：`--ppo_epochs 1 --adv_clamp 5.0 --target_kl 0.03 --batch 4096 --lr 1e-4 --entropy_schedule 0:0.03,600000:0.01
  --gamma 0.995 --gae_lambda 0.95 --dup_k 8`；`games_per_iter 256` 场（≈2,500 局/迭代，与单局 2048 局/迭代同阶）。
- **H 臂**：`--hanchan_credit rank`；**H0 臂**（消融）：`--hanchan_credit none`（逐局裸点差 + 终局 uma 一次性给，γ=0.995 下跨局信号基本衰减）。
  预算 150k 场/臂（≈1.5M 局）。单局对照 = exp68r3-P3 / G3 / exp27A（三个独立读数，噪声地板已知）。
- 终评（工作站）：①半庄头对头 H vs P3、H vs bc49（`play_pair_vector(hanchan=True)` n≥600 场，两向复式）；②单局配对牌山 H vs P3 / bc49
  （双段 20k 对，同刻度可比）；③defense_iq + 暴露席放铳率；④风格向量 vs bc49；⑤梯子 Elo（单局 + 半庄池）。

## Success Criteria（预注册）
1. **防守涌现**：H 暴露席放铳率 ≤ 0.17（P3 0.196、exp27A 0.156、bc49 0.126）且 defense_iq ≥ 0.08。
2. **强度（半庄刻度，主判据）**：H vs P3 半庄头对头 ≥ 0.55（n≥600 场）；H vs bc49 半庄 ≥ 0.30（P3 单局 0.33 作参照）。
3. **强度（单局刻度）**：H vs P3 ≥ 0.50（不因设定变更而毁掉单局能力）。
4. H vs H0：rank 归因是否优于裸 uma（≥0.53 → 归因有效）。
- 反假设成立（H vs P3 单局 <0.47 且放铳率无改善）→ 半庄设定不是纯血线的出路，记录并停。
- 预算：两臂 ≈ 4–6h ≈ **$6，上限 $10**。

## Progress

## Results

## Conclusion

## Next Steps

## Artifacts
