# exp69 待张辅助预测头（纯血线）：显式学"读线索→对手待张"能否让防守涌现

- **Date**: 2026-09-06  **Status**: running（实现 + 冒烟）
- **Git**: 见 Progress；RunPod Secure L40S；纯血无核心资产
- **Env**: `train_dnn_ppo.py --arch cnn_m_r_aux --critic_feats oracle --aux_waits_coef c`；配方 = exp27-A 钉死旗标（exp68 第 3 轮复现过）

## Purpose & Hypothesis
exp68 三轮：输入侧 oracle guiding 只加速学习，防守不迁移（defense_iq ≈0，放铳率与对照相同）。本实验换机制：**不给策略看暗手，
而是要求策略从公开观测预测三家隐藏待张（3×34 多标签，引擎标签）**，辅助 BCE 与 PPO 联合训练、共享 trunk。
假设：显式的"读线索"任务把对手听牌/待张的表示学进 trunk，弃张决策自然利用它 → 暴露席放铳率下降、defense_iq > 0、对 P 头对头为正。
反假设：辅助任务与策略梯度争抢容量，或"知道待张"与"选择弃张"之间还差价值判断，防守仍不出现。
纯度：标签来自引擎（自对弈特权），推理时不需任何隐藏信息；零人类知识。

## Method
两臂同批、同种子、同旗标（`--ppo_epochs 1 --adv_clamp 5.0 --target_kl 0.03 --batch 4096 --games_per_iter 2048 --lr 1e-4
--entropy_schedule 0:0.03,600000:0.01 --dup_k 8`，100 万局）：
- **A 臂**：`cnn_m_r_aux --critic_feats oracle --aux_waits_coef 0.5`（oracle 向量只作为目标；critic_feat_dim=0 → value 路径不读它）；
- **P 臂**：`cnn_m_r`（= exp68 P3 配方；P3 结果可作为第二个对照读数）。
诊断：aux_bce、aux_sep（正负标签平均预测概率之差；目标 >0.2 表示头真的在预测）；里程碑 defense 探针（60 万局）。
终评：A vs P、A vs exp27A、A vs bc49（双牌山段 20k 对）；defense_iq + 暴露席放铳率；风格向量。

## Success Criteria（预注册）
1. 辅助头有效：aux_sep ≥ 0.2（否则任务没学会，后续结论作废）。
2. **防守**：A 暴露席放铳率 ≤ P − 0.03 **且** defense_iq ≥ 0.08（exp27A 0.055、bc49 0.209）。
3. **强度**：A vs P ≥ 0.53 → 纯血新配方候选（走纯血谱系加冕检查：A vs exp27A ≥ 0.53）。
- 中间档：辅助头有效、放铳下降但强度平 → 防守表示学到了但进攻退化；调 coef。
- 预算 ≈ 4h ≈ $4.5，上限 $8。

## Progress
- [09-06 07:10] git 7c10a81：`cnn_m_r_aux` / `--aux_waits_coef`（带按批正样本权重，否则头塌缩为全 0：冒烟 BCE 0.01、sep 0）；
  测试 5/5；冒烟 2,560 局 aux_sep 0 → 0.02。pod `aoctz6js8u2iue` Secure L40S，A 70 局/s、P 77 局/s，两臂并行。
- [09-06 09:48] **60 万局**：辅助头 aux_sep **0.39–0.40**（判据 1 ≥0.2 ✅，12 万局起即稳定），aux_bce 0.37–0.40（带权）。
  防守探针：A600 defense_iq −0.054、暴露席放铳 0.210；P600 −0.020 / 0.222。A 熵 0.94 vs P 0.68（A 62 万局，P 74 万局；辅助损失在争容量）。

## Results

## Conclusion

## Next Steps

## Artifacts
