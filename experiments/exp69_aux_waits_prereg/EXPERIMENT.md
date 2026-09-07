# exp69 待张辅助预测头（纯血线）：显式学"读线索→对手待张"能否让防守涌现

- **Date**: 2026-09-06  **Status**: done（判负：感知不是瓶颈）
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

- [09-06 11:10] **两臂完成**：各 489 迭代 / 100 万局，A 235 分钟（70 局/s）、P 205 分钟（81 局/s）；pod terminate，≈4h ≈ **$4.4**。
  A 终局 aux_sep **0.442**、aux_bce 0.354；熵 A 0.657 vs P 0.593；迭代胜率 0.804 vs 0.823。

## Results

| 判据 | 目标 | 实测 | 判定 |
|---|---|---|---|
| 1 辅助头有效 aux_sep | ≥ 0.2 | **0.44**（12 万局起 ≥0.37，单调升） | ✅ 策略 trunk 里确实有了"对手待张"的表示 |
| 2 防守：暴露席放铳率 A vs P | ≤ P − 0.03 | **A 0.212 vs P 0.200**（exp27A 0.156） | ❌ 反而略高 |
| 2 防守：defense_iq | ≥ 0.08 | A −0.025、P 0.000、exp27A 0.055 | ❌ |
| 3 强度 A vs P（双段合并 20k 对） | ≥ 0.53 | 67M 0.4828、68M 0.4965 → **0.4938±0.0035（−1.8σ）** | ❌ 略负 |
| 强度 A / P vs exp27A | — | 0.4892 / **0.4974**（P 再次复现冠军配方） | — |
| A / P vs bc49 | — | 0.3086 / 0.3180 | — |
| A/A | — | 0.500 | ✅ |

## Conclusion
**反假设成立：感知不是防守缺失的瓶颈。** 辅助头把"从公开线索预测对手待张"学得很好（正负分离 0.44，与 exp68 里直接看暗手的差距不大），
但策略并没有用这份表示去弃张：暴露席放铳率与对照相同（0.212 vs 0.200），强度还略降（辅助损失争容量，熵更高、学得更慢）。
结合 exp68（看得见暗手时也只是"不打进听牌家"、退火后消失），从零 RL 缺的不是"知道对手要什么"，而是**在当前奖励下弃张不划算**：
单局终局点差里，放铳的代价（几千分）被自摸/荣和的机会和 σ≈5,000 的运气噪声淹没，四家同水平自对弈的均衡就是"都不防"。
人类先验模型的防守来自人类数据里已经内化的（多局顺位）代价结构，不是来自更好的感知。

## Next Steps
- 纯血线在"单局 + 终局点差"设定下的三条感知/课程杠杆（oracle 输入、oracle critic、待张辅助头）全部关闭；配方钉死后 100 万局的平台 = exp27A。
- **剩余的结构性杠杆只有改设定**：半庄级 episode + 顺位（uma）奖励从零训练（`play_hanchan_gen` 与 `--hanchan` 路径已存在，
  exp55-D 在人类先验线验证过管线；从零未跑过）。放铳在半庄顺位刻度下的代价大得多，这是让防守"划算"的唯一纯血手段。需用户批准（设定变更 + 更长 horizon 的方差）。
- 纯血冠军：G3/P3/P（exp69）三个 100 万局产物都与 exp27A 打平（0.49–0.50），按"打平不换冠军"不加冕。

## Artifacts
| Path | Size | Description |
|---|---|---|
| experiments/exp69_{A,P}/ + gs://llm-mahjong-experiments/exp69_{A,P}/ | 12 ckpt 各 | 里程碑 ckpt、latest.pt、train_log.json（含 aux_bce/aux_sep）、TB |
| experiments/exp69_pull/*.log | — | pod 日志 |
| experiments/probes/exp69_arms.json、exp69_defense_600k.json、exp69_defense_final.json | — | 终评与防守探针 |
| arch_zoo `cnn_m_r_aux`、train_dnn_ppo `--aux_waits_coef`、tests/test_aux_waits.py | — | 基建 |
