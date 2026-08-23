# exp31（第二批）— 目标熵配方 × 规模复检（预注册）

- **Date**: 2026-08-23  **Status**: running（用户批准 ~$15 总预算，含 exp30）
- **Git**: 本提交（`--entropy_floor_schedule` 线性目标熵）；纪元 4 引擎 + 随机场上下文
- **Env**: g4-standard-48 flex ×4（跑完即删），git-clone 固定 SHA 发射

## Purpose & Hypothesis
exp28 表明硬编码降熵计划损伤贪心上限；exp27-C/B 表明大模型被冠军配方按住。本批检验：
- H1（目标熵优于时间表）：对偶控制把熵钉在 0.5→0.25 线性目标（臂 1）可以既收窄贪心−采样差距又不损贪心上限
  （对照 = exp27-A 的 1121.8 ± 14）。
- H2（训练期熵下限）：恒定目标 0.2（臂 2）若过早收敛（decisive 卡住 / KL→0 / Elo 平台低），给出下限实测。
- H3（规模是配方问题）：`cnn_xl_r`（臂 3）与 `handset_xl`（臂 4）改用恒定低熵系数 0.01——若 C 类塌缩消失、
  Elo 显著高于 862/1058，则"配方卡住"成立，exp15/18/27 的规模/架构结论需要按配方条件化重写。
观察项：起始顺位分桶的攻防分化（随机场上下文首批 1.0M 训练）。

## Method
四臂同基（exp17-C 配方，只动熵律），从零 1.0M 局，v1r/374，seed 42：
| 臂 | 架构 | 熵律 |
|---|---|---|
| 1 target_decay | cnn_m_r | `--entropy_auto --entropy_floor_schedule 0:0.5,1000000:0.25` |
| 2 target_02 | cnn_m_r | `--entropy_auto --entropy_floor 0.2` |
| 3 cnnxl_lowent | cnn_xl_r | `--entropy_coef 0.01`（恒定） |
| 4 handset_lowent | handset_xl_cnn_m_r | `--entropy_coef 0.01`（恒定） |

## Success Criteria
1. H1：臂 1 T=0 Elo ≥ 1111（A−10）且贪心−采样差距 ≤ 120（A 的一半）。
2. H2：臂 2 相对臂 1 的 Elo 差与训练健康信号明确可判。
3. H3：臂 3 ≥ 1000（vs 862）、臂 4 ≥ 1090（vs 1058，T=0 口径另评）即判"配方卡住"成立。
4. 发射清单：前 3 轮吞吐对照基准、GPU 利用率、KL/熵曲线。

## Progress
- [发射时填]

## Results / Conclusion
（待）
