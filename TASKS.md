# TASKS — 指针页（2026-08-30 起瘦身；2026-09-29 更新）

任务与进度的权威位置：
- **实验总账**：[experiments/INDEX.md](experiments/INDEX.md)
- **结果台账**：[experiments/FINDINGS.md](experiments/FINDINGS.md)
- **进行中/预注册**：`experiments/*_prereg/EXPERIMENT.md`
- **教训与状态快照**：[SKILLS.md](SKILLS.md)；状态看 [README.md](README.md)「Current status」

## 当前队列（快照，详情见上述文件）
0. **【终极目标，用户 2026-09-25】纯血线自我提升；其他实验皆辅助。** $200 计划累计 ≈$166（剩 ≈$34）。计划书 §16。
   **exp101 容量×规模：H0**（cnn_l 112–232M 平台 ≈0.455 = Q1x）；p2 仍在跑作对照，**建议 224M 判决后停机**（跑满 400M 还要 ≈$28）——待用户批准。
   **在跑：exp102 熵系数收尾退火**（A4000 $0.25/h，≈$5）：216M **0.4655 vs 对照 0.4529（+1.26 pp，≈2.5σ），纯血新高**；224M 正式判决今日上午
   ——[experiments/exp102_entropy_anneal_prereg/EXPERIMENT.md](experiments/exp102_entropy_anneal_prereg/EXPERIMENT.md)。
   下一步：分巡目弃和 + 安全牌库存探针（$0）→ 若 H1：剂量臂 α→0.0003、拆分臂（对手席 T=0）；若库存是瓶颈：承诺型宏动作（需先定纯度边界）。
1. 已关闭的杠杆（09-13→09-25，全部 H0，别再重试）：惩罚+退火（exp95）、价值法 ×4（exp59/60/71/96）、参数噪声（exp97，训练器口径失效）、贪心续打+单点偏离（exp98）、
   危险度安全集宏（exp99）、模式序列 PIMC 搜索（exp100）、静态留牌（exp94 附录 4）；博弈论防线（池/PSRO/熵/风格人口）亦已量过——见 FINDINGS 09-26。
2. 待用户决定：①exp95 S410 延长（+$1.9，越 $8 上限，建议不跑）；②~~开新 PR~~ 已开 [PR #21](https://github.com/hongdp/LLM_Mahjong/pull/21)（09-26，含 exp94 附录–exp101）；③纯度边界（人写的"向听阈值"候选规则算不算人类先验）——exp100 已改用搜索选择，暂不需要。
3. 人类先验线（辅助）：部署冠军仍 bc49；纪元 7（评分体系 v2）待开；数据扩容（10 万局级）仍是该线最高期望值杠杆。
4. 工程储备：推理服务器混动作空间托管；PPO 分布式采样（仅在需要 >2× 吞吐且接受配方变化时）。
