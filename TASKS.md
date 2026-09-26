# TASKS — 指针页（2026-08-30 起瘦身；2026-09-26 更新）

任务与进度的权威位置：
- **实验总账**：[experiments/INDEX.md](experiments/INDEX.md)
- **结果台账**：[experiments/FINDINGS.md](experiments/FINDINGS.md)
- **进行中/预注册**：`experiments/*_prereg/EXPERIMENT.md`
- **教训与状态快照**：[SKILLS.md](SKILLS.md)；状态看 [README.md](README.md)「Current status」

## 当前队列（快照，详情见上述文件）
0. **【终极目标，用户 2026-09-25】纯血线自我提升；其他实验皆辅助。** $200 计划累计 ≈$111。
   **在跑：exp101 容量 × 规模长跑**（cnn_l_v3r 从零 400M 局，Secure RTX 4000 Ada，上限 $100；判据 112M ≥0.462 续 / ≤0.452 转纯规模；终点 ≥0.50 达标）
   ——[experiments/exp101_capacity_scale_prereg/EXPERIMENT.md](experiments/exp101_capacity_scale_prereg/EXPERIMENT.md)。有 A40/3090/4090 货即先基准再迁移（用户 09-26 同意）。
1. 已关闭的杠杆（09-13→09-25，全部 H0，别再重试）：惩罚+退火（exp95）、价值法 ×4（exp59/60/71/96）、参数噪声（exp97，训练器口径失效）、贪心续打+单点偏离（exp98）、
   危险度安全集宏（exp99）、模式序列 PIMC 搜索（exp100）、静态留牌（exp94 附录 4）；博弈论防线（池/PSRO/熵/风格人口）亦已量过——见 FINDINGS 09-26。
2. 待用户决定：①exp95 S410 延长（+$1.9，越 $8 上限，建议不跑）；②为 PR #20 合并后的 ~50 个本地提交开新 PR；③纯度边界（人写的"向听阈值"候选规则算不算人类先验）——exp100 已改用搜索选择，暂不需要。
3. 人类先验线（辅助）：部署冠军仍 bc49；纪元 7（评分体系 v2）待开；数据扩容（10 万局级）仍是该线最高期望值杠杆。
4. 工程储备：推理服务器混动作空间托管；PPO 分布式采样（仅在需要 >2× 吞吐且接受配方变化时）。
