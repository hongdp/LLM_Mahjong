# 份额 → 半庄 uma 换算（论文 L3）（2026-11-06）

- **Date**: 2026-11-06 09:50 PST  **Status**: running
- **Git**: 43e02ce；向量评测代码 `/home/hongdp/Workspace/LLM_Mahjong_exp124`（f11278a）；脚本本目录
- **Env**: 本机 RTX 4080；$0

## Purpose & Hypothesis
论文会话 L3：给出"单局复式份额每 1 pp ≈ 半庄 X uma"的换算及其不确定度。描述性拟合。

## Method
- 13 个已有 10,500 去重牌山份额读数的检查点（exp27A 1M；cnn_l 32/64/112/160/192M；exp124 A 16/32/64/112/160M；exp125 XL 32/64M）各跑 **1500 场 2v2 复式半庄对 bc49**（向量评测 `play_pair_vector(hanchan=True)`，T=0 两边，同一种子 201000000 ⇒ 所有检查点打同一批半庄，跨检查点也配对）。
- 指标：每 A 座位每半庄 uma（= 每场 diff / 8，diff = 两个方位 A−B uma 之和）± SE（按场聚类）。
- 拟合：uma = a + b·(share − 0.5)，加权最小二乘（权重 1/(SE_uma² + b²·SE_share²) 迭代），报告 b（uma / pp）± SE 与残差；另报不含 exp27A（远端点）的拟合，检验线性。bc49 本身（share 0.5、uma 0）作为定义点不入拟合，用截距 a 检验。

## Success Criteria
描述性；截距 a 应与 0 相容（否则说明份额与半庄刻度有偏移，要报告）。

## Progress
- [2026-11-06 09:50 PST] 接口校验：40 场测试，diff/8 = A 每座位每半庄 uma（第一场 diff −4000 ⇔ 两方位 A 合计 −2000 ⇔ 每座位 −500，与 rows 中 a_pts 一致）。发射 13 个检查点 × 1500 场。

## Results

## Conclusion

## Artifacts
