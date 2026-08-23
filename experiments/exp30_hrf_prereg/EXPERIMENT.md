# exp30 — HandRiverFormer：手牌×牌河双序列注意力（预注册）

- **Date**: 预注册 2026-08-23  **Status**: planned（发射条件：第二批（目标熵/规模配方）结论 + 预算批准）
- **Git**: 本提交（arch_zoo `HandRiverFormer` / encoder v4 / tests/test_hrf.py）；设计定稿 docs/design_hrf_exp30.md（用户逐条审阅）
- **Env**: g4-standard-48 flex；纪元 4 引擎（随机场上下文）

## Purpose & Hypothesis
exp27 证明"手牌实例集合"本身无增益；理论分析指出实例真正有个体属性的是**牌河事件**（何时/谁/摸切/宣言/被鸣）。
本实验检验：手牌 query cross-attend 事件序列的纯注意力模型（无 CNN）能否 (H1) 达到 CNN 冠军强度（Elo T=0 ≥ 对照−10），
且 (H2) 在**读牌维度**（对锚点放铳率、defense_iq）显著优于对照——这是该架构应赢的维度。
消融：(H3) cross-attn > 均池拼接；(H4) 时间 PE（巡目+新近度）有实质贡献（对 v3 平面结论的架构级复检）；(H5) 正弦 rank > 自由查表。

## Method
主臂 `hrf_xl_v4`（23.2M）+ 同配方 `cnn_m_r` 对照，从零 1.0M 局，配方取第二批胜者；
消融臂（nocross/notime/freerank）仅在主臂 H1 成立时发射。

## Success Criteria
1. H1：T=0 Elo ≥ 对照 − 10。
2. H2：对锚点贪心放铳率 ≤ 对照 − 1.5pp 或 defense_iq ≥ 对照 + 0.02（读牌收益）。
3. 吞吐 ≥ 35 局/s（本地 48.6 已过预检）；训练健康按发射清单核对。

## Progress
- [预注册] 实现 + 196 tests + 本地吞吐完成。

## Results / Conclusion / Artifacts
（待）
