# exp47（预注册）— 人类先验谱系第一迭代：trunk × 输入 × 动作空间因子分解

- **Date**: 2026-08-27  **Status**: launching  **Cost**: 本地 4080，~5 臂 × (0.5-1h BC + 15min 评估)
- **Git**: prereg at HEAD（engine/exp45-action-gaps 分支）
- **谱系**: 人类先验谱系（用户 2026-08-27 正式开启）。北极星 = **简单输入平面（≤v3r 56 面、零派生特征）
  的模型综合能力（Elo + defense_iq）超越 Mortal**。

## Purpose & Hypothesis

exp45 榜首两臂恰好构成 3 因子对照的两个角点：

|  | trunk | 输入 | 动作 | Elo | defense_iq |
|---|---|---|---|---|---|
| bc45_conv | ConvFormer 2M | v1r 21 面 | 374 | 1133.2 | 0.110 |
| bc45_mortal46 | SE-ResNet40 19M | 934 面（派生清零） | 46 | 1147.4 | 0.144 |

Δ=14±19 统计平手，但归因完全未知。本实验补齐因子矩阵，检验：

- **H1（动作空间假设）**：mortal46 的名义优势主要来自 46 槽动作压缩（鸣牌语境化=样本效率），
  而非 934 面输入布局——若成立，"简单输入"目标零代价达成；
- **H2（可组合性）**：ConvFormer trunk × 简单输入 × 46 槽 ≥ 1147（把两臂各自的赢因子拼起来）；
- **H3（输入布局无关）**：SE-ResNet40 × v3r 56 面 × 46 槽 ≈ bc45_mortal46——934 布局
  （即使派生清零）相对 v3r 无增量。

## Method

五臂新增（10% 数据点 2000 局、收敛早停 patience 3、同 seed/lr，与 exp45 完全同协议）：

| 臂 | trunk | 输入 | 动作 | 检验 |
|---|---|---|---|---|
| mortal_bb_xl_r | SEres40 | v1r 21 | 374 | trunk 单独效应（ZOO 现成） |
| convformer_m_v3r | Conv | v3r 56 | 374 | 输入单独效应（ZOO 现成） |
| convformer_m_r_m46 † | Conv | v1r 21 | 46 | 动作单独效应 |
| convformer_m_v3r_m46 † | Conv | v3r 56 | 46 | H2 旗舰候选 |
| mortal_bb_xl_v3r_m46 | SEres40 | v3r 56 | 46 | H3（输入布局代换） |

† 需新增 ConvFormer 46 槽结构化头（tile token→槽 0-36、global token→槽 37-45，零初始化保持
exp19 的均匀起步性质）；MortalBackbone 的 46 头是平铺 flatten——结构化头如更优属 trunk 效应的一部分，如实记。

评估：每臂 Elo（T=1、mismatch 打标、seed0 45100001 与 exp45 配对）+ defense_iq（800 局、seed0 45200001）。

## Success Criteria

1. **主判据**：存在简单输入臂（v1r/v3r）Elo ≥ 1147.4 − 15 且 defense_iq ≥ 0.10
   → 北极星的"简单输入不让分"前提成立，该臂成为谱系旗舰进入 exp46（RL-on-prior）与全量数据训练；
2. 因子报告：三因子主效应与交互（8 个格点已有 3 个，本轮补 5）；
3. H3 否定分支：若 934 布局仍显著超 v3r（>2σ），"简单输入"目标需重审——如实记录并停在这里等用户裁定；
4. 北极星外部读数（本轮不测，登记）：对真 Mortal 的比较 = 雀魂 maka 实战 +（未来）真 Mortal 权重入场；
   本轮只解决内部尺。

## Progress
- [2026-08-27] 预注册。ConvFormer46 头待实现。
