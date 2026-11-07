# R3：概念探针的配对安慰剂（按向听 × 巡目分层，三家牌河轮流作假现物）（2026-11-07）

- **Date**: 2026-11-07 01:45 PST  **Status**: running
- **Code**: `probe_concepts_matched.py`（本目录）= exp129 `probe_concepts.py`（843a2a3，md5 4b85fcc0…）的副本 + 新增 `m_` 记录与 `summarize_matched`；**原有记录与输出逐字不变**（结果 json 同时含原 exp129 全部字段与新 `R3_matched`）。`PROBE_CODE_ROOT=/home/hongdp/Workspace/LLM_Mahjong_exp124`
- **Env**: 本机 CPU（6 进程，与 akochan、R4/R2 并跑）；$0
- **Approval**: 论文会话转达其用户 11-07 的批准

## Purpose & Hypothesis
exp129 的安慰剂（无人立直时把**对家**牌河当假现物）与威胁局面在巡目、向听分布上不匹配，且只用一家。R3 用匹配后的安慰剂重算 T7、T8（威胁 − 配对安慰剂），检验涌现结论对安慰剂构造是否稳健。描述性。

## Method（发射前写定）
- **匹配方式：分层后重加权（post-stratification）**。层 = 本家向听档（weak ≥2 / mid 1 / tenpai ≤0，即 exp129 的分桶）× 巡目档（本家已打张数 j0 ≤5 / j1 6–11 / j2 ≥12）。威胁记录与安慰剂记录都按层记；安慰剂率在每层内计算后，按**威胁局面在各层的计数**加权平均 ⇒ 配对安慰剂。只在威胁与安慰剂都有数据的层上合并（报覆盖的威胁权重）。
- **假现物来源**：无人立直局面里，**三家对手的牌河轮流**作假现物，各计 1/3 权重后平均（不再只用对家）。
- 度量：T7 每档弃和率（威胁 − 配对安慰剂）、defense_iq（weak − tenpai，各自配对后相减）；T8b 六个类对偏好（weak / mid / any；any 跨三档分层）。
- SE：按局的簇自助法（300 次，multinomial 权重）。
- 模型：exp124 A 全部 20 点（同 exp129）、bc49、exp129 的 11 个 CNN 点（exp101 cnn_l 8M–392M）；T=0，牌山 50,000,000–50,003,999，4000 局（同 exp129）。
- 冒烟（bc49，200 局）：配对 iq 0.144 ± 0.036（原"iq − 对家安慰剂" 0.252）；T8b 筋∪壁 vs 无筋 3–7（weak）配对后 0.538 ± 0.092（原 0.470）。冒烟读数不作结论。

## Success Criteria
描述性；与 exp129 原读数并列报告，按 exp129 的涌现规则（|z| ≥ 3 且此后保持）重判涌现点。

## Progress
- [2026-11-07 01:45 PST] 预注册；冒烟通过；发射。

## Results

## Conclusion

## Artifacts
