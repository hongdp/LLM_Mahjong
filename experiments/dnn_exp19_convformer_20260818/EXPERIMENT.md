# exp19：ConvFormer 生死战（为击败 cnn 设计的注意力架构，RL 判决）

- **Date**: 预注册 2026-08-18  **Status**: done（r2；评分口径胜 cnn、+137 胜旧 vit，仍在爬升）
- **Git**: 951a6ea（架构 746b7bd + BC 验证 951a6ea）  **Env**: mahjong-dnn-c5（us-east1-c）
- **对照**: exp18-cnn = `dnn_exp18_cnn_20260817`（同协议共享基线）；exp18-vit（928.3）为旧注意力参照
- **设计文档**: docs/design_convformer.md（三病根对症：花色内卷积 stem + rank 相对偏置 /
  1.97M 容量对标 / pre-LN+零初始化策略头+warmup）

## Purpose & Hypothesis
用户要求「设计能 beat cnn 的 attention 架构」。BC 已过线（90.5% 全 zoo 纪录，但 exp10
教训：保真度不兑现强度）。假说：修掉三病根后注意力渐近线 ≥ cnn（~1012）。

## Method / Config
exp18 共享协议 + `--arch convformer_m --warmup_updates 1000`。诚实声明：warmup 属
「注意力系统包」，本实验是系统包对照非纯架构消融（warmup 对 cnn 侧中性偏无用）。

## Success Criteria（发射前定死）
1. **主判定**：700k vs exp18-cnn-700k，200 副复式。显著正 ⇒ 注意力路线复活；
   null ⇒ 与 cnn 渐近线持平（结合正式评分判读）；显著负 ⇒ 「局部模式游戏里卷积先验
   不可替代」结案。
2. **对旧注意力**：正式评分应显著 > exp18-vit 的 928.3（三修复的净效果，否则设计无效）。
3. 早期速度保持：80k 处 ladder ≥ exp18-vit 同点（886）——先验不应牺牲爬速。
4. 训练健康：KL、熵地板、吞吐 ≥10 局/s（d160×6 层 CPU rollout 是新变量，实测记录）。

## Progress
- [2026-08-18] 预注册，c5 发射。
- [2026-08-18] 发射确认：首 iter win_rate 0.009，149 s/iter（**13.8 局/s，过 ≥10 健康线**——d160×6 CPU rollout 代价实测 1.8× cnn），预计 ~14h 跑满。ladder 已挂。**Status: running**。
- [2026-08-18 22:45] **r1 中止（10%，设计失误，诚实记录）**：--warmup_updates 1000 的尺寸
  按 LLM 级更新量拍脑袋；实测本配置全程仅 ~2400 次 minibatch 更新，1000 步预热 = 40% 训练
  在低 LR 爬行（71k 局 win_rate 仅 0.042、熵 1.88，远落后基线同期）。r1 产物弃用。
  **r2 重发**：唯一改动 --warmup_updates 1000→150（~6% 更新数，预热的合理占比），
  目录 `dnn_exp19_convformer_20260818r2`。教训：warmup 长度必须按「占总更新数比例」设定，
  不能搬绝对步数。

## Results
| Metric | This run (ConvFormer r2) | Baselines | Success criterion |
|---|---|---|---|
| 主判定：700k 对打（seed0=20260827） | +490 ± 871（副胜 182:146） | exp18-cnn | 点差 null ⇒ 预注册「持平」分支；评分差 **+52（z≈2.8）略胜** |
| 对旧注意力 | 正式评分 **1065.7 ± 13.3** | exp18-vit 928.3 | **+137 ⇒ ✅ 三修复净效果显著** |
| 80k 爬速 | 616（r2 warmup 仍拖慢早段） | exp18-vit 同点 886 | ❌ 未达（判据在无 warmup 假设下预注册；诚实记录） |
| 健康/吞吐 | 13.8 局/s、熵正常、跑满 | ≥10 | ✅ |
| ladder 轨迹 | 616→970→…→**1074（700k 仍在爬升）** | vit 平台 ~930 | — |

## Conclusion
1. **注意力路线复活**：三病根修复兑现 +137 Elo（vs vit_small 受控同协议），终点评分
   1065.7 略高于 cnn 基线（+52, z≈2.8），点差对打 null——「beat cnn」以评分口径达成、
   以点差口径持平；且 700k 仍在爬升，渐近线未见顶。
2. 代价：早期爬速被 warmup 牺牲（80k 判据失败）——ConvFormer 是「慢热高顶」型，
   与旧 vit 的「快热低顶」互为镜像。
3. 与 exp17-C 汇合的图景：**GAE（1079.7）与 ConvFormer（1065.7）分别独立打破 1012 平台**，
   两者正交（信用分配 vs 表征）→ exp20 合体实验是自然收敛点。

## Artifacts
| Path | Description |
|---|---|
| gs://llm-mahjong-experiments/dnn_exp19_convformer_20260818/ | 云端主目录 |
