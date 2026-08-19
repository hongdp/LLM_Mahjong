# exp19：ConvFormer 生死战（为击败 cnn 设计的注意力架构，RL 判决）

- **Date**: 预注册 2026-08-18  **Status**: launching
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

## Results
| Metric | This run | Baselines | Success criterion |
|---|---|---|---|
| （待运行） | | | |

## Conclusion
（待运行）

## Artifacts
| Path | Description |
|---|---|
| gs://llm-mahjong-experiments/dnn_exp19_convformer_20260818/ | 云端主目录 |
