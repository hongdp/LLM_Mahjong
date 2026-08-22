# exp21：防守测量套件（行为探针）+ 历代谱系回填

- **Date**: 预注册 2026-08-19  **Status**: running
- **Git**: 6e31a3c + probe_defense.py（本次新增）  **Env**: 本地 CPU ×16 workers
- **背景**: 用户问「如何判定学会防守/没学会问题在哪」；五件套探针的行为部分（1/2/条件放铳）。

## Purpose & Hypothesis
测量各谱系的**条件折牌能力**。核心指标 defense_iq = fold_weak − fold_tenpai
（弱牌曝露时现物率 − 听牌曝露时现物率）：永远推或永远折 ≈0，真防守 >0。
假说（生态压力论 H1）：defense_iq 随谱系立直率上升而上升（威胁供给驱动共进化）——
预测排序 GAE(立直24%) > exp18-cnn(12%) ≈ bc_cnn(教师零防守)；bcrl14 未知（教师无防守
但 RL 阶段立直环境浓）。

## Method
scripts/probe_defense.py：引擎状态直采（零文本解析）。每模型 800 自对弈局（4 同策），
曝露定义=有对手立直且自身未立直的打牌决策；现物=对**所有**立直者均为现物（诚实下界，
无筋模型）；放铳者由 result_summary `放铳:玩家X` 精确定位。冒烟 24 局通过。

## Success Criteria
探针类：产出五谱系 defense_iq 表 + 条件放铳率表即成功；副判据=H1 排序是否成立。

## Progress
- [2026-08-19] 冒烟通过（GAE 预告 defense_iq +0.071）；全量 5 模型 ×800 局发射。

## Results
（待运行）

## Conclusion
（待运行）

## Artifacts
| Path | Description |
|---|---|
| experiments/exp21_defense_probe.json | 全部指标 |
