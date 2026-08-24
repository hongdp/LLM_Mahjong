# exp32 — 跨代联赛第一期（gen1）预注册

- **Date**: 2026-08-23 预注册  **Status**: planned（发射条件：基建冒烟通过 + 用户批 ≤$10；臂 5 已完成作对照）
- **Git**: 设计定稿 docs/design_exp32_generational_league.md（配方形式化 / 晋升门 / 停机规则）
- **Env**: g4-standard-48 flex；纪元 4 引擎；池 = {exp27-A frozen}

## Purpose & Hypothesis
防守悖论的解法检验：让强对手（上代冠军）出现在熵高原孵化期。
H1 defense_iq ≥ 0.05 且对锚点放铳率 ≤ 臂5 − 1.5pp；H2 Elo ≥ 臂5 + 15；H3 不被池内任何成员剥削。
对照=臂 5（同配方纯镜像×纪元 4，1031.9±12.2）；负参照=exp22（warm-start 联赛，defense_iq 0.016 无效）。

## Method / Config
`--arch cnn_m_r --league --league_frac 0.5 --gpu_infer --gpu_infer_opponents --games_per_worker 32
 --entropy_schedule 0:0.03,600000:0.01 --total_games 1000000`（其余同冠军配方）；池文件含 exp27-A。
发射前：向量化 worker × 联赛路由冒烟（本地 512 局，校验对手席位模型正确、learner-only episodes、吞吐）。

## Success Criteria
见设计文档 §3；停机与诚实条款 §5。

## Progress
（发射时填）
