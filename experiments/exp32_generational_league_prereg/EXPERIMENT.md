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

## Progress
- [17:4x] 基建冒烟通过（向量化 worker × 联赛路由，第 1 轮 decisive 34% = 池模型在对手席生效的旁证）。
- [17:5x] gen1 发射于 mahjong-g4-gen1（G4 flex，SHA 05dc327+池清单）：245 局/s。教训：experiments/ 的 gitignore
  挡了池清单 → VM 差点缺文件，已 `git add -f` 修复（清单类小文件必须显式入库）。
- [样本效率条款] 联赛局只收学习者席位（对手席=冻结策略，π_new/π_frozen 比率失控故不可用）：每局有效样本
  ≈ 镜像的 63%（实测 116k vs 183k/轮），墙钟反而更快（245 vs 196 局/s）。判读规则：gen1 在样本劣势下胜臂 5
  则净收益被低估；**若 H2 落在 ±15 边界内，补样本对齐臂 gen1s（1.45M 局）排除样本亏空解释**；
  长期旋钮 = 联赛局学习者席位 1–2 → 2–3（份额 81%），须单变量消融后再改。
