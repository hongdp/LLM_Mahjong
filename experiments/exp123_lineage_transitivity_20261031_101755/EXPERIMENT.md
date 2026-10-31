# exp123 — exp113 C 谱系检查点传递性梯子 + 末尾检查点权重平均的推理期增益（用户点名复测）

- **Date**: 2026-10-31 10:18  **Status**: running（本地 CPU，$0）
- **Git**: 本 worktree（`scripts/run_pairwise_hanchan.py`、`scripts/avg_checkpoints.py` 新增；`scripts/run_hanchan_arena.py` 加 `--temperature/--threads/LLM_MAHJONG_ROOT`）；
  库代码（`cnn_l_v3r`、半庄驱动）取 `claude/continue-task-3da2e7` 02d0cbe 的 detached worktree
- **Env**: 本地 24 核 CPU，6 个进程 × 4 线程；雀魂服务在跑（空闲）
- **提议人**: 用户 2026-10-31："这个也排队"（传递性诊断）+"这个加上"（检查点集成）

## Purpose & Hypothesis
**先验（必须先写明）**：这两项在 64–112M 规模的 cnn_m 纯血谱系上**已经测过并判负**——
对手池 exp82 中性、防守者池 exp85 H0、PSRO exp87（M 不可剥削）、风格人口 exp89 H0；FINDINGS 2026-09-26 综述："非传递性未观察到：族外梯子传递、族内头对头稳定 0.50、
曲线是平台不是震荡"；exp92 平台模型 logit 集成 +0.3 pp（半庄同样 H0）。本实验是在 **384M 规模的 cnn_l 谱系**上复测，且换了两个未测过的具体形态：
(a) 同一谱系跨 8× 训练量（48M→384M）的全配对半庄梯子；(b) 同一 run 连续检查点的**权重平均**（SWA 式，exp92 是不同模型的 logit 集成）。
唯一能让规模改变结论的已知先例是防守涌现（exp120）；平台值本身 0.455→0.47 几乎没动，所以预期仍是 H0。

- **H0-a（传递）**：Bradley–Terry 评分沿谱系单调不降，所有配对的观测份额与 BT 预测一致（|z 残差| < 2.5），没有旧检查点 2σ 胜新检查点。
- **H1-a（非传递）**：任一配对 |z 残差| ≥ 2.5（21 对的 Bonferroni 近似），或任一"旧胜新"达 2σ → 自对弈有循环 → 才值得写 league 续训预注册。
- **H0-b（集成无增益）**：W3（368/376/384M 平均）、W5（352–384M 平均）对 C384 的半庄份额 < 0.56。
- **H1-b**：≥ 0.56（≈2σ，n=300）→ 再上 T=0 族外梯子 vs bc49 确认（半庄份额不是强度终审刻度）。

## Method
- 选手（谱系顺序）：L48 `exp101_L/games_48000000` → L96 `exp101_L_p2/games_96000000` → L192 `exp101_L_p2/games_192000000` →
  C288 `exp113_C/games_288000000` → C384 `exp113_C/games_384000000`（谱系：exp101_L → p2 → exp102_A@192M 熵退火 → exp107_P → exp113_C，均 `--resume` 续链）；
  加 W3、W5（`avg_checkpoints.py`，60/68 张量为浮点平均，BN 计数器取末尾）。7 人 21 对。
- 协议：2v2 半庄（A 占 0/2 席偶数局、1/3 席奇数局），**T=0**，每对 2 个不重叠牌山段（seed0 123000001 / 456000001）×150 = 300 半庄，
  所有配对共用同一组牌山（跨对 CRN）。得分 sign(uma_A − uma_B)。半庄刻度结果**不写入单局梯子历史**。
- 分析：`run_pairwise_hanchan.py` 的 BT-MLE（Elo 单位，均值 0）+ 每对 z 残差 + "旧胜新"检验；输出 `ladder.json`。
- 预算：42 个 arena 作业 × 150 半庄 ≈ 6300 半庄，单进程 4 线程 ≈1.4 s/半庄 → 6 进程 ≈ 25–35 min。$0。

## Success Criteria（发射前定）
见 H0/H1。附报：BT 评分表、份额矩阵、W3/W5 vs 各检查点。若 H1-a 成立，下一步是 league 续训预注册（对手 50% 最新 + 50% 池均匀，对照纯续训，费用按 exp113 单价）；
若 H0-a 且 H0-b，本线关闭，纳入收官报告。

## Progress
- [10:17] 建目录；W3/W5 生成；T=0 arena 冒烟 4 局通过（W3 0/4 负于 C384，样本太小不作数）。
- [10:20] 发射 42 作业，6 并行；心跳 10 min 检查 pairs/ 新文件，30 min 无新文件报警。

## Results
（待填）

## Conclusion
（待填）

## Artifacts
| Path | Size | Description |
|---|---|---|
| `experiments/exp123_lineage_transitivity_20261031_101755/pairs/*.json,*.log` | — | 每对每段 arena 结果（含逐局 uma/顺位） |
| `experiments/exp123_lineage_transitivity_20261031_101755/ladder.json` | — | BT 评分 + 配对残差汇总 |
| `experiments/exp123_lineage_transitivity_20261031_101755/W3_368_376_384.pt`, `W5_352_384.pt` | 2×48 MB | 权重平均检查点（不入库） |
| `experiments/exp123_lineage_transitivity_20261031_101755/run.log`, `heartbeat.log` | — | 驱动日志 / 心跳 |
