# 成对半庄裁决：exp124 192M / 224M / 256M / 288M 对 bc49

- **Date**: 2026-11-13 08:22 PST  **Status**: running
- **Git**: 1d1de45（无未提交改动）；驱动 `hanchan_dup.py` 是 `share_to_uma_20261106_095032` 的原样拷贝，调用稳定入口 `scripts/run_elo_league.play_pair_vector(hanchan=True)`
- **Env**: 本机 RTX 4080，rlhf_mahjong 环境；$0

## Purpose & Hypothesis
用户 2026-11-13 要求立即跑。exp124 @ 256M 于 11-12 按单局复式（对 bc49 0.5230 ± 0.0033，20,000 牌山）列为并列部署冠军，
但没有半庄读数——CLAUDE.md 评测协议要求终审含半庄，模型卡 §7 判据也允许 "n ≥ 300 半庄"。本次补上半庄刻度。
假设：256M 对 bc49 的半庄 uma 显著为正；按族内换算 1 pp 份额 ≈ 511 uma/座位/半庄（share_to_uma L3），
+2.3 pp 份额对应约 +1,200 uma，但该换算只覆盖份额 0.32–0.50，此处外推。

## Method
- 2v2 复式半庄对 bc49（`_anchors_epoch6/bc49.pt`），两边 T=0，向量竞技场 8 路并行，每场 = 2 个座位朝向。
- 检查点：exp124 A 192M（`exp124_A_p2/games_192000000.pt`）、Stage 2 的 224M / 256M / 288M（`exp124s2_S2/`）。
- 牌山种子：**判决段**两段新种子 202000000、202100000，各 1,500 场（合计 3,000 场；SKILLS 10-15：半庄读数一律两段合并）；
  **衔接段** 201000000 × 1,500 场（与 share_to_uma 的 13 检查点曲线同一批半庄，只作衔接，不进判决）。
- 同一种子 ⇒ 所有检查点打同一批牌山，检查点之间按场配对。
- 指标：每 A 座位每半庄 uma（diff / 8），SE 按场聚类；场胜率（match-win）。

## Config
`run.sh`（本目录）；每检查点每段 1,500 场，约 3–4 分钟。

## Success Criteria（开跑前写定）
- **主判据**：256M 对 bc49，判决段 3,000 场，uma/座位/半庄 > 0 且 z ≥ 2 ⇒ 半庄刻度上加冕判据满足（模型卡 §7）。
  0 < z < 2 ⇒ 方向为正但未达显著，加冕仍只在单局刻度成立；≤ 0 ⇒ 半庄刻度不支持，模型卡须写明。
- 次判据：同场配对 256M − 192M（判决段）；192 → 224 → 256 → 288M 的走势（只描述，不另下判决）。

## Progress
- [2026-11-13 08:22 PST] 预注册写定；冒烟（256M，16 场）0.1 min 正常。开跑。

## Results
（待填）

## Conclusion
（待填）

## Artifacts
| Path | Size | Description |
|---|---|---|
| `experiments/probes/hanchan_verdict_exp124/<tag>_<seed>.json` | — | 每检查点每段逐场 diff / score |
| 本目录 `run.sh`、`run.log`、`hanchan_dup.py`、`analyze.py` | — | 驱动、日志、分析 |
