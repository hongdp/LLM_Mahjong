# exp109 Stage 0 — 冻结策略下把独立 Q critic 训到收敛：oracle Q / 引擎精确放铳代价分解

- **日期**：2026-10-12  **状态**：running（本机 4080，$0）
- **Git**：exp109 提交（`feat(exp109)`）；策略 = `exp102_A/games_final.pt`（224M，冻结）
- **Env**：Rust 引擎自对弈 T=1，每迭代 2048 局（K=1024），数据无限（每迭代全新）；critic 128ch×4 blocks，trunk 从策略热启动，Adam 1e-4，每批 2 pass，bf16 前向
- **用户指示**：2026-10-12 "做"（Stage 0 先在本机 $0 验证 oracle Q 的排序质量，再决定上云）

## Purpose & Hypothesis
exp108 判 oracle 信息 H0 用的是 EV（状态价值层面），而 oracle 最该起作用的是**动作之间**的差；且 OQ critic 在 0.46M 局时并没有学到"切待张 ⇒ 几千分代价"（待张牌只比安全牌低 59 分，盲 critic 42 分）。
原因：Q 只在采样动作上有监督（策略 94% 避开待张）、交叉项难学、训练量不足。
**修法**：放铳代价不需要学——引擎对每张合法切牌精确给出即时荣和代价 c_ron(s,h,a)（`Table.ron_costs`，含双响、振听、赤五，`tests/test_ron_cost.py` 对账为 0 误差），Q 拆成 Q_full = Q_rest + c_ron（**ODQ**），critic 只学平滑的剩余部分。
- **H1**：ODQ（可能也包括训够的 OQ）在暴露决策上的跨动作排序显著优于策略自身（避待张率 > 0.939）且在非暴露状态上与策略 argmax 的一致率 ≥ 盲 Q；Q(待张) − Q(安全) 达千分量级（ODQ 由构造保证，看 Q_rest 是否把它抵消）。
- **H0**：训到收敛后 OQ ≈ BQ（oracle 的动作级信息也学不出来）；或 ODQ 的 Q_rest 系统性抵消 c_ron（critic 把已知代价"学回去"）。

## 先验声明
c_ron 与 oracle 向量同类：训练期引擎事实（隐藏状态下的规则结果），仅进 critic/优势，策略输入不变，推理时不存在。用户 2026-10-12 确认不影响纯血。

## Method
`scripts/critic_stage0.py`：三 critic 同批同配方——**BQ**（盲）、**OQ**（oracle 319 维）、**ODQ**（oracle + 分解，目标 R_t − c_ron(a_t)）。每 25 迭代在**训练前的新批**上做诊断（out-of-sample）：
- `*/ev`：Q(s,a_t) 的解释方差；
- 暴露决策（≥1 家立直、待张与安全切牌都合法）：`*/avoid_waits`（critic argmax 避开所有待张的比率）vs `policy_avoid_waits`，`*/wait_gap_pts`（Q(待张)−Q(安全)），`*/top2_gap_pts`；
- 全部多动作状态：`*/agree_policy`（argmax 一致率）。
ckpt 每 100 迭代（`critic_{BQ,OQ,ODQ}_latest.pt`），`metrics.jsonl`，`config.json`。计划 1500 迭代（≈3M 局，本机 ≈3–4 h）；曲线未平则延长。

## Success Criteria（预注册，取最后 5 次诊断均值）
1. **信息**：OQ − BQ 的 ev ≥ +3 pp **或** OQ 的 avoid_waits − BQ ≥ +5 pp ⇒ oracle 的动作级信息可学（exp108 判据 1 被 EV 低估）；否则维持 H0。
2. **ODQ 闸门（上云条件）**：ODQ avoid_waits ≥ 0.95（> 策略 0.939）**且** ODQ agree_policy ≥ BQ agree_policy − 2 pp（分解没有破坏其他状态的排序）**且** ODQ ev ≥ BQ ev − 1 pp ⇒ 批准 Stage 1（云上 PPO，优势 = Q_full 的 all-action 形式，v4 配方：自归一化 + 熵控制 0.28；≈$6）。
3. 任一不达 ⇒ 记录为 H0，oracle/价值侧彻底关闭，不上云。
4. 健康：critic 损失单调不发散；吞吐 ≥ 800 局/s（4080）。

## Progress
- [10-12 21:47 PDT] 冒烟 11 迭代通过（ODQ 由构造 avoid_waits 0.99、wait_gap −5.7k；ODQ ev 0.129 vs BQ 0.112 已多出 1.7 pp）。Rust 对拍套件（table/game/score/rollout/encoder parity）全过，win_result 重构无行为变化。
- [10-12 21:52 PDT] **发射**：1500 迭代 × 2048 局，本机 4080，PID 2929563，日志 、。

## Results
（待）

## Conclusion
（待）

## Artifacts
（待）
