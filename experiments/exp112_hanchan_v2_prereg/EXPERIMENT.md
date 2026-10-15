# exp112 — 纯血半庄 v2：自拟合顺位势函数信用 + 场况输入 + 强起点 + 规模

- **日期**：2026-10-15 预注册  **状态**：building
- **Env**：纯血线；Rust 引擎镜像半庄（`--hanchan_pure`）；策略 `cnn_l_v3rh`（v3r + 局数/本场/all-last 三标量，由 exp106 R 240M 的 v3r 权重 widening 加载）
- **用户指示**：2026-10-14 "把目标换成半庄…那条路的问题是学习效率而不是方向——继续往这里探索"

## Purpose & Hypothesis
exp109/110 + exp46/104 定案：单局点差目标下推/退边际 ≈ 0（对纯血网和 bc49 都是），人类防守理论最值钱的部分被目标定义掉了；半庄刻度把 bc49 的优势放大 1.8×（exp53）。
三次半庄尝试（exp70 从零 150k 场、exp84 W 微调 1M 场、exp91 P4 微调 1M 场）全部 H0，共同弱点：
1. **信用**：Rust 路径只有 `credit none`（逐局裸点差 + 终局 uma 一次性给，跨 ~120 步衰减且方差 ≈ 3× 单局）；exp70 的解析顺位势函数（名次跳变不连续）比 none 更差；
2. **输入**：exp84/91 用 v3r，策略看不到局数/本场/all-last——顺位博弈最关键的场况信息缺席；
3. **规模与起点**：≤1M 场（≈8M 局，单局线平台预算的 1/30），起点 W/P4 半庄 vs bc49 仅 0.36。
**H1**：修掉这三点后（① 纯自对弈拟合的光滑 W 做逐局 ΔW 信用，telescoping 到终局 uma；② v3rh 输入；③ 从 exp106 R 起、5M 场 ≈ 40M 局），半庄 vs bc49 显著高于 credit-none 对照与起点，且出现**选择性**防守（南场领先格子放铳率低于东场同名次）。
**H0**：半庄 ≤ 对照 + 3 pp，或行为仍一刀切 ⇒ 半庄目标在可负担预算内不是纯血线杠杆（第四次）。

## 先验声明
W 从**本策略自对弈**的（局间状态 → 终局 uma）拟合，零人类数据；ΔW 是势函数塑形（PBRS 的局级版），不改变最优策略；场况标量是引擎事实。纯血。

## Method
**工程**
- Rust：编码器 `v3rh`（+3 标量，与 Python `encode_state(variant="v3rh")` 逐位一致）；`DealFact` 增加局前场况（dealer/rw/honba/kyotaku/n）与 houjuu/winners；`Finished` 导出每席每局的末步下标 `deal_ends`，供主机侧重写奖励。
- Python：`scripts/fit_pure_w.py`（Rust 半庄自对弈采样局间状态 → PlacementValue 残差 W，holdout MAE 对比 rank-uma 基线）；`collect_rust` 新信用 `wpure`：每席每局末步奖励 := W(局后) − W(局前)（末局 W(局后) := 真实 uma），其余步 0，重算折扣回报；不变量测试 Σ奖励 = uma − W(局首)。`cnn_l_v3rh` 入 zoo，v3r ckpt 经 scalar widening 加载。
**臂**（均从 `exp106_R/games_final.pt` 240M 起，Rust 半庄镜像，5M 场；配方 = exp106：lr 3e-5、α 0.003、γ 0.995、λ 0.95、target_kl 0.03、无 adv_clamp；`games_per_iter 256` 场）
- **H**：`cnn_l_v3rh --hanchan_credit wpure`；
- **N**（对照）：`cnn_l_v3rh --hanchan_credit none`（同输入、同起点、同规模；隔离信用方案）。
**评测**：半庄 vs bc49 2v2 复式 n=1,200 场（seed 59300000，exp83/84/91 同口径）每 1M 场一次；终点 n=2,400；单局 20k 对 vs bc49（允许下降，记录代价）；行为分格（东/南 × 名次：立直率、放铳率，来自 DealFact）。
**机器/费用**：Secure 4090（Rust 半庄 cnn_l ≈ 200–300 场/s）：5M 场 ≈ 5–7 h ≈ $5–6/臂 ⇒ **≈$12，上限 $16**。

## Success Criteria（预注册）
1. 健康：KL < 0.03、H 0.2–0.4；W 拟合 holdout MAE 优于 rank-uma 基线 ≥ 5%；信用不变量测试通过。
2. **主判据**（终点 n=2,400 场）：H vs bc49 ≥ **0.41**（纯血最高 0.371）且 H − N ≥ **+3 pp**（SE ≈ 1 pp）⇒ H1；H ≥ 0.41 但 H ≈ N ⇒ 输入/起点/规模是杠杆、信用不是（也是正结论）；H ≤ 0.38 ⇒ H0。
3. 选择性：H 在"南场·前两名"格子的放铳率 < "东场·前两名" − 2 pp，且 < N 同格子（exp91 的 P4 是 0.112→0.156 反向）。
4. 代价：单局 vs bc49 允许下降，但 < 0.43 时记"顺位目标以单局强度为代价"。
5. 早停：1M 场时半庄 < 起点 − 5 pp 且下行 ⇒ 停该臂。
6. 运维：TRAIN_DONE ⇒ md5 握手 ⇒ 自毁；拉取循环主 checkout 绝对路径。

## Progress
- [10-15] 基建就位（提交 `0b42cd3`+）：Rust v3rh 编码（与 Python 逐位一致，parity 测试扩展）、DealFact 场况/结局 + `deal_ends` 导出、`collect_rust --hanchan_credit wpure`（主机侧 ΔW 重写，Σ奖励 = uma − W(局首) 不变量测试通过）、`cnn_l_v3rh`（v3r ckpt widening 加载）、`scripts/fit_pure_w.py`、`scripts/hanchan_eval.py`（exp83/84/91 口径）、`scripts/hanchan_behavior_grid.py`。
- [10-15] **起点行为分格（exp106 R 240M，2000 场镜像半庄，T=0）**：放铳率 E/top2 0.151、E/bot2 0.143、S/top2 **0.151**、S/bot2 0.143；立直率 S/top2 0.304 最高——领先时推得更凶，完全没有顺位意识（判据 3 的起点）。
- [10-15] 纯 W 拟合中：exp106 R 镜像半庄 300k 场（≈12M 行，129 场/s）。

## Results
（待）

## Conclusion
（待）

## Artifacts
（待）
