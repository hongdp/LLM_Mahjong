# exp108 — 独立 oracle critic（V / Q）在纯血 PPO 上重测

- **日期**：2026-10-12 预注册  **状态**：prereg
- **Git**：`5a28dc1`（实现 + 闸门）；repo.tgz md5 `8f9879d8…`
- **Env**：纯血线；Rust 引擎镜像自对弈；`cnn_l_v3r` 策略从 `exp102_A/games_final.pt`（224M）`--resume`；RunPod Secure RTX 4090
- **用户指示**：2026-10-11 "把 oracle Q 和 V 都在纯血 PPO 上分离开重测一遍"

## Purpose & Hypothesis
exp67（09-05）把 319 维隐藏信息喂给**共享 trunk** 的价值头、随 PPO 每迭代 1 pass 训 100 万局，EV 0.102 vs 盲 0.107，判零增益；当时的 R² 分解（紧凑特征 6.0%）把它归因于"结局由摸牌顺序决定"。
本实验改成能赢的形态：**独立网络、独立优化器、每迭代多 pass、并增加动作条件的 Q 头**。
- **H1-V**：独立 oracle V 的 EV 明显高于盲 V（≥ +5 pp），GAE 优势噪声下降，策略窗口均值比对照高 ≥ +0.5 pp。
- **H1-Q**：oracle Q 给出整行动作的优势 A(s,a) = Q(s,a) − Σπ(a′)Q(s,a′)，优势方差比盲 GAE 低 ≥ 50%，策略窗口均值比对照高 ≥ +0.5 pp，且暴露决策（对手立直）上的出生牌率下降。
- **H0**：EV 增益 < 5 pp 且策略无差 ⇒ exp67 的"方差来自未来摸牌"结论在独立网络形态下再次成立，oracle 线彻底关闭。
- 关键混杂：**独立 critic 本身**（更大容量、多 pass）可能就有收益，与 oracle 信息无关 ⇒ 盲 Q 对照臂。

## 先验声明
训练期自对弈特权信息（引擎状态），零人类数据、零外部模型；推理时不存在（策略头从不读 oracle）。用户 2026-09 已认可此类为纯血合法。

## Method
- **oracle 向量**：Rust `VecEnv.oracle_info()` / `Table.oracle(pid)`（`rust/riichi_rs/src/oracle.rs`），与 Python `oracle_features.py` 逐槽一致（`tests/test_oracle_rust_parity.py`：120 局随机对局含中断态全部相等）。`collect_rust(cfg["oracle_feats"])` 把它按 (slot, seed, seat) 挂到每个 episode 的 `cfeats` [T, 319]（float16）。
- **独立 critic**（`src/agents/dnn/oracle_critic.py::SeparateCritic`）：与策略同族的 CNN trunk（128 ch × 4 blocks ≈ 策略同量级）+ 标量 FC + oracle 投影（319→128），头 → V（1）或 Q（374）。自己的 Adam（lr 1e-4）、每迭代 2 pass、MSE 回归 MC 回报（Q 只回归已采样动作）。`--oracle_warmup_iters 200`（≈0.4M 局）期间 critic 只训练、优势仍用盲 GAE；之后**且** critic 在当批的 EV ≥ 策略自身价值头 EV − 0.02（`--oracle_gate_margin`，每迭代重判；冒烟发现半生不熟的 critic 优势经归一化后是单位尺度噪声，KL 0.002→0.04）才切换：V 模式把 GAE 的 V 换成 oracle V；Q 模式优势 = Q(s,a_t) − Σ_a π(a|s)Q(s,a)（π = 当前采样策略，合法动作上 softmax）。策略自己的价值头照常训练（两臂一致）。
- **臂**（均从 224M 续 16M → 240M，配方 = exp106 C：lr 3e-5、α 0.003、gpi 2048、dup_k 8、γ 0.995、λ 0.95、target_kl 0.03、clip 0.2、batch 4096、1 pass、amp_update、**无 adv_clamp**（10-11 测得 5σ 裁剪只动 0.1% 梯度质量，用户同意去掉）；seed 1081）：
  - **OV**：`--oracle_critic v`；
  - **OQ**：`--oracle_critic q`；
  - **BQ**（对照）：`--oracle_critic q --oracle_blind`（同网络同训练，oracle 输入置零）；
  - **盲对照**：exp106 C / C′（224→240M，同配方，仅 adv_clamp 差异 <0.1%）。
- **评测**：每里程碑（228/232/236/240M）10k 牌山 × 2 段（seed0 102224000/102224500）T=0 vs bc49，与 exp102/103/105/106 同牌山可配对；本机自动评测器。
- **行为探针**（终点 ckpt，$0）：暴露决策出生牌率分巡目曲线（`tile_danger_probe`），对照 C 240M。
- **机器/费用**：3 × Secure 4090（$0.89/h），critic 多一次前向 + 2 pass 更新，预计吞吐 ≈ 700–850 局/s ⇒ 16M ≈ 5.5–6.5 h ≈ $5–6/臂，**合计 ≈ $16，上限 $20**。

## Success Criteria（预注册）
1. **诊断（在轨，warmup 后 1M 局均值）**：`oracle/ev`（OV：V 的 EV；OQ：Q(s,a_t) 的 EV）vs `oracle/blind_ev`（策略自己的价值头）——OV/OQ ≥ 盲 + **5 pp** 才算"用上了信息"；BQ 的 EV 应 ≈ 盲（否则是独立网络容量效应）。OQ/BQ 记 `oracle/adv_std_ratio`（Q 优势 std / 盲 GAE 优势 std）与 `oracle/adv_corr`。
2. **主判据（窗口均值 228–240M × 2 段，配对）**：OV − mean(C, C′) ≥ +0.5 pp 且点差 > 0 ⇒ H1-V；OQ − mean(C, C′) ≥ +0.5 pp 且 **OQ − BQ ≥ +0.5 pp** ⇒ H1-Q（信息本身有效）；OQ ≈ BQ > C ⇒ 收益来自独立 Q critic/all-action 优势而非 oracle（也是正结论，归入纯血杠杆）；全部 |Δ| < 0.5 pp ⇒ H0。
   seed 方差下限（exp106）：单臂窗口差 < 0.5 pp 一律"无差"。
3. **机制**：若 OQ 为正，暴露决策向听 ≥ 2 的出生牌率应低于 C（exp102 口径 0.37）。
4. **健康/早停**：三臂 KL < 0.03、H 0.2–0.4；critic 损失发散（> 2× 初值）或 EV < 0 持续 100 迭代 ⇒ 停该臂；吞吐 < 600 局/s ⇒ 报用户。
5. **运维**：TRAIN_DONE ⇒ md5 握手 ⇒ 自毁；critic ckpt (`critic_games_*.pt`) 一并镜像。

## Progress
- [10-12] 实现 + 测试（`tests/test_oracle_rust_parity.py` 3 项：Rust/Python oracle 向量 120 局含中断态逐槽相等、VecEnv 行一致性、critic 形状/盲控制/all-action 优势恒等式）+ 本机冒烟（V / Q / 盲 Q 各 4 迭代）：
  critic 参数 2.8–3.0M；无闸门时 Q 一激活 KL 0.002→0.038（归一化噪声）⇒ 加质量闸门，再冒烟 KL 保持 0.002、`oracle/active`=0 直到 EV 追上。本机吞吐比纯策略慢 ≈25–50%（critic 前向 + 2 pass）。
- [10-11 20:28 PDT] 用户批三臂。**Secure 4090 全部 Out**（含不限 vCPU），改 **Secure RTX A5000 $0.27/h**（CA-MTL-1，9 vCPU，CUDA 13.0，库存 Low 但三台都拿到）：
  OV `dnqt8sgfr5xmn1` 63.141.33.157:22022 / OQ `flq63kqcieg69r` 63.141.33.146:22041 / BQ `6ha8z3a27rc2vf` 69.30.85.237:22080。
  纯血线权重按 09-01 安全分层本可上 community，但 community 坏宿主抽奖费时，A5000 Secure 更稳。预计吞吐 ≈300–450 局/s ⇒ 16M ≈ 10–15 h ≈ **$3–4/臂，三臂 ≈ $10**（上限 $20 不变）。
- [10-11 20:40 PDT] 三臂启动（20:31），实测 **OV 232 / OQ 219 / BQ 207 局/s**（GPU 100%，A5000 被 critic 的前向 + 2 pass 顶满；8 核）⇒ 16M ≈ 19–21 h ≈ **$5.2–5.8/臂，三臂 ≈ $17**（上限 $20 内）。
  预热期三臂逐迭代 pg/H/KL/EV 完全相同（同 seed、同数据、优势未切换）⇒ 激活前为同一轨迹，A/B 干净。拉取循环 ×3、心跳、TB、自动评测器已挂。
- [10-11 21:25 PDT] **v1 中止（≈0.7M 局/臂，≈$0.7）**：345 迭代后从零初始化的独立 critic EV 仍 ≈0（OV −0.025 / OQ +0.008 / BQ −0.004，策略自身价值头 0.103），闸门因此始终未放行（KL 0.002 正常，策略未受损）。
  原因：策略的价值头靠的是 224M 局练出来的 trunk 特征；独立 critic 从随机初始化起步，0.7M 局远不够（exp67 的 critic trunk 来自 bc65，不存在这个问题）。16M 局内大概率也追不上 ⇒ 结果会是"从未激活"的无效空。
  修正：`--oracle_init_from_policy`——critic 的 stem/blocks/scalar_fc 从策略 ckpt 复制（60 个张量），oracle 投影与头全新；之后训练完全独立。本机冒烟：Q critic EV 第 2 迭代 0.053、第 4 迭代 0.075（盲头 0.10），闸门会在 ≈0.08 放行。
  三臂同配置重发（seed 1081、warmup 200、margin 0.02 不变）；v1 日志保留在 `experiments/exp108_*_v1/`（TB `exp108_*_v1_scratch`）。
- [10-11 22:35 PDT] **v2 预热后读数（340 迭代 ≈0.7M 局）**：
  | 臂 | critic EV（it 190–200） | 策略盲头 EV | 激活 | 激活后 KL | H |
  |---|---|---|---|---|---|
  | OV | **0.060**（0.071→0.057 下行） | 0.101 | 从未 | — | 0.276 |
  | OQ | **0.088** | 0.101 | 间歇（83%） | 0.005–0.006 | **0.273 → 0.202** |
  | BQ | **0.085** | 0.101 | 间歇（37–57%） | 0.004–0.005 | **0.273 → 0.206** |
  ①**判据 1（信息）已失败**：oracle Q 比盲 Q 只多 0.001–0.003 EV（门槛 +5 pp）；oracle V 甚至低于策略自身的盲头。exp67 的"隐藏信息不增加可解释方差"在独立、trunk 预训练、2 pass 的 critic 上**复现**。
  ②**Q 模式的尺度问题**：Q 优势 std 只有盲 GAE 优势的 1.9%（`adv_std_ratio`），按自身 std 归一化后 100 分级的动作差被放大成单位尺度的推力，策略 100 迭代内 H 0.27→0.20，与 exp105 熵塌缩的损伤模式同形；且闸门在 0.08 阈值附近来回翻转（优势来源交替）。新增 `--oracle_adv_norm blind`（按盲 GAE 优势 std 缩放 ⇒ 单位真实价值的步长与盲配方一致，只有噪声不同）；冒烟激活后 KL 0.002–0.003、H 0.29–0.31 稳定。
- [10-11 22:50 PDT] **用户裁决：停 OV/OQ（oracle 信息判据 1 失败 ⇒ H0），BQ 修正后重发**。OV/OQ 工件（latest.pt、critic_latest.pt、train_log）镜像到 `exp108_{OV,OQ}_v2/` 后终止 pod（v1+v2 合计 ≈$1.5）。
  **BQ v3**：`--oracle_critic q --oracle_blind --oracle_init_from_policy --oracle_adv_norm blind --oracle_gate_margin 0.03`（EV 0.085 vs 盲头 0.101，0.02 的 margin 会在阈值上翻转），其余不变。
  问题改写为：**all-action Q 优势（独立 Q critic，无 oracle）是否是纯血降方差杠杆**——对照 exp106 C/C′，判据同主判据 2（BQ − mean(C,C′) ≥ +0.5 pp 且点差 > 0）。
- [10-11 23:30 PDT] **BQ v3 预热后（350 迭代）**：闸门稳定放行（active 1.00）、KL 0.002、EV_q 0.084 vs 盲头 0.102；但 **pg_loss ≈ 0（|pg| 0.00005 vs 盲 0.001）、H 0.27 → 0.33 单调上行**。
  解释：按盲 std 缩放后 Q 优势只有 0.02 的量级，策略梯度项被 α=0.003 的熵项压过——盲配方里 H≈0.28 的平衡其实是"噪声优势的自我锐化 + 真实信号" vs α，去掉噪声后锐化项消失，熵只能往上漂。
  ⇒ 两种缩放都不"公平"：自归一化（v2）推力 50×、熵塌缩；盲 std 缩放（v3）推力不足、熵发散。正确做法是**固定熵水平、放开 α**：v4 = 自归一化 + 训练器现成的对偶熵控制器（`--entropy_auto --entropy_floor 0.28`，= exp106 C/C′ 的均值水平，`--entropy_coef 0.003` 初值，dual_lr 0.1），其余不变。
  本机冒烟 10 迭代：激活后 KL 0.011–0.021（target 0.03 内）、pg −0.03、H 0.25–0.30、α 开始响应。记录偏离：BQ 的 α 不再是常数（对照 C/C′ 为常数 0.003），熵水平匹配、更新预算（target_kl/clip）相同。v3 日志归档 `exp108_BQ_v3_blindnorm/`。
- [10-12 00:45 PDT] **BQ v4 预热后（420 迭代）**：闸门稳定放行；激活瞬间 H 0.29→0.18（100 迭代），控制器把 α 从 0.0012 推到 **0.08**（盲配方的 27×）才把 H 拉回 0.27 并继续上行——振荡周期 ≈300 迭代、幅度 0.18–0.33，正在收敛。KL 0.005（盲 0.002），pg −0.005，决胜率 0.88–0.90 正常。
  读法：自归一化的 all-action Q 优势是一股比盲梯度强一个数量级的锐化力，需要很大的 α 才能维持同样的熵——训练目标已接近"软 Q"（π ∝ exp(Q/τ)）体制。按预注册继续，等 228M 配对读数。

## Results
（待）

## Conclusion
（待）

## Artifacts
（待）
