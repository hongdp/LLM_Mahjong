# exp108 — 独立 oracle critic（V / Q）在纯血 PPO 上重测

- **日期**：2026-10-12 预注册  **状态**：prereg
- **Git**：（发射前填）
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

## Results
（待）

## Conclusion
（待）

## Artifacts
（待）
