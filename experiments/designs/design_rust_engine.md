# Rust 引擎移植（riichi_rs）：把纯血线每百万局成本再降一个量级（2026-09-08 立项，用户批准）

## 0. 动机与量化依据
- 现状：L40S Secure $1.09/h，单臂 97 局/s ⇒ $3.1/百万局；GPU 利用率 7–12%，瓶颈全在 Python rollout（16 vCPU 只开 6 worker）。
- 剖析（单进程，CPU 推理，82 ms/局）：网络推理 ≈50 ms（GPU 路径已剥离）、**引擎 ≈22 ms**（table.py 6.3 + 向听库 14.1 + 内建 ~2）、**编码 ≈6.5 ms**、胶水 ~2 ms；
  GPU 路径实测 11–15 局/s/worker = 65–90 ms/局/worker ⇒ 另有 ~35–50 ms 的多进程/RPC/pickle 同步开销。
- 目标：引擎 + 编码 + 批处理循环整体进 Rust（libriichi 式 VecEnv），Python 只剩网络前向与 PPO 更新。
  预期：每核 ≥10× rollout 吞吐（Python 毫秒级/步 → 微秒级/步），端到端 5–10×（GPU 前向成为新瓶颈：1,000 局/s ≈ 84k 决策/s，L40S 批 2048 的 cnn_m 前向可达数十万/s）。
  信心：引擎层 ≥10× 高；端到端 10× 中高（5× 保底）。
- 叠加社区 3090（$0.22/h）+ 满核：每百万局 ≈ $0.05–0.1（现 $3.1）。1.3 亿局 ≈ $10–15，这才是让规模路线进入 $200 预算的唯一途径。

## 1. 契约：与 Python 引擎位级一致（parity）
- **发牌一致**：重实现 CPython `random`（MT19937 init_by_array 种子、random()、getrandbits、_randbelow、shuffle、randrange、choice、uniform、gauss（含缓存）、choices），
  `random.seed(deal_seed)` 后的牌山、王牌、起手、庄家/场风、randomize_round 的点数/供托与 Python 完全一致 ⇒ 配对牌山评测与 dup_k 组基线跨引擎可比。
- **规则一致**：合法动作列表（字符串与顺序）、打断动作、step/step_ron/step_interrupt/advance_turn/resolve_pending_kan 的 rewards/done/info、
  result_summary 字符串（下游 style_stats/yaku_features 解析）、final_rewards、points。役种名 = 计分库 `str(yaku)`（"Riichi"、"Menzen Tsumo"、"Dora N"、…）。
- **计分一致**：重实现 `mahjong` 2.0.0 库的 divider/fu/yaku/scores（has_open_tanyao、aka、double yakuman、kazoe 默认=役满、renhou 关），
  含 _find_win_groups 的多分解取最高（han, fu, Σfu_details）。差分测试：随机和牌形 ≥10 万手 vs 库。
- **编码一致**：v1r 的 21×34 平面与 20 标量位级一致；动作空间 native 374 的 mask/lookup 一致。
- 非契约：引擎内部 `random.choice`（非法动作的强制弃牌）在 Python 端本就跨局共享全局 RNG、不可复现，Rust 用表内 RNG 即可。

## 2. 架构
```
rust/riichi_rs/            PyO3 扩展（maturin develop --release 装进 rlhf_mahjong 环境）
  src/pyrandom.rs          CPython random 位级移植
  src/tiles.rs             牌表示（34 索引 + 红五标记）、字符串互转（'0m'/'5m'/'1z'…）
  src/shanten.rs           向听/听牌/待张（一般形/七对/国士；等价于 mahjong.shanten 结果）
  src/score.rs             分解器 + 符 + 役 + 点数（等价 HandCalculator）
  src/table.rs             PyMahjongTable 语义（reset/legal/interrupt/step/ron/kan/advance/settle/abort/ryuukyoku）
  src/encoder.rs           v1r 平面/标量 + native 动作 mask/lookup
  src/vecenv.rs            K 桌交错驱动：一次返回全部待决策的 (planes, scalars, mask, seat, game) 批；接收动作索引批
src/agents/dnn/rust_rollout.py   Python 侧：VecEnv ↔ 推理（本进程 GPU）↔ episode 打包（与 _package_game 同格式）
tests/test_rust_parity.py        差分测试：随机策略（共享 RNG 选合法动作序号）同步驱动两引擎，逐步比对
```
- 阶段化交付，每阶段有 parity 门：
  - **P0** crate/构建/CPython random 位级一致（seed/random/shuffle/gauss/choices 各 1e5 次）。
  - **P1** 牌山与 reset 一致（1e4 种子：wall/dead_wall/hands/dealer/rw/points/kyotaku）。
  - **P2** 向听/待张一致（1e6 随机手 vs 库）；计分一致（1e5 随机和牌形 vs 库，含副露/红五/宝牌/立直/场况）。
  - **P3** 全局对局一致（1e4 局随机策略逐步比对合法动作/奖励/结果串）。
  - **P4** v1r 编码位级一致；VecEnv 集成 PPO 训练器（`--engine rust`），冒烟 + 吞吐基准；exp76 = 同配方同种子对照（Rust 引擎训出的模型 vs P3 头对头 ≈0.50 才算通过）。
  - **P5**（可选）半庄层（HanchanTable/MatchState）与 v3/v3r 编码。
- 引擎指纹守卫：Rust 引擎是 Python 引擎的等价实现，评测仍以 Python 引擎为裁决口径直到 P4 对照通过；之后评测也可切 Rust（同牌山）。

## 3. 风险
- 计分库细节（分解顺序、fu_details 排序、双役满、kazoe）——用差分测试兜底，不靠阅读。
- 浮点：randomize_round 的 gauss/log/cos 用同一 libm（Linux glibc），预期位级一致；若个别种子差 1 ulp 导致 round 半整不同，记录为已知偏差（只影响起手点数场景）。
- 工程量：预计 5–8 个工作日等价；分阶段提交，任何阶段可停且已有价值（P2 的向听/计分即可替换 Python 热点，单独 1.3–1.5×）。
