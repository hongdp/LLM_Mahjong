# LLM Mahjong — 日麻 RL（双谱系）

[English](README.md) | 中文

> 英文版 [README.md](README.md) 是权威版本；本文为其中文对照。

自对弈 RL 训练的日本麻将智能体，两条并行主线（2026-08-27 起）：

- **人类先验谱系**（当前架构迭代主载体）：北极星 = 从人类牌谱先验迭代出**简单输入平面**模型，综合实力超过 Mortal。
  路径：凤凰卓 BC（bc49）→ 先验上做 RL（exp46 修好训练器四病因）→ 半庄排位训练（exp55-D）。
- **纯血谱系**（AlphaZero 式）：零人类数据、零外部模型作为训练信号，从随机初始化自己发现整个技能栈。
  人类/教师系模型只做标尺，永不进冠军谱系或对手池；情景课程永久否决；自身冻结快照的联赛与 EMA 自锚算纯。

**必读**：[CLAUDE.md](CLAUDE.md)（规则 + 项目地图）· [SKILLS.md](SKILLS.md)（教训 / 硬件 / 坑）·
[experiments/INDEX.md](experiments/INDEX.md)（每个 run 一行）· [experiments/FINDINGS.md](experiments/FINDINGS.md)（结果台账）。
**要跑最强模型**：[docs/champion_model.md](docs/champion_model.md)（模型卡 / 四种跑法 / 资源 / 版本历史）·
[experiments/LEADERBOARD.md](experiments/LEADERBOARD.md)（现行榜单）。

## 当前状态（2026-09-26）

**部署冠军 = bc49**（人类先验谱系；ConvFormer × v3r 编码 × 46 槽动作头，2.00M 参数，纯 BC，凤凰卓 18.5k 局，holdout 精度 0.806）。
实战与终审一律 T=0 贪心。半庄全贪心梯子（最干净的刻度，13 锚，每对 200 复式）：Mortal 298k 1465 ± 12、exp46-I 1457 ± 11、**bc49 1443 ± 11**；
对 Mortal 头对头 1,200 场 0.477 ± 0.014——打平：「平 Mortal」达成，「超」未达成。雀魂 maka 两轮 S+。
配置与 ckpt 路径见 [docs/champion_model.md](docs/champion_model.md)，数字见 [experiments/LEADERBOARD.md](experiments/LEADERBOARD.md)。

**纯血谱系——$200 计划（2026-09-06 → 09-21，已用 ≈$92）**。目标：从零 RL 的模型达到 bc49 / Mortal 水平
（T=0 对 bc49 单局配对牌山 share ≥ 0.50，≥20k 对；半庄 ≥ 0.50，≥1,200 场）。
目前最强 **exp88_Q1x：对 bc49 单局 0.4568 ± 0.0035**；半庄 0.37–0.39。终点数字全在 [experiments/INDEX.md](experiments/INDEX.md)，
计划书与判决在 [experiments/designs/design_pure_line_200usd_program.md](experiments/designs/design_pure_line_200usd_program.md)。
exp70–exp97 的结论：

- **缩放是对数线性的**，每倍增 ≈ +1.5–2 pp；同配方续训渐近 0.45–0.46（学习率/熵退火已尽，exp78/80；高熵正则 +1 pp，exp88）。
- **已关闭的杠杆**（每项预注册、每项用 20k 对 T=0 终评判决）：半庄/顺位目标（exp70/83/84/91）、价值方法 ×4（exp59/60/71/96）、
  PBRS（exp73/74）、反事实 rollout 优势（exp72）、PSRO 最佳响应（exp87）、QRE/KL 磁铁（exp88）、风格条件化人口（exp89）、
  全局与向听加权放铳惩罚 + 退火（exp85/90/95——惩罚一撤防守就弹回）、集成（exp92）、运行时适应（exp93）、oracle guiding 与待张辅助头（exp68/69）、
  同谱系对手池（exp82）、推理时搜索（exp86）。
- **诊断**（exp94 反事实接管探针，真牌桌克隆、全贪心、按局聚类 SE）：与 bc49 的差距是一项技能——远手面对立直时的**多步**续打。
  bc49 从暴露决策起接管值 **+258 ± 94 分/决策**，全部来自放铳 16.5% → 11.5%、和牌率不变；任何**单步**改动对两种学习者、在两种桌上都 ≈0。
  策略梯度只看得到单步量，所以这条链爬不上去。第一个为正的纯血宏：「向听 ≥2 时只切现物、集内由模型自家牌效选」= +108–153 分/决策（z≈3，18k 局），折合 ≈ +19 ± 9 分/局。
- **09-21 后关闭**（均 H0，均预注册）：参数空间噪声（exp97，无 IS 口径训练器失效）、贪心续打+单点偏离（exp98）、学到的危险度安全集宏（exp99）、模式序列 PIMC 搜索（exp100）、静态留牌。
  博弈论防线（对手池、PSRO、熵/QRE、风格人口）也量过：人口坐在一个没人能剥削的均衡上，但是"低质量均衡"——双方缺同一项多步技能；无循环、无退化。
  分巡目探针显示模型在最后几巡已会弃和（出生牌率比机会率低 96%），缺口向序盘张开。
- **在跑**：**exp101**——最后一条未试的轴，容量 × 规模：`cnn_l_v3r`（4.0M 参数，cnn_m 的 2×）从零到 4 亿局，单张 Secure RTX 4000 Ada（$0.28/h，≈408 局/s，上限 $100）。
  判决点 32M / 64M / 112M（对 bc49 ≥0.462 = 容量兑现；≤0.452 = 预算转 Q1x 纯规模）；终点 ≥0.50 达标。累计已用 ≈$111 / $200。

**基础设施**：牌局引擎已移植到 Rust（`rust/riichi_rs`，与 Python 引擎位级一致，单张 RTX 2000 Ada ≈575 局/s，每百万局成本 $3.1 → $0.07）。
训练在 RunPod（pod 上有冠军权重时一律 Secure Cloud）；本地 RTX 4080 只做冒烟、探针和评测。

## 两个阶段

- **Phase 1（2026-05 → 08-14，已归档）**：LLM（Qwen + LoRA）+ 文本 rollout + PBRS/PPO。竞技场全 null、回报不可解码 → 退役。
  遗产：引擎、奖励 registry、竞技场协议、GCP 工作流。档案：`experiments/reports/report_exp1..exp5`、`src/core/`、`scripts/phase1_ce/`。
- **Phase 2（当前）**：小型专用网络（2–23M 参数）+ 张量编码 + 自对弈 PPO / BC / DQN。

## 架构（Phase 2 活跃部分）

```
src/tasks/mahjong/
├── table.py            # 136 张牌桌引擎；纪元 4 规则 = 雀魂单局（赤宝牌/途中流局/双倍役满/流局满贯/
│                       #   明杠宝牌时机/国士抢暗杠/抢杠振听）+ 场因素随机化；奖励 = 起点差分（+ 顺位奖）
├── claims.py           # 响应窗口裁决（和 > 碰杠 > 吃，双响/三响）
├── hanchan.py          # 完整半庄：连庄/本场/流满/uma——裁决刻度
├── arena.py            # 复式牌竞技场（A−B 对称配对）
└── shanten.py          # 向听/受入/宝牌映射
rust/riichi_rs/         # 引擎 + 编码器 + VecEnv 的 Rust 移植（向量化自对弈、半庄、联赛席位路由、
│                       #   逐席奖励风格、现物掩码）；与 Python 引擎位级一致
src/agents/dnn/
├── encoder.py          # 观测编码 v1/v1r/v3/v3r(+赤)/v3s/v4；374 槽与 46 槽动作空间
├── arch_zoo.py         # cnn_m_r / cnn_m_v3r（纯血线）/ convformer_m_v3r_m46（bc49）/ 集成 /
│                       #   AnchoredQPolicy（先验锚定 Q 头，exp96）/ handset / HandRiverFormer / vit
├── net.py              # 基类 + load_compatible（跨动作空间/变体加载）
├── selfplay.py         # Python 引擎自对弈（play_game / 生成器 play_game_gen）——探针与 bc49 桌
├── rust_rollout.py     # Rust 引擎 rollout（collect_rust）：联赛席位、奖励风格、uint8 平面
├── parallel_rollout.py # 多进程 Python rollout + GPU 批推理（infer_server.py）
├── style_stats.py      # 能力指标（和牌/放铳/立直/副露率、巡目）——TB 与评测共用
└── mjai_bridge.py      # 雀魂实战桥接（MJAI 影子桌；编码器与合法动作零改动复用）
scripts/
├── train_dnn_ppo.py    # PPO 训练器：--engine rust、dup_k=8 复式牌组基线、GAE、熵时间表/目标熵对偶控制、
│                       #   KL 锚、联赛池、PBRS 塑形、放铳惩罚（全局/向听加权/退火）、反事实优势
├── train_dnn_bc.py     # 天凤牌谱行为克隆（bc49 配方）
├── train_dnn_dqn.py / train_dnn_qstitch.py   # 价值方法线（先验上 Double DQN；先验锚定 Q-stitch）
├── run_elo_league.py   # 锚池 Elo（单局与 --hanchan，向量化 GPU，引擎指纹守卫）
├── rating.py           # 评分体系 v2（只追加账本、四实体同桌、pt 主榜）——纪元 7 待开
├── probe_*.py / eval_style_profile.py         # 探针族：防守 IQ、牌效拆分、条件熵、风格
├── serve_mjai_bot.py   # 雀魂桥接用 HTTP 代理服务
└── phase2_dnn/         # 云发射脚本（GCP G4 flex；RunPod runbook 在 SKILLS.md）
tools/webui/            # 检视台：训练曲线、自对弈看板（逐步概率/V）、复盘
tools/majsoul_bridge/   # MahjongCopilot 插件（实战 = 冠军贪心；三把人类尺之一）
```

## 评估体系

三个刻度不可混列（见 [experiments/LEADERBOARD.md](experiments/LEADERBOARD.md) 顶部说明）：

1. **锚池 Elo**（`experiments/elo_league/`）：13 锚点 sign-MLE，`bc_cnn` 钉 1000。引擎变更 ⇒ 历史作废、整池重校（引擎指纹守卫）。锚在自己被标定的温度下应战（写在池文件里）。
2. **半庄刻度**（`hanchan.py`，自己的锚池在 `experiments/elo_league/hanchan/`）：裁决刻度。向量化 GPU 评测 ≈967 场/分；Mortal 用 `rate_mortal_hanchan.py` 评分。
3. **探针与人类尺**：defense IQ、风格剖面（人类参照 agari .212 / houjuu .125 / riichi .182 / call .338）、反事实接管探针（exp94）、雀魂 maka 档位。

**评测协议**（CLAUDE.md）：终审一律候选 T=0 + 族外梯子 + 半庄 n ≥ 300；单局终评 = 两段种子 20k 配对牌山（SE ≈ 0.0035）；T=1 族内曲线不得单独下结论。
每局取多个决策的探针必须报**按局聚类**的标准误（逐决策 SE 低估约 2×）。加冕只认双方 T=0 头对头显著为正，梯子分只排名不加冕。

## 快速开始

```bash
conda activate rlhf_mahjong
python -m pytest tests -q                              # 引擎一致性、编码器、训练器、Rust ↔ Python
# 构建 Rust 引擎（改动 rust/riichi_rs 后）
cd rust/riichi_rs && maturin build --release -i "$(which python)" && pip install --force-reinstall target/wheels/riichi_rs-*.whl && cd -
# 跑冠军（bc49，贪心）给雀魂桥接用——完整手册见 docs/champion_model.md
PYTHONPATH=. python scripts/serve_mjai_bot.py --ckpt experiments/_anchors_epoch6/bc49.pt --temperature 0
# 纯血线 PPO 冒烟（Rust 引擎；长训练一律上云——RunPod runbook 见 SKILLS.md）
python scripts/train_dnn_ppo.py --engine rust --arch cnn_m_v3r --total_games 200000 --games_per_iter 2048 \
  --dup_k 8 --games_per_worker 1024 --workers 1 --exp_dir experiments/my_run_$(date +%Y%m%d_%H%M%S)
# 某 ckpt 对 bc49 的 T=0 配对评测（单局 / 半庄）
python -c "from scripts.run_elo_league import play_pair_vector as p; sc,_,_=p('CAND.pt','experiments/_anchors_epoch6/bc49.pt',4000,67000000,20,'cuda',temp_a=0.0,temp_b=0.0,hanchan=False); print(sum(sc)/len(sc))"
conda run -n rlhf_mahjong python tools/webui/server.py --port 8642   # 检视台
```

每个 run 发射**前**要有 `EXPERIMENT.md`（目的/方法/成功标准），发射后挂心跳与 TensorBoard 镜像，收尾在 `experiments/INDEX.md` 记一行——见 CLAUDE.md。

## 雀魂实战测试（Windows 打牌机）

人类刻度（maka 档位 / 顺位 / 放铳）只能在真实对局上读出来。标准拓扑是两台机——模型机跑本仓库与 checkpoint，打牌机（Windows）跑
[MahjongCopilot](https://github.com/latorc/MahjongCopilot)（MC）+ Chrome，用 SSH 隧道连起来；单机部署也可行。

```
打牌机 Windows 11: MC + 插件 ── mitmproxy:10999 ──► Chrome(雀魂)
                       └─ bot_llmmahjong ──► 127.0.0.1:8765 ──ssh -L 隧道──► 模型机: serve_mjai_bot.py
```

1. **模型机**（Linux，仓库根目录）启动 agent 服务，冠军 + 贪心：
   ```bash
   PYTHONPATH=. python scripts/serve_mjai_bot.py --ckpt experiments/_anchors_epoch6/bc49.pt \
     --temperature 0 --log experiments/exp24_majsoul_live_$(date +%Y%m%d_%H%M%S)/mjai_session.jsonl
   ```
   `curl localhost:8765/health` 返回 ok 即就绪。服务**无鉴权、只监听 127.0.0.1**，不要暴露到公网。
2. **打牌机装 MC**（PowerShell；不要用机器上的老 conda）：
   ```powershell
   winget install Python.Python.3.12 --scope user
   git clone https://github.com/latorc/MahjongCopilot $env:USERPROFILE\MahjongCopilot
   cd $env:USERPROFILE\MahjongCopilot; python -m venv venv; .\venv\Scripts\pip install -r requirements.txt; .\venv\Scripts\playwright install chromium
   ```
3. **打三处 Windows 补丁 + 装 bot 插件**（补丁基于 MC `31be3de` 验证）：
   ```powershell
   git apply <本仓库>\tools\majsoul_bridge\mahjongcopilot_windows.patch
   python <本仓库>\tools\majsoul_bridge\install.py $env:USERPROFILE\MahjongCopilot
   ```
   三处补丁对应 Windows 上的三个坑：Playwright 自带 Chromium 的 SxS 报错（改用系统 Chrome）、雀魂 46 MB wasm 被 mitmproxy 缓冲导致黑屏（大响应流式透传）、
   mitm 根证书要管理员（改装当前用户存储）。
4. **开隧道**（打牌机，常驻）：`ssh -N -L 8765:127.0.0.1:8765 <模型机>`；MC 侧 URL 保持 `http://127.0.0.1:8765`。
5. **配置并启动 MC**：`settings.json` 里 `"model_type": "LLM_Mahjong"`、`"llmmahjong_url": "http://127.0.0.1:8765"`、`"ai_randomize_choice": 0`；
   `enable_automation` = `false` 辅助模式（自己点，面板看概率/V）、`true` 自动打牌（正式计分）。
   启动：`cd $env:USERPROFILE\MahjongCopilot; .\venv\Scripts\python.exe main.py` → 启动浏览器 → 登录雀魂。
6. **验收 + 计分**：先在友人房跑一局，确认 MC 日志里每个 `Bot in: tsumo` 都有 `Bot out: dahai`、`no op list` 为 0；
   打完在模型机上 `python scripts/analyze_majsoul_session.py <session>.jsonl` 出顺位/和牌/放铳/立直/副露。

**详细 runbook**：[tools/majsoul_bridge/README.md](tools/majsoul_bridge/README.md)（通用流程、两种模式、牌局留底格式、协议坑）·
[tools/majsoul_bridge/WINDOWS.md](tools/majsoul_bridge/WINDOWS.md)（Windows 实录：三处补丁的现象与根因、新版 Unity 客户端兼容、排障速查）·
[docs/champion_model.md](docs/champion_model.md)（用哪个 ckpt、要多少资源）。

> **风险**：使用第三方自动化工具违反雀魂服务条款，**存在封号风险**，只用可承受损失的账号。

**纪律**（CLAUDE.md 强制）：每个 run 先预注册；发射后核对吞吐；每个长跑任务挂心跳；云机用完即终止（不是停止）；奖励逻辑走 registry；新教训带日期追加 SKILLS.md。
