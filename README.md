# LLM Mahjong — 日麻 RL（双谱系）

[English](README.en.md) | 中文

**两条并行主线（2026-08-27 起）**：
- **人类先验谱系（当前架构迭代主载体）**：北极星 = 用人类牌谱先验迭代出**简单输入平面**模型，综合实力超过 Mortal。
  路径：凤凰卓 BC（两万局 ≥ 两百万局自对弈，exp45）→ RL 增益（exp46 系列已修好训练器四病因）→ 半庄排位训练（exp55-D）。
- **纯血谱系（AlphaZero 式）**：零人类/教师知识，从随机初始化自己发现技能栈；规则不变——教师/人类系模型只做标尺，
  永不进其冠军谱系或对手池；情景课程永久否决；联赛（对手=自身冻结历史）与 EMA 自锚算纯。

**必读**：[CLAUDE.md](CLAUDE.md)（规则+项目地图）· [SKILLS.md](SKILLS.md)（教训/硬件）·
[experiments/INDEX.md](experiments/INDEX.md)（总账）· [experiments/FINDINGS.md](experiments/FINDINGS.md)（结果台账）
**要跑最强模型**：[docs/champion_model.md](docs/champion_model.md)（冠军模型卡：配置 / 四种跑法 / 资源需求 / 版本历史）·
[experiments/LEADERBOARD.md](experiments/LEADERBOARD.md)（现行榜单）

## 两个阶段

- **Phase 1（2026-05~08-14，已归档）**：LLM（Qwen + LoRA）+ 文本 rollout + PBRS/PPO。
  结论：竞技场全 null、回报不可解码 → 路线退役，遗产 = 引擎、奖励 registry、竞技场协议、GCP 工作流。
  档案：experiments/reports/report_exp1..exp5、`src/core/`、`scripts/phase1_ce/`。
- **Phase 2（当前）**：小型专用网络（2–23M）+ 张量编码 + 纯自对弈 PPO。一条 1.0M 局训练 =
  g4-standard-48 flex 上 ~85 分钟、~$3。

## 架构（Phase 2 活跃部分）

```
src/tasks/mahjong/
├── table.py            # 136 张牌桌引擎；纪元 4 规则 = 雀魂单局对齐（赤宝牌/途中流局/双倍役满/
│                       #   流局满贯/明杠宝牌时机/国士抢暗杠/抢杠振听）+ 场因素随机化
│                       #   （东1 恒等起点，点差 σ=4500√k 随局数增长，供托/西场，奖励=起点差分+顺位奖）
├── claims.py           # 响应窗口裁决（和>碰杠>吃、双响、三响流局）
├── arena.py            # 复式牌竞技场（A−B 对称配对分差；同 seed 同场上下文）
└── shanten.py          # 向听/受入/宝牌映射
src/agents/dnn/
├── encoder.py          # 观测编码 v1/v1r(+赤/役牌平面)/v3(完整公开记录)/v4(事件缓冲)；
│                       #   374 动作空间（11 类型×34 关键牌，老 checkpoint 自动加宽）
├── arch_zoo.py         # cnn_m_r(冠军 2M) / cnn_xl_r / handset_*(实例集合注意力) /
│                       #   HandRiverFormer(exp30：手牌 token cross-attend 牌河事件序列) / ConvFormer / vit
├── net.py              # 基类 + load_compatible（跨动作空间/变体的 checkpoint 加载）
├── selfplay.py         # 自对弈（play_game / 生成器版 play_game_gen）、DnnGame 风格事实
├── parallel_rollout.py # 多进程 rollout；向量化 worker（每进程 K 局一次批量 RPC，cnn 204 局/s 本机）
├── infer_server.py     # GPU 批推理服务（共享内存槽位/CUDA graph 分桶/多模型托管）
├── style_stats.py      # 能力指标聚合（和牌/放铳率与巡目、立直/副露率）——训练 TB 与评估共用
└── mjai_bridge.py      # 雀魂实战桥接（MJAI 影子桌，编码器/合法动作零改动复用）
scripts/
├── train_dnn_ppo.py    # PPO 训练器（GAE λ=0.95、dup_k=8 复式牌 leave-one-out 基线、熵时间表/
│                       #   目标熵对偶控制、混合温度行为策略 logprob、--gpu_infer、style/* TB 指标）
├── run_elo_league.py   # Elo 锚点池（纪元 6 = 13 员，混动作空间同池，含引擎指纹守卫、--temperature 贪心评分）
├── elo_ladder_watcher.py / watch_run.sh   # 训练中阶梯评分 + 心跳（每个长跑任务必挂）
├── probe_defense.py / probe_decomposition.py / probe_conditional_entropy.py / eval_style_profile.py
│                       # 探针族：防守 IQ / 牌效拆分 / 条件熵曲线 / 风格（--vs_anchors 生态无关读数）
├── run_arena_dnn.py    # 复式竞技场（--override_* 诊断包装、每边独立温度）
└── phase2_dnn/         # 云工作流：launch_g4_git.sh（G4 flex + git 固定 SHA 门）、run_dnn_cloud.sh
tools/webui/            # 检视台：训练曲线 + 自对弈看板（逐步概率/V）+ 雀魂式复盘
tools/majsoul_bridge/   # MahjongCopilot 插件（实战 = 冠军贪心；maka/顺位/放铳三把尺之一）
```

## 评估体系（三把尺）

1. **Elo 锚点池**（`experiments/elo_league/`）：13 锚点 sign-MLE，bc_cnn 钉 1000；纪元 6 现役。
   **纪元规则**：引擎变更 ⇒ 历史作废、整池重校（引擎指纹守卫）。**评测协议（2026-08-30 定）**：
   终审一律候选 T=0 + 族外梯子 + 半庄 n≥300；T=1 族内曲线禁止单独下结论（快路径 `play_pair` 同空间 ~23×）。
2. **半庄刻度**（`src/tasks/mahjong/hanchan.py` + `run_hanchan_arena`）：连庄/本场/流满/uma 全套，
   1.8× 单局放大；这是裁决刻度。训练侧 = exp55-D 四席桌 + 排位价值 W 逐局归因。
3. **探针族**：defense_iq、风格剖面（对人类精确参照：agari .212/houjuu .125/riichi .182/call .338）。
4. **人类刻度**：雀魂实战（贪心）——maka 档位：纯血冠军 C+ → **人类 BC 旗舰 bc49 两轮 S+**。

## 当前状态（2026-08-30）

- **部署冠军 = bc49**（人类先验谱系，conv×v3r×46 全量 BC，2.00M 参数 / 56 平面 / 46 动作）：
  纪元 6 部署刻度 **1189.0 ± 7.9**，雀魂 maka 两轮 S+；真 Mortal 298k 参照 1199.6 ± 8.0（同协议）
  ——**北极星差距 ≈ 10 ± 11**。怎么配置、怎么跑、要多少机器 → [docs/champion_model.md](docs/champion_model.md)。
- **纯血谱系冠军 = exp27-A**（不变，纪元 6 池内 1064.5）。
- **exp46 C~J 收官**：定位并修复训练器四病因（价值梯度扰乱 trunk→`--value_detach`、熵扩散→KL 锚、
  优势尾部审查→去 clamp、T=1 族内度量失真→换协议）；最强 RL 产物 **exp46-I**（锚×detach）
  T=0 族外 **1210.0±8.4 = 与冠军统计持平**（首个不毁本金的 RL）。详见
  [experiments/exp46_rl_on_prior_prereg/EXPERIMENT.md](experiments/exp46_rl_on_prior_prereg/EXPERIMENT.md)。
- **纪元 6 已开闸并完成重校**（引擎动作缺口修复合并 + T=0 协议 + 13 锚全体重锚 + 混动作空间原生托管）：
  完整榜单见 [experiments/LEADERBOARD.md](experiments/LEADERBOARD.md)。
- **exp55-D 就绪**：半庄排位训练管线全冒烟（残差 W 信用 + v3rh 编码器 + 四席 rollout），待发射。
- **待决**：exp54 离线 RL 立项；数据扩容（10 万局级，当前最高期望值杠杆）。

## 快速开始

```bash
conda activate rlhf_mahjong
python -m pytest tests -q                        # ~196 项
# 跑现役冠军 bc49（雀魂实战服务，T=0 贪心）——完整手册见 docs/champion_model.md
PYTHONPATH=. python scripts/serve_mjai_bot.py --ckpt experiments/_anchors_epoch6/bc49.pt --temperature 0
# 本地训练（4080 实测 cnn_m_r ~100 局/s 训练口径）
python scripts/train_dnn_ppo.py --arch cnn_m_r --total_games 1000000 --gpu_infer \
  --games_per_worker 32 --infer_max_batch 512 --exp_dir experiments/my_run_$(date +%Y%m%d_%H%M%S)
# 云端（G4 flex，先 push 再发射——脚本会校验 SHA 已在 origin/master）
bash scripts/phase2_dnn/launch_g4_git.sh my-vm us-central1-b my_run $(git rev-parse HEAD) -- \
  scripts/train_dnn_ppo.py --arch cnn_m_r ... --exp_dir experiments/my_run
conda run -n rlhf_mahjong python tools/webui/server.py --port 8642   # 检视台
```

## 雀魂实战测试（Windows 打牌机）

人类刻度（maka 档位 / 顺位 / 放铳）只能在真实对局上读出来。标准拓扑是**两台机**——模型机跑本仓库与
checkpoint，打牌机（Windows）跑 [MahjongCopilot](https://github.com/latorc/MahjongCopilot)（MC）+ Chrome，
两者用 SSH 隧道连起来；单机部署（两者同一台）也完全可行。

```
打牌机 Windows 11: MC + 插件 ── mitmproxy:10999 ──► Chrome(雀魂)
                       └─ bot_llmmahjong ──► 127.0.0.1:8765 ──ssh -L 隧道──► 模型机: serve_mjai_bot.py
```

1. **模型机**（Linux，本仓库根目录）启动 agent 服务，冠军 + 贪心：
   ```bash
   PYTHONPATH=. python scripts/serve_mjai_bot.py --ckpt experiments/_anchors_epoch6/bc49.pt \
     --temperature 0 --log experiments/exp24_majsoul_live_$(date +%Y%m%d_%H%M%S)/mjai_session.jsonl
   ```
   `curl localhost:8765/health` 返回 ok 即就绪。服务**无鉴权、只监听 127.0.0.1**，不要暴露到公网。
2. **打牌机装 MC**（PowerShell；机器上的老 conda 不要用）：
   ```powershell
   winget install Python.Python.3.12 --scope user
   git clone https://github.com/latorc/MahjongCopilot $env:USERPROFILE\MahjongCopilot
   cd $env:USERPROFILE\MahjongCopilot; python -m venv venv; .\venv\Scripts\pip install -r requirements.txt; .\venv\Scripts\playwright install chromium
   ```
3. **打三处 Windows 补丁 + 装我们的 bot 插件**（补丁基于 MC `31be3de` 验证过）：
   ```powershell
   git apply <本仓库>\tools\majsoul_bridge\mahjongcopilot_windows.patch
   python <本仓库>\tools\majsoul_bridge\install.py $env:USERPROFILE\MahjongCopilot
   ```
   三处补丁分别解决 Windows 上必踩的三个坑：Playwright 自带 Chromium 的 SxS 报错（改用系统 Chrome）、
   雀魂 46 MB wasm 被 mitmproxy 缓冲导致黑屏（大响应流式透传）、mitm 根证书要管理员（改装当前用户存储）。
4. **开隧道**（打牌机，常驻）：`ssh -N -L 8765:127.0.0.1:8765 <模型机>`；MC 侧 URL 保持 `http://127.0.0.1:8765`。
5. **配置并启动 MC**：`settings.json` 里 `"model_type": "LLM_Mahjong"`、`"llmmahjong_url": "http://127.0.0.1:8765"`、
   `"ai_randomize_choice": 0`；`enable_automation` = `false` 辅助模式（自己点，面板看概率/V）、`true` 自动打牌（正式计分）。
   启动：`cd $env:USERPROFILE\MahjongCopilot; .\venv\Scripts\python.exe main.py` → 启动浏览器 → 登录雀魂。
6. **验收 + 计分**：先在友人房跑一局，确认 MC 日志里每个 `Bot in: tsumo` 都有 `Bot out: dahai`、`no op list` 为 0；
   打完在模型机上 `python scripts/analyze_majsoul_session.py <session>.jsonl` 出顺位/和牌/放铳/立直/副露。

**详细 runbook**：[tools/majsoul_bridge/README.md](tools/majsoul_bridge/README.md)（通用流程、两种模式、牌局留底格式、协议坑）
· [tools/majsoul_bridge/WINDOWS.md](tools/majsoul_bridge/WINDOWS.md)（Windows 实录：三处补丁的现象与根因、
新版 Unity 客户端兼容、`spawn UNKNOWN` / 黑屏 / 证书 / 「主进程发生错误!」排障速查表）
· [docs/champion_model.md](docs/champion_model.md)（用哪个 ckpt、要多少资源）。

> **风险**：使用第三方自动化工具违反雀魂服务条款，**存在封号风险**，只用可承受损失的账号。

**纪律**（CLAUDE.md 强制）：任何 run 先写 `EXPERIMENT.md`（目的/方法/成功标准）再发射；发射后核对吞吐符合预期；
每个长跑任务挂心跳；VM 用完即删；奖励逻辑走 registry；新教训追加 SKILLS.md。
