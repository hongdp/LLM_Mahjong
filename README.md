# LLM Mahjong — 日麻 RL（双谱系）

[English](README.en.md) | 中文

**两条并行主线（2026-08-27 起）**：
- **人类先验谱系（当前架构迭代主载体）**：北极星 = 用人类牌谱先验迭代出**简单输入平面**模型，综合实力超过 Mortal。
  路径：凤凰卓 BC（两万局 ≥ 两百万局自对弈，exp45）→ RL 增益（exp46 系列已修好训练器四病因）→ 半庄排位训练（exp55-D）。
- **纯血谱系（AlphaZero 式）**：零人类/教师知识，从随机初始化自己发现技能栈；规则不变——教师/人类系模型只做标尺，
  永不进其冠军谱系或对手池；情景课程永久否决；联赛（对手=自身冻结历史）与 EMA 自锚算纯。

**必读**：[CLAUDE.md](CLAUDE.md)（规则+项目地图）· [SKILLS.md](SKILLS.md)（教训/硬件）·
[experiments/INDEX.md](experiments/INDEX.md)（总账）· [experiments/FINDINGS.md](experiments/FINDINGS.md)（结果台账）

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
├── run_elo_league.py   # Elo 锚点池（纪元 4 = 9 员，含引擎指纹守卫、--temperature 贪心评分）
├── elo_ladder_watcher.py / watch_run.sh   # 训练中阶梯评分 + 心跳（每个长跑任务必挂）
├── probe_defense.py / probe_decomposition.py / probe_conditional_entropy.py / eval_style_profile.py
│                       # 探针族：防守 IQ / 牌效拆分 / 条件熵曲线 / 风格（--vs_anchors 生态无关读数）
├── run_arena_dnn.py    # 复式竞技场（--override_* 诊断包装、每边独立温度）
└── phase2_dnn/         # 云工作流：launch_g4_git.sh（G4 flex + git 固定 SHA 门）、run_dnn_cloud.sh
tools/webui/            # 检视台：训练曲线 + 自对弈看板（逐步概率/V）+ 雀魂式复盘
tools/majsoul_bridge/   # MahjongCopilot 插件（实战 = 冠军贪心；maka/顺位/放铳三把尺之一）
```

## 评估体系（三把尺）

1. **Elo 锚点池**（`experiments/elo_league/`）：9 锚点 sign-MLE，bc_cnn 钉 1000；纪元 5 现役。
   **纪元规则**：引擎变更 ⇒ 历史作废、整池重校（引擎指纹守卫）。**评测协议（2026-08-30 定）**：
   终审一律候选 T=0 + 族外梯子 + 半庄 n≥300；T=1 族内曲线禁止单独下结论（快路径 `play_pair` 同空间 ~23×）。
2. **半庄刻度**（`src/tasks/mahjong/hanchan.py` + `run_hanchan_arena`）：连庄/本场/流满/uma 全套，
   1.8× 单局放大；这是裁决刻度。训练侧 = exp55-D 四席桌 + 排位价值 W 逐局归因。
3. **探针族**：defense_iq、风格剖面（对人类精确参照：agari .212/houjuu .125/riichi .182/call .338）。
4. **人类刻度**：雀魂实战（贪心）——maka 档位：纯血冠军 C+ → **人类 BC 旗舰 bc49 两轮 S+**。

## 当前状态（2026-08-30）

- **部署冠军 = bc49**（人类先验谱系，conv×v3r×46 全量 BC）：T=1 梯子 1191.4 / **T=0 部署刻度 1210.6±15**；
  雀魂 maka 两轮 S+。真 Mortal 参照 1218.6（同协议）——**部署刻度差距 ≈ 8±20**。
- **纯血谱系冠军 = exp27-A**（不变，1052 池内）。
- **exp46 C~J 收官**：定位并修复训练器四病因（价值梯度扰乱 trunk→`--value_detach`、熵扩散→KL 锚、
  优势尾部审查→去 clamp、T=1 族内度量失真→换协议）；最强 RL 产物 **exp46-I**（锚×detach）
  T=0 族外 **1210.0±8.4 = 与冠军统计持平**（首个不毁本金的 RL）。详见
  [experiments/exp46_rl_on_prior_prereg/EXPERIMENT.md](experiments/exp46_rl_on_prior_prereg/EXPERIMENT.md)。
- **exp55-D 就绪**：半庄排位训练管线全冒烟（残差 W 信用 + v3rh 编码器 + 四席 rollout），待纪元 6 开闸后发射。
- **待决**：纪元 6（PR #8 = 开闸载体：引擎修复 + 协议切换 + 全体重锚）；exp54 离线 RL 立项待发。

## 快速开始

```bash
conda activate rlhf_mahjong
python -m pytest tests -q                        # ~196 项
# 本地训练（4080 实测 cnn_m_r ~100 局/s 训练口径）
python scripts/train_dnn_ppo.py --arch cnn_m_r --total_games 1000000 --gpu_infer \
  --games_per_worker 32 --infer_max_batch 512 --exp_dir experiments/my_run_$(date +%Y%m%d_%H%M%S)
# 云端（G4 flex，先 push 再发射——脚本会校验 SHA 已在 origin/master）
bash scripts/phase2_dnn/launch_g4_git.sh my-vm us-central1-b my_run $(git rev-parse HEAD) -- \
  scripts/train_dnn_ppo.py --arch cnn_m_r ... --exp_dir experiments/my_run
conda run -n rlhf_mahjong python tools/webui/server.py --port 8642   # 检视台
```

**纪律**（CLAUDE.md 强制）：任何 run 先写 `EXPERIMENT.md`（目的/方法/成功标准）再发射；发射后核对吞吐符合预期；
每个长跑任务挂心跳；VM 用完即删；奖励逻辑走 registry；新教训追加 SKILLS.md。
