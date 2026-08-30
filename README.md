# LLM Mahjong — 纯自对弈日麻 RL

[English](README.en.md) | 中文

**北极星（目标 a）**：受 **AlphaZero** 启发——零人类/教师知识的纯自对弈，模型从随机初始化自己发现整套技能栈
（牌效 → 立直/门清 → 价值 → 防守），向人类高手水平攀升；防守涌现是里程碑，抵达本身就是目标。
（b) 从模型学到新麻将知识、(c) 方法论迁移是后续方向。纯度规则：教师系模型只做 Elo 标尺，
永不进冠军谱系或训练对手池；情景课程（从对手立直局面开局）永久否决；联赛（对手=自身冻结历史）算纯。

**必读**：[SKILLS.md](SKILLS.md)（经验/硬件/教训，开发前先读）·
[docs/roadmap_epoch3.md](docs/roadmap_epoch3.md)（当前路线图：队列/熵与随机性/能力出场顺序）·
[experiments/INDEX.md](experiments/INDEX.md)（实验总账）

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

1. **Elo 锚点池**（`experiments/elo_league/`）：9 锚点 sign-MLE，bc_cnn 钉 1000；
   **纪元规则**：引擎变更 ⇒ 历史对局作废、整池重校（anchors.json 带引擎内容指纹，评分自动拒绝错配）。
   纪元 4 现役：教师参照线 bcrl14 1107.7，冠军 exp27A 1059（池内）/ T=0 候选 1121.8。
2. **探针族**：defense_iq（防守条件化）、拆分探针（牌效最优一致率）、条件熵曲线（随机性是否跟着价值差走，
   已验证三代模型均单调下降）、风格剖面（对固定锚点的和/铳/巡目）。
3. **人类刻度**：雀魂实战（MahjongCopilot 桥接，贪心）——maka 档位（首读 C+，n=1）、顺位/放铳统计。

## 当前状态（2026-08-23）

- **冠军 = exp27-A**（cnn_m_r，纪元 3 原生：识赤宝牌/役牌平面）：从零 1.0M 局追平旧谱系 2.1M 局，
  T=0 1121.8 为史上最高；实战部署一律 A 系贪心。
- **纪元 4 生效**：规则审查修复 + 场因素随机化（给防守/顺位压力提供单局内学习信号）。
- **进行中**：exp31 四臂（目标熵配方 × 规模复检，G4 flex）；exp30 HandRiverFormer 已预注册待发射。
- **已否定的假设**（详见 INDEX）：教师先验路线、情景课程、输入 v3、GAE×ConvFormer 可加性、
  攻击饱和自发防守、联赛催生防守、手牌实例集合优于 CNN、混合温度增益。
- **防守现状**：defense_iq≈0 且与熵水平无关——瓶颈是信用不是采样；当前主攻 = 场上下文随机化 + 顺位压力，
  下一步（需以场为单位重设计 reward，暂缓）= 多局结构。

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
