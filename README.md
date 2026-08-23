# LLM Mahjong — 基于 LLM 的多智能体日麻自对弈 RL 系统

用 4 个共享权重的 LLM 座位进行四人立直麻将自对弈（POMDP），通过自研的解耦 RL 管线
（批量并发 rollout → ReplayBuffer → REINFORCE / PPO）训练 Qwen3.5-2B + LoRA，
从基础牌效率逐步进化到价值判断与防守。

**必读文档**：
- [SKILLS.md](SKILLS.md) — 项目经验/硬件约束/性能诊断（每次开发前先读）
- [TASKS.md](TASKS.md) — 任务看板（持久快照，随里程碑更新）
- [docs/reward_energy_pbrs.md](docs/reward_energy_pbrs.md) — 势函数奖励的数学与保证
- [docs/v3_threaded_context_design.md](docs/v3_threaded_context_design.md) — 下一代串联上下文架构设计稿
- [docs/engine_known_issues.md](docs/engine_known_issues.md) — 引擎规则保真度与有意延后项
- [docs/mahjong_design_document.md](docs/mahjong_design_document.md) / [docs/implementation_plan.md](docs/implementation_plan.md) — 原始设计与三阶段部署方案

## 架构总览

```
src/
├── core/                    # 通用 RL 引擎（与游戏解耦）
│   ├── trainer.py           # 主入口：SFT warm-up + RL 循环（REINFORCE 或 PPO）
│   ├── ppo.py               # PPO clipped-surrogate 损失（纯张量，单测覆盖）
│   ├── rollout.py           # ReplayBuffer / TrajectoryStep / Return-to-Go / 协变量基线
│   ├── chat_format.py       # 唯一的模板渲染源（SFT 与 rollout 必须一致）
│   ├── task.py / registry.py / base_reward.py
└── tasks/mahjong/
    ├── table.py             # 136 张牌桌引擎（真实算分/立直/振听/流局听牌费/结算分发）
    ├── shanten.py           # 向听/受入/宝牌映射
    ├── orchestrator.py      # LangGraph 顺序 rollout（legacy，parallel_games=1）
    ├── batch_rollout.py     # N 局并发批量 rollout（生成器协议 + 批量 generate）
    ├── rewards.py           # 奖励注册表：step / potential / potential_value / settlement
    ├── arena.py             # 2v2 复式对战（按座位路由 adapter，原始点数评分）
    └── task.py              # MahjongTask（collect_rollouts 入口）

scripts/
├── generate_sft_data.py     # 忠实 CoT 教师语料（--value_facts 价值感知模式）
├── analyze_defense_probe.py # 防守探针：对手立直后弃和率/放铳率
├── audit_think_accuracy.py  # think 声明忠实度审计（vs 真实向听/受入计算器）
├── run_arena.py             # 竞技场：两 adapter 复式对战 + 配对点差裁决
├── phase1_ce/               # GCP 单卡 VM 工作流（start_vm / sync_code / run_training）
└── phase2_vertex/           # Vertex AI（未启用）

tools/webui/                 # 本地训练检视台（曲线 + 雀魂式复盘 + 实时视图）
tools/majsoul_bridge/        # 雀魂实战桥接（MahjongCopilot 插件 + 安装器 + 运行手册）
src/agents/dnn/mjai_bridge.py # MJAI 协议 bot：引擎影子桌复用编码器/合法动作（scripts/serve_mjai_bot.py 提供 HTTP）
configs/                     # 训练配置 JSON（传给 trainer --config）
experiments/<name>_<ts>/     # 每次训练的输出（记录入 git，重产物不入）
```

## 奖励系统（registry 模式，`reward_model` 配置项选择）

| 名称 | 说明 |
|---|---|
| `step` | 旧版绝对分塑形（可刷分，仅为历史复现保留） |
| `potential` | **势函数塑形（PBRS）**：Φ=−2·向听+0.05·受入；折扣和 telescoping 到发牌常数——与终局结算严格一致、不可刷分、最优策略不变 |
| `potential_value` | PBRS + 0.3×宝牌持有项（引导价值探索，保证不变） |
| `settlement` | **纯目标**：零塑形，只保留格式 −10 / 幽灵牌 −5 合法性护栏（exp1 结论催生）|

终局结算 = 真实点差×0.001 + 顺位奖 ±2/±0.5（四家轨迹全部分发）；流局听牌费（场 3000）已实现。
可选 `--covariate_baseline`：按起手质量回归消除发牌运气方差（默认关）。

## 训练算法

- `--rl_algo reinforce`（默认）：advantage-weighted NLL，每批 1 遍
- `--rl_algo ppo`：clipped surrogate（ε=0.2）+ 每批 ≤3 遍复用 + approx-KL 早停；
  无 critic（GRPO 风格：全 buffer 归一化 MC return，±5 截断）；rollout 时记录行为策略
  原始 logits logprobs，按 token id 重建序列（无重分词漂移）
- 精度：`--use_qlora`（nf4，16GB 卡）或 `--bf16_lora`（A100 推荐，解码 +55%）
- `--parallel_games N`：N 局并发批量 rollout（A100 实测 24 路近线性，单局成本 5.2× 优化）

## 快速开始

**本地（RTX 4080 16GB，QLoRA）**：
```bash
conda activate rlhf_mahjong
python -m src.core.trainer --config configs/v2_ppo_smoke.json   # PPO+并发冒烟
python -m unittest discover tests                               # 47 项单测
```

**GCP Phase 1（A100 40GB，bf16）**：
```bash
VM_NAME=mahjong-a100 ZONE=us-central1-b bash scripts/phase1_ce/start_vm.sh
VM_NAME=... bash scripts/phase1_ce/sync_code.sh    # 数据同步（代码建议走 git pull）
ssh <vm> 'nohup bash LLM_Mahjong/scripts/phase1_ce/run_training.sh configs/v2_pbrs_run.json > ~/train_nohup.log 2>&1 &'
# 完赛自动：上传 gs://llm-mahjong-experiments/<exp>/ → shutdown（EXIT trap，崩溃同样生效）
```
注意：A100 配额每区域 1 块，多 VM 并发需跨区域；VM 停机不保留 GPU 容量（详见 SKILLS.md）。

**训练检视台**：
```bash
conda run -n rlhf_mahjong python tools/webui/server.py --port 8642
```
曲线（含 PPO 健康指标）、雀魂式逐步复盘（巡数/结算明细/奖励解读）、进行中对局实时视图、跨 VM 一键同步。

## 主力配置

| 配置 | 用途 |
|---|---|
| `configs/v2_pbrs_run.json` | 基线：PBRS + REINFORCE，50×12，bf16 |
| `configs/v2_ppo_run.json` | Arm A：+PPO（对基线单变量） |
| `configs/v2_ppo_value_run.json` | Arm B：PPO + 价值 bundle（potential_value + 价值事实模板 + 价值语料 adapter） |
| `configs/v2_ppo_smoke.json` | 本地冒烟（1 epoch，4 局并发） |

## 当前状态（2026-08-02）

- **exp1「塑形三臂」已完成**（基线 PBRS+REINFORCE / +PPO / PPO+价值 bundle，infra rev3，ep24-26 停跑）：
  竞技场裁决三臂均与各自 SFT 起点无显著差异，但行为风格大幅迁移 → 密集塑形主导学习方向。
  完整结论见 [docs/report_exp1_shaping_arms_20260802.md](docs/report_exp1_shaping_arms_20260802.md)。
- **exp2「settlement vs PBRS」提案中**（待批准）：纯结算奖励 vs 现行塑形，竞技场 64 副裁决。
- 命名规范：实验用描述性名字（exp1/exp2…），`infra revN` 只指训练栈配置版本。
