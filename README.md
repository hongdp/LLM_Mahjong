# LLM Mahjong — 基于 LLM 的多智能体日麻自对弈 RL 系统

用 4 个共享权重的 LLM 实例进行四人立直麻将自对弈（POMDP），通过自定义的 Advantage-Weighted RL 循环（Replay Buffer + 策略梯度）训练模型，从基础牌效率逐步进化到防守与全局判断。

详细设计见 [docs/mahjong_design_document.md](docs/mahjong_design_document.md)，GCP 三阶段部署方案见 [docs/implementation_plan.md](docs/implementation_plan.md)。项目经验与硬件约束记录在 [SKILLS.md](SKILLS.md)（每次开发前先读）。麻将引擎的已知问题与修复优先级见 [docs/engine_known_issues.md](docs/engine_known_issues.md)（**跑正式 RL 实验前先看 P0 部分**）。

## 架构总览

```
src/
├── core/                  # 通用 RL 引擎（与游戏解耦）
│   ├── trainer.py         # 主入口：SFT warm-up + RL 循环（NLL × Advantage）
│   ├── rollout.py         # ReplayBuffer / TrajectoryStep / Return-to-Go 计算
│   ├── task.py            # Task 抽象基类
│   ├── registry.py        # 任务注册表
│   └── base_reward.py     # 奖励模型基类
└── tasks/mahjong/         # 麻将领域逻辑
    ├── table.py           # 136 张牌桌引擎（发牌/合法动作/吃碰杠和）
    ├── shanten.py         # 向听数 / 受入计算
    ├── orchestrator.py    # LangGraph 回合图（turn 节点 + interrupt 节点）
    ├── rewards.py         # 奖励塑形（格式/向听/受入/和牌）
    └── task.py            # MahjongTask（collect_rollouts 入口）

scripts/
├── generate_sft_data.py   # 生成 SFT 冷启动数据（最优受入轨迹 → ChatML JSONL）
├── phase0_local/          # 本地小规模验证（run_local_test.sh）
├── phase1_ce/             # GCP Compute Engine（start_vm / sync_code / run_training）
└── phase2_vertex/         # Vertex AI 容器化训练（build_docker / submit_job）

configs/                   # 训练配置（JSON，传给 trainer --config）
data/sft_mahjong.jsonl     # SFT warm-up 数据
experiments/<name>_<ts>/   # 每次训练的输出目录（config、日志、checkpoint、tensorboard）— 不入 git
```

## 快速开始（本地 Phase 0）

```bash
conda activate rlhf_mahjong
bash scripts/phase0_local/run_local_test.sh                       # 冒烟测试（1 epoch）
bash scripts/phase0_local/run_local_test.sh configs/full_run.json # SFT + RL 完整流程
```

监控训练：

```bash
tensorboard --logdir experiments/<exp_name>/tensorboard
```

关键指标：`sft/epoch_loss`、`rl/loss`、`rl/avg_episode_reward`、`rl/format_compliance`。
每轮 rollout 的完整 prompt/输出会写入 `experiments/<exp_name>/live_rollout.txt`，用于人工检查模型行为。

## 配置说明

| 配置 | 用途 |
|---|---|
| `configs/experiment_1.json` | 最小冒烟测试（1 epoch, 1 episode） |
| `configs/full_run.json` | 完整流程：3 epoch SFT warm-up + 100 epoch RL |
| `configs/baseline.json` | 跳过 SFT（`sft_epochs: 0`），通过 `peft_model_path` 加载已有 SFT adapter 直接进 RL |

重要参数：`training_phase` 1=仅格式奖励，2=完整策略奖励；`num_episodes` 每个 epoch 的自对弈局数；`peft_model_path` 复用已训练的 LoRA adapter。

## 当前状态（2026-08）

- **Phase 0 已完成**：完整 136 张牌引擎、吃/碰/杠/和逻辑、LangGraph 中断路由、SFT warm-up + RL 双阶段训练管线均可跑通（本地 Qwen2.5-0.5B + QLoRA）。
- **已知阻塞问题**：最近一次 baseline RL run 中，rollout 阶段约 91% 的输出没有产生合法 `<action>` 标签（模型只输出牌面列表），被 fallback 解析为 `skip`——RL 实际在垃圾数据上训练。原因很可能是 `baseline.json` 加载的是 `config_test_run` 的 adapter（只训了 1 个 SFT epoch、500 条样本），格式能力不足。下一步应改为加载 `full_run` 的 3-epoch SFT checkpoint 或提高 SFT 强度后再验证。
- 本地 16GB 显存无法跑 Gemma 系列（peft fp32 embedding 上浮），Phase 1 需上 GCP。
