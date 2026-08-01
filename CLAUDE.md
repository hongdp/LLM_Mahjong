# CLAUDE.md

在这个仓库工作前，必须先读 [SKILLS.md](SKILLS.md)（项目经验、硬件约束、当前状态快照）和 [README.md](README.md)（架构与运行方式）。

关键规则（详见 SKILLS.md）：
- 未经用户明确确认，禁止执行 `git commit`。
- 每次新训练 run 必须创建新的带时间戳的 `experiments/` 目录（除非 `--resume`）。
- 本地 GPU 为 16GB（RTX 4080）：只能用 Qwen2.5-0.5B + QLoRA 做验证，Gemma 系列会 OOM。
- 奖励逻辑必须模块化（registry + BaseRewardModel），不得硬编码进训练循环。
- 有新的经验教训 / 架构决策时，追加到 SKILLS.md。
