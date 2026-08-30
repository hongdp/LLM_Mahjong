# CLAUDE.md

在这个仓库工作前，必须先读 [SKILLS.md](SKILLS.md)（项目经验、硬件约束、当前状态快照）和 [README.md](README.md)（架构与运行方式）。

关键规则（详见 SKILLS.md）：
- 允许在逻辑里程碑处（功能/修复完成且相关测试通过、实验记录更新）自动执行 `git commit`；commit 需单一主题、信息清晰。`git push` 与任何历史改写仍须用户明确确认。
- **任何训练 / 实验 / 评估 run 启动前，必须先调用 `ml-experiment-tracking` skill 并遵循其流程**：先写好 `EXPERIMENT.md`（目的 / 方法 / 成功标准）才允许启动，运行中记进度，结束后补结果、结论、artifact 清单，并更新 `experiments/INDEX.md` 总账。没有 EXPERIMENT.md 就启动训练视为违规。
- 每次新训练 run 必须创建新的带时间戳的 `experiments/` 目录（除非 `--resume`）。
- **每发射一个新实验，立即完成 TensorBoard 三件套**（用户多次强调，不得等提醒）：①`experiments/_cloud_mirror/<run>/tensorboard` 建目录+首次 rsync；②心跳循环内挂周期 rsync；③重启 tensorboard 把 `<run>_LIVE` 面板加进 `--logdir_spec`（按 PID 隔离单命令 kill，等端口释放再起）。
- 本地 GPU 为 16GB（RTX 4080）：只能用 Qwen2.5-0.5B + QLoRA 做验证，Gemma 系列会 OOM。
- 奖励逻辑必须模块化（registry + BaseRewardModel），不得硬编码进训练循环。
- 有新的经验教训 / 架构决策时，追加到 SKILLS.md。
