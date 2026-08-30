# CLAUDE.md

在这个仓库工作前，必须先读 [SKILLS.md](SKILLS.md)（教训/方法论/硬件约束）和 [README.md](README.md)（架构与运行方式）；当前实验状态看 [experiments/INDEX.md](experiments/INDEX.md) 与 [experiments/FINDINGS.md](experiments/FINDINGS.md)。

## 文档分区（什么写到哪里，2026-08-30 定）
| 内容 | 位置 |
|---|---|
| 规则（本文件）：唯一权威，别处不重复 | `CLAUDE.md` |
| 教训 / 方法论 / 运维事故 / 硬件细节（带日期追加） | `SKILLS.md` |
| 实验一行总账（每个 run 一行，云上 run 也要记） | `experiments/INDEX.md` |
| 实验结果综合台账 | `experiments/FINDINGS.md` |
| 系列级预注册 + 进度 + 判决（云 run 的本地记录本体） | `experiments/<系列>_prereg/EXPERIMENT.md` |
| 本地训练 run（EXPERIMENT.md + config 快照 + 工件） | `experiments/<name>_<时间戳>/` |
| 云 run 工件（ckpt/日志/TB） | GCS `gs://llm-mahjong-experiments/<run>/`；本地只镜像 TB 到 `experiments/_cloud_mirror/` |
| 实验设计文档 | `experiments/designs/` |
| 历史发射配置（无代码引用的 run config） | `experiments/configs/` |
| 实验报告（成文版） | `experiments/reports/` |
| 长期参考（引擎已知问题、总设计书、GCP 成本、路线图） | `docs/` |
| 论文素材（不入库） | `paper/`（.git/info/exclude 屏蔽） |

## 关键规则
- 允许在逻辑里程碑处自动 `git commit`（单一主题、信息清晰）；`git push` 与历史改写须用户确认（已白名单的分支除外）。
- **任何训练 / 实验 / 评估 run 启动前，必须先走 `ml-experiment-tracking` skill**：EXPERIMENT.md（目的/方法/成功标准）先行，进度随记，收尾补结果与 artifact 清单，并更新 `experiments/INDEX.md`。云上 run 同样记账（prereg 文件 + INDEX 行 + GCS 路径）。
- 每次新训练 run 新建带时间戳目录（除非 `--resume`）；云 run 用新的 GCS 命名空间。
- **每发射一个新实验，立即完成 TensorBoard 三件套**（不得等提醒）：①`experiments/_cloud_mirror/<run>/tensorboard` 建目录+首次 rsync；②心跳循环内挂周期 rsync；③重启 tensorboard 把 `<run>_LIVE` 加进 `--logdir_spec`（按 PID 隔离单命令 kill，等端口释放再起）。
- **心跳分相报警**：发射相 15 分钟死线（GCS 无日志对象即 ALERT）、训练相 30 分钟 STALL、终止标记即时报；触发标记必须取被监控日志实际会出现的行。
- 本地 GPU（RTX 4080 16GB）只做冒烟/debug/benchmark；小时级训练一律上云。LLM 验证仅 Qwen2.5-0.5B + QLoRA，Gemma 会 OOM。
- 评测协议：终审一律 T=0 + 族外梯子 + 半庄 n≥300；T=1 族内曲线不得单独下结论。
- 奖励逻辑必须模块化（registry + BaseRewardModel），不得硬编码进训练循环。
- kill 命令必须单独成调用且不含目标字符串明文（pkill 自匹配史）；生成代码后必须**独立**跑一次 ast/语法校验。
- 有新教训/架构决策 → 追加 SKILLS.md（带日期）；有新结果 → FINDINGS.md / 对应 EXPERIMENT.md，不进 SKILLS。
