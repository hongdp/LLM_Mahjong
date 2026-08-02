# Phase 1 交接：在 GCP VM 上继续 v2 引擎实验

> 写于 2026-08-01。本地实验 `v2_engine_full_run_20260801_165312` 在 SFT 阶段被主动暂停（无 checkpoint 产出），实验设计原样迁移到 GCP Compute Engine 继续。新会话开工前先读：`CLAUDE.md` → `SKILLS.md` → 本文件 → `experiments/v2_engine_full_run_20260801_165312/EXPERIMENT.md`（被中止实验的完整设计与预注册成功标准）。

## 当前项目状态（截至 f132ba1 + 后续小提交）

- **引擎 v2 已完成并验证**：真实算分（役/番/符/宝牌）、完整立直规则、振听、副露手可和牌、中断优先级裁决、终局结算分发到全部四条轨迹、顺位奖励（+2/+0.5/−0.5/−2）、放铳无重复计分（HOUJUU_EXTRA=0，纯点棒经济）。16 个单测 + 50 局随机压测通过。问题清单见 `docs/engine_known_issues.md`（P0–P2 全清，P3 剩半庄结构/一发里宝/同巡振听为有意延后）。
- **训练管线关键修复**（详见 SKILLS.md "Engine v2 & Training Lessons"）：
  - `src/core/chat_format.py` 是全链路唯一的模板渲染源（Qwen3/3.5 需 `enable_thinking=True` 对齐，SFT 与 rollout 格式必须一致——冒烟证明错位会把格式合规率打到 69.5%，对齐后 100%）。
  - rollout 采样 temperature 0.9 / top_p 0.95（贪心解码会让 RL 零探索）。
  - 切牌排序必须"向听优先、受入次之"（`evaluate_discards_ranked`）。
  - Qwen3.5 系需显式 LoRA `target_modules`；248k 词表下 16GB 卡 batch 上限 2（更大显存的 VM 可以放宽）。
- **SFT 数据**：`data/sft_mahjong.jsonl`，16601 样本、300 局、忠实 CoT（真实向听/受入对比/听牌/役种写进 think）、行级 shuffle、seed 42。sha256 = `b3eefd6d144e662b6ed4239cfbdb62197a2c4a941264ae360ab5a250615becf6`。**同步该文件后必须校验哈希**（或用相同代码+seed 重生成后核对）。
- **实验设计**（复用 `configs/v2_full_run.json`）：Qwen3.5-2B QLoRA，SFT 3 epoch × 2000 样本 → RL 50 epoch × 4 局自对弈，seed 42，min_format_rate 0.3 熔断，checkpoint 按 top-3 奖励 + 最新保留。四条预注册成功标准在被中止实验的 EXPERIMENT.md 里，原样继承。

## GCP 上要做的事（Phase 1 路线图，implementation_plan.md 的方案）

1. **配置 `scripts/phase1_ce/` 三个脚本**——目前全是占位符（`your-gcp-project-id` 等），需要真实的 PROJECT_ID / ZONE / VM_NAME。先问用户：GCP 项目号、偏好区域、GPU 配额情况。
2. **建 VM**：Deep Learning VM 镜像。GPU 选型建议给用户两档：L4 24GB（便宜，跑 2B batch 4-8 绰绰有余）或 A100 40GB（快 3-5 倍，且可上 Qwen3.5-4B/9B 或大幅加 num_episodes）。**如果换模型或改超参，就是新实验，需要新的 EXPERIMENT.md 并让用户拍板**；原样复现 2B 配置则继承现有设计。
3. **环境**：同步代码（git clone `hongdp/LLM_Mahjong` 或 rsync 工作区）、conda env（requirements.txt）、`.env` 里的 HF_TOKEN、rsync `data/sft_mahjong.jsonl` 并校验 sha256。
4. **启动训练**：走 ml-experiment-tracking 流程——新建带时间戳的实验目录 + EXPERIMENT.md（标注继承自 `v2_engine_full_run_20260801_165312` 的设计与成功标准、记录 VM 机型/镜像/job 信息）→ `run_training.sh` 内 tmux/nohup 启动 → 训练完自动 `sudo shutdown -h now` 止损。
5. **监控**：tail train.log 的 Format compliance / epoch 行；TensorBoard 可 ssh 隧道。远端任务用较长轮询间隔。
6. **跑完后**：拉回 checkpoint 与日志，补全 EXPERIMENT.md 的 Results/Conclusion，更新 `experiments/INDEX.md`，做防守探针分析（对手立直后弃和率/放铳率，数据源 `mahjong_epoch_N_rollouts.txt`）。

## 排队中的下一轮迭代（本轮实验之后）

- 教师防守词汇 + 场景课程（对手已立直的初始局面），加速防守行为萌芽。
- 专家迭代：从本轮高分对局挖轨迹 → 下一轮 SFT 语料（打破教师天花板的机制）。
- 硬约束提醒：项目规则禁止未经用户确认的 `git commit`；每次训练必须先写 EXPERIMENT.md 才能启动。
