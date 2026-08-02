# 实验总账

> 记录规范见 `~/Workspace/SKILLS/ml-experiment-tracking/SKILL.md`。2026-05 的三次实验为事后补录（当时无档案，信息从日志反推）。

| 日期 | 实验目录 | 一句话目的 | 关键结果 | 结论 |
|---|---|---|---|---|
| 2026-08-01 | `v2_engine_pbrs_run_20260802_054918` | 基线（rev3）：PBRS+REINFORCE，bf16，50×12，central | ⏹ ep26 用户停跑转竞技场：格式100%全程,和牌~62%,风格迁移(立直6.6→3.1,副露49→86) | 竞技场对战出最终裁决 |
| 2026-08-01 | `v2_engine_ppo_run_20260802_054914` | Arm A（rev3）：PPO 消融，同基线仅换算法，east | ⏹ ep24 停跑转竞技场：格式≥99.6%,和牌~62%,罚分自愈现象 | 竞技场裁决 |
| 2026-08-01 | `v2_engine_ppo_value_run_20260802_054921` | Arm B（rev3）：PPO+价值 bundle，复用 041051 价值 adapter，europe | ⏹ ep25 停跑：副露流门风,格式侵蚀至97.5%(ref-KL依据),和牌~70% | 竞技场裁决 |
| 2026-08-01 | `v2_engine_pbrs_run_20260802_041048` | PBRS+REINFORCE 基线（022840 的 infra rev2 重启，设计不变，central A100） | ⏹ epoch 1 内暂停调优，无产出；设计迁移 rev3 重启 | 见 perf_tuning_east_20260802 |
| 2026-08-01 | `v2_engine_ppo_run_20260802_041049` | Arm A：PPO 消融——对 041048 仅换更新算法（同 adapter/奖励/seed，east A100） | ⏹ epoch 1 内暂停调优，无产出；设计迁移 rev3 重启 | 见 perf_tuning_east_20260802 |
| 2026-08-01 | `v2_engine_ppo_value_run_20260802_041051` | Arm B：PPO+价值 bundle（Φ宝牌项 + prompt 价值事实 + 价值教师新 SFT，europe A100） | ⏹ SFT 完成(loss 0.0877, adapter 存续复用)，RL epoch 1 内暂停 | 设计迁移 rev3 重启 |
| 2026-08-01 | `v2_engine_pbrs_run_20260802_022840` | PBRS 奖励 RL run（复用 005918 SFT adapter） | ⏹ epoch 1 中止（无产出）：诊断出 rollout host-launch-bound（A100 11 tok/s，GPU 18%），预计 50h/$180 | 设计原样迁移到 infra rev2 重启（fast-path 内核 + 4 局并发批量 rollout）；教训见 SKILLS.md 性能诊断节 |
| 2026-08-01 | `v2_engine_full_run_20260802_005918` | 原实验设计迁移 GCP A100；实际只完成 SFT 阶段 | SFT 3×2000 完成，final epoch loss ≈0.105，adapter 产出 | ⏹ RL 开跑前用户叫停：step 奖励可刷分且与结算不自洽，RL 阶段改用 PBRS 奖励另起新实验（复用本次 SFT adapter，单变量对照） |
| 2026-08-01 | `v2_engine_full_run_20260801_165312` | v2 引擎首次完整训练：忠实 CoT SFT (3ep×2000) + 50 epoch RL，Qwen3.5-2B | ⏸ 用户在 SFT 阶段主动暂停（无 checkpoint） | 实验设计迁移至 GCP VM 继续（见 docs/handoff_gcp_phase1.md），成功标准原样继承 |
| 2026-08-01 | `v2_smoke_20260801_164239` | 冒烟：模板对齐修复后验证 SFT+RL 管线 | 格式合规 100% (86/86)，SFT loss 0.46，无 OOM | ✅ 管线就绪；Qwen3.5 需 enable_thinking 模板对齐 |
| 2026-08-01 | `v2_smoke_20260801_162930` | 冒烟：Qwen3.5-2B + 新引擎 + 忠实 CoT 数据首测 | 格式合规仅 69.5%，失败全为裸推理文本 | ❌ 暴露 chat template think 块错位 → 促成 chat_format.py 统一渲染 |
| 2026-08-01 | `format_check_20260801_150549` | 验证引擎修复（和牌校验/回合路由/空手牌保护）后完整对局 + RL epoch 可跑通 | 格式合规率 99.5% (186/187)，2 局完整对局 + 1 RL epoch 无崩溃 | ✅ full_run adapter + 引擎修复有效，可进入正式 RL；但 P0 奖励漏洞未修（见 docs/engine_known_issues.md） |
| 2026-08-01 | `format_check_20260801_145812` | 验证「加载 full_run 3-epoch SFT adapter 可恢复格式合规率」假设 | 100% 输出带 action 标签、93% 类型合法（5 月 baseline 仅 ~9%）；但引擎在空手牌处崩溃 (`table.py` random.choice) | ✅ 假设验证：5 月失败根因是加载了弱 adapter；同时暴露引擎回合路由 bug → 触发引擎修复 |
| 2026-05-17 | `full_run_20260517_035217`（补录） | 完整流程：3 epoch SFT warm-up + RL | SFT 最终 loss 0.0725；产出 `checkpoints_sft_warmup_mahjong`（后被验证为格式合格的 adapter） | SFT 阶段成功；RL 部分未见完整记录 |
| 2026-05-17 | `baseline_local_run_20260517_034738`（补录） | 加载已有 SFT adapter 跳过 warm-up 直接 RL | rollout 中 ~91% 输出无 action 标签（122/134 fallback 为 skip），RL 在垃圾数据上训练 | ❌ 失败：`peft_model_path` 误指向 config_test_run 的 1-epoch 弱 adapter；2026-08 已修正配置 |
| 2026-05-17 | `config_test_run`（补录，无时间戳） | 配置系统冒烟测试（1 epoch，1 episode，1 SFT epoch） | 管线可跑通；产出的 1-epoch SFT adapter 格式能力不足 | 冒烟通过；其 adapter 不可用于正式 RL（教训：冒烟产物勿当训练资产复用） |
