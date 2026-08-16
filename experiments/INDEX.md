# 实验总账

> 记录规范见 `~/Workspace/SKILLS/ml-experiment-tracking/SKILL.md`。2026-05 的三次实验为事后补录（当时无档案，信息从日志反推）。

| 日期 | 实验目录 | 一句话目的 | 关键结果 | 结论 |
|---|---|---|---|---|
| 2026-08-01 | `v2_engine_pbrs_run_20260802_054918` | exp1 基线：PBRS+REINFORCE，bf16，50×12，central | ⏹ ep26 用户停跑转竞技场：格式100%全程,和牌~62%,风格迁移(立直6.6→3.1,副露49→86) | 竞技场:+1038±1876 不显著;总报告 docs/report_exp1_shaping_arms_20260802.md |
| 2026-08-01 | `v2_engine_ppo_run_20260802_054914` | exp1 Arm A：PPO 消融，同基线仅换算法，east | ⏹ ep24 停跑 | 竞技场:+331±1320 不显著;打点+18% |
| 2026-08-01 | `v2_engine_ppo_value_run_20260802_054921` | exp1 Arm B：PPO+价值 bundle，复用 041051 价值 adapter，europe | ⏹ ep25 停跑 | 竞技场:−1475±1652 不显著;打点4424 但漏点;详见总报告 |
| 2026-08-01 | `v2_engine_pbrs_run_20260802_041048` | PBRS+REINFORCE 基线（022840 的 infra rev2 重启，设计不变，central A100） | ⏹ epoch 1 内暂停调优，无产出；设计迁移 infra rev3 重启 | 见 perf_tuning_east_20260802 |
| 2026-08-01 | `v2_engine_ppo_run_20260802_041049` | Arm A：PPO 消融——对 041048 仅换更新算法（同 adapter/奖励/seed，east A100） | ⏹ epoch 1 内暂停调优，无产出；设计迁移 infra rev3 重启 | 见 perf_tuning_east_20260802 |
| 2026-08-01 | `v2_engine_ppo_value_run_20260802_041051` | Arm B：PPO+价值 bundle（Φ宝牌项 + prompt 价值事实 + 价值教师新 SFT，europe A100） | ⏹ SFT 完成(loss 0.0877, adapter 存续复用)，RL epoch 1 内暂停 | 设计迁移 infra rev3 重启 |
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
| 2026-08-06 | `exp2_smoke_20260806_232343` | exp2 发射前基建冒烟：settlement reward + PPO + ref-KL 锚 + γ0.995 端到端 | ref_kl 0.0016 / ppo_passes 3 / format 100% / 退出码 0 | ✅ 通过；4080 需 batch 1（bf16 PPO 更新 OOM 边界）；产物不复用 |
| 2026-08-07 | `exp2_settlement_20260807_070806` | exp2 双臂主臂：纯结算奖励 + PPO + ref-KL 0.05 + γ0.995，50×12，flex-start 临时机 | 50/50 clean；竞技场 S vs P 无差异（−183±1591） | ✅ 完成；H3 证伪：稀疏信号下副露 6.02/局 极端迁移 |
| 2026-08-07 | `exp2_pbrs_20260807_065729` | exp2 对照臂：PBRS 奖励，其余与主臂全同（单变量） | 50/50 clean；vs 锚 +511±1774 无差异；ref_kl 5× S 臂 | ✅ 完成；PBRS 奖励曲线与强度脱钩再确认 |
| 2026-08-08 | `exp2_arena_20260808_1050` | exp2 主判据执行：S vs P / S vs 锚 / P vs 锚，各 64 副复式（seed0 同 exp1） | 三场全 null（CI ±1500-1800） | ✅ 完成；报告 docs/report_exp2_settlement_vs_pbrs_20260809.md |
| 2026-08-09 | `exp3_nothink`（预注册总纲 exp3_nothink_PREREG.md） | think 消融：配对语料（教师动作 100% 一致）双 SFT + 同 seed 48 局速度/质量对比 | pre-registered | 判据：≥2× 提速且质量差 <10pp → no-think 成 v3 默认 |
| 2026-08-09 | `exp3_sft_think_20260809_072304` / `exp3_sft_nothink_20260809_073440` | 配对语料双 SFT（post-audit 引擎，11390×2，教师动作一致） | loss 0.080 / 0.070，锚点各产出 | ✅ exp3 阶段 1-2 |
| 2026-08-09 | `exp3_probe_think_20260809_074227` / `exp3_probe_nothink_20260809_084310` | 同 seed 48 局 ×24 并发速度/质量探针 | 60.7 vs 63.2min；质量完全一致 | ✅ exp3 判定：速度不达标 → 保留 think；风格 5× 敏感度副产品 |
| 2026-08-09 | `exp4_critic_20260809_103720` | critic value head vs exp2-S 对照（单变量 use_critic；checkpoint_every 10 阶梯赛） | running | 判据：explained_var>0.2 门槛 / 方差降幅 / 竞技场 |
| 2026-08-10 | `exp4_critic_20260810_062927` | exp4 重发（run-1 全损事故后，加固 infra） | critic 机制判据双双不达标（EV 0.020 / 方差降 1%）；**阶梯赛 ep40 vs 锚 +2950, p=0.0027 校正后显著** | ✅ 完成；报告 docs/report_exp4_critic_20260811.md |
| 2026-08-11 | `exp4_arena_20260811` | exp4 竞技场 6 场：主判据 + ep10/20/30/40/50 阶梯轨迹 | 主判据 null；ep40 首个显著增益 | ✅ 完成；阶梯赛制确立为标准收尾 |
| 2026-08-12 | `exp5_confirm_20260811` | ep40 峰值独立复现（96 副新种子）+ ep35/45/exp2S-ep25 探索 | 主判据未确认（+1093, p=0.076）；**汇总 7 点得 +1076±753, p=0.005，Q=3.58 无异质性** | ✅ 完成；正确图景=平坦 ~+1000 增益；报告 docs/report_exp5_confirm_20260812.md |
| 2026-08-15 | `dnn_scratch_massive_20260815` / `dnn_ppo_massive_20260815` | 纯自对弈（无教师无塑形）标度曲线 + REINFORCE vs PPO | 和牌率 0.53%→45.8%；PPO-240k vs RF-240k **+1785±473**；标度 +3533 vs +1845 | ✅ 报告 docs/report_exp8_dnn_scaling_20260815.md |
| 2026-08-15 | `exp8_arena_final_*`（600k 终点） | 两臂跑满 60 万局的最终对决与标度终点 | **反转**：240k 时 PPO +1785，600k 时 REINFORCE +749（p=0.0016）；PPO 240k→600k 倒退 −809（p=0.0058） | ✅ 报告已按全程数据改写 |
| 2026-08-15 | `arch_sweep`（exp10） | 架构 zoo BC 扫描：CNN×3 / ViT×3 / 顺序编码×2 | vit_small 89.3%（0.82M）vs 现役 81.4%（1.94M）；**决胜竞技场 +270±555 无显著差异** | BC 保真度不兑现强度；RL 毕业赛排队 |
| 2026-08-16 | `dnn_ppo_reuse1_20260815`（exp9） | PPO 过度复用消融：44→11 步/批，从 240k 峰值续训至 600k | **新冠军**：vs PPO44 +1802、vs REINFORCE +1608（均 p<1e-5）；无倒退 | ✅ 假说确认；11 步成默认配方 |
| 2026-08-16 | `exp12_plateau_prereg`（E/B/L 臂目录 `dnn_exp12_*_20260816`） | PPO 平台归因：熵/批量/lr 三臂单变量消融（各 +100k 局自 exp9-600k） | **E（熵 0.03→0.01）+1133±1020 发现 + 独立确认 +1042±730，合并 +1073±594**；B（4× 批量）null 点估计 −841；L（lr↓）null +588 | ✅ 平台=熵奖励均衡；**新冠军 E-700k**；配方=末段熵退火；B 反证噪声地板假说 |
| 2026-08-16 | `exp13_entropy_prereg`（臂目录 `dnn_exp13_{H,S,A}_20260816`） | 熵退火是梯子还是一次性红利：hold/手工阶梯/拉格朗日自动三臂，自 E-700k 续 200k | **四场竞技场全 null**（S-H +166±1090；A-H −831±1133；A-S −340±1167；S vs E-700k +584±1080） | ✅ 梯子假说被否：熵系数价值曲线是台阶形；冠军仍 E-700k；配方定稿=饱和后一步降至 0.01；自动控制器机械完美、留待 scratch run 用 |
