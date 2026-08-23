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
| 2026-08-16 | `exp10_arch_prereg` 毕业赛（run `dnn_vit_rl_20260815r4`，结果 `exp10_arena_vit240*.json`） | vit_small 用 exp9 配方从零 RL 240k，与 cnn-PPO44-240k 同尺竞技场 | **+2205±959（200 副）/ +1633±678（400 副确认，p≈2e-6）**；vit-80k<vit-240k 曲线自检通过 | ✅ vit 毕业：结构先验 BC 不兑现但 RL 兑现，迄今最大单点效应；下一步 vit 续训 700k 挑战 E-700k |
| 2026-08-16 | `dnn_exp11_a1_20260815` / A0=`dnn_exp11_a0_20260815r4` | exp11 A1：价值-距离剖面特权 critic vs A0 普通 critic，各 600k 从零单变量 | 主判定 **+494±1008 null**；explained_var 全程与 A0 重合 | ✅ 双 null：剖面注入无增量（critic 自学得到/用不上）；A2 hazard 头为更强检验，已在 c3 发射 |
| 2026-08-16 | `elo_league_calib_20260816_144910` | Elo 锚点池校准：7 冻结锚点全循环 21×200 副，双标度联合拟合（设计 docs/design_elo_league.md） | 排序与全部既有显著结论零冲突；**bc_cnn 竟居全池第 2，仅冠军 E-700k 显著在上（+1072±250）** | ✅ 标尺冻结生效；教师先验价值被低估 → 提议 exp14 BC-warm-start+RL；轻度非传递性 ~800 分已记录 |
| 2026-08-16 | `dnn_exp11_a2_20260815` | exp11 A2：hazard 分解 critic（役种完成率 × 价值 + BCE 监督）vs A0 单变量 | 主判定 **−2029±797 显著为负**（92:203）；BCE 收敛但 EV 不升；风格反向（副露 0.95、立直 0.02） | ✅ exp11 收官：外部役种知识注入 critic 两式全败；「良性乐观闭环」风险如实发生；**副产品：A0 纯自对弈+GAE 自发立直 14.9%（旧谱系 3%）→ 提议 exp17 GAE 单变量** |
| 2026-08-16 | `dnn_exp14_bcvit_rl_20260816`（参照线，非冠军谱系） | BC(vit) warm-start + 冠军配方 600k：教师先验价值定量（测量仪定位） | **240k 处 +3482±1042（史上最大效应）**；600k vs 起点 +989±889 显著正；先验存活（立直 26%/副露 40%，人类区间）；**Elo 1117.7±15.3 全史最高** | ✅ 三判据全过；266k 后平台 ~1100（先验+RL 该配方 ~300k 见顶）；已冻结为池顶锚点 `bcrl14_600`，不入冠军谱系 |
| 2026-08-16 | `probe_dup_diversity_20260816` | 探针：同山重放线多样性是否随策略变尖枯死 + dup_k 边际 | e700 的 24 重放 99.6% 条条不同；组内回报 std 晚期反而更大；adv 相关度 K=8 已 0.94 | ✅ 「多样性枯死」假设证伪（诚实记录）；**dup_k=8 近最优，无需测更大**；exp16 岔线重放降级候选 |
| 2026-08-16 | `dnn_exp15_vitscale_20260816`（纯自对弈主线） | vit 谱系冠军配方自 240k 续至 1.2M：样本量假说 + 挑战 e700 | 冠军战 **−1475±1103 败于 e700**；vs vit240 +1246±1209（增益全在 240k→350k，此后 850k 局平台）；终评 943.6±12.7，距参照线缺口 174 分未收敛 | ✅ 样本量假说否定分支触发：平台是机制约束非样本饥饿；vit 天花板反低于 cnn；**exp17（GAE 归因，A0=1022 线索）升主线头号候选** |
| 2026-08-16 | `dnn_exp16_advclamp_20260816` | 优势裁剪界消融：±5σ cap 掉役满 ~8σ 梯度是否有害（--adv_clamp 10，唯一差异 vs exp15） | 主判定 **−1064±1015 显著负**（±10 更弱）；满贯+ 47 vs 51 无右移；终评 923.2±12.7 | ✅ **大番 reward-cap 问题结案**：±5 winsorize 是有益方差控制，保留默认；连同 A2 两条「解禁大额信号」路线全败——瓶颈不在价值幅度端 |
| 2026-08-17 | `exp18_archpair_prereg`（臂 `dnn_exp18_{cnn,vit}_20260817`） | 用户设计：从零全同配置 cnn vs vit 配对对照（唯一变量=arch），清算 exp15 混杂与 exp10 旧账 | **700k：cnn 胜 −2000±1151（197:97）**；240k +471 null（exp10 的 +1633 未复现）；cnn 终评 **1013.4** 与 e700 逐分重合，vit 928.3 | ✅ cnn 渐近线高 ~85 分为真；exp10 结论推翻（配方依赖幻象）；配方渐近线 ≈1012 可复现；**主线 backbone 回归 cnn**，唯一在案爬升假说=exp17-C（GAE） |
| 2026-08-18 | `dnn_exp17c_gae_20260818` | GAE 单变量归因（exp18-cnn 共享基线 + `--gae_lambda 0.95`），验证 A0=1022 线索 | 点差 +244±984 null 但评分 **1079.7±13.5 纯自对弈新纪录**（基线 1013.4，z≈3.5）；立直率 0.243 vs 基线 0.117；**700k 仍在爬升无平台** | ✅ GAE 归因确认（评分+风格+副胜三线汇聚）；**新纯自对弈冠军**；与教师参照线缺口 174→38 分；冠军配方候选=GAE+熵台阶 |
| 2026-08-18 | `dnn_exp19_convformer_20260818`（r2） | ConvFormer 生死战：三病根修复的注意力 vs exp18-cnn 共享基线（r1 因 warmup 尺寸失误 10% 处重发） | 评分 **1065.7±13.3**（vs 旧 vit 928.3 **+137**；vs cnn 基线 +52 z≈2.8）；点差 +490±871 null；**700k 仍在爬升**；80k 爬速判据败于 warmup 代价 | ✅ 注意力路线复活（慢热高顶型）；BC 90.5% 新纪录；GAE 与 ConvFormer 正交地各自打破 1012 平台 → exp20 合体候选 |
| 2026-08-19 | `dnn_exp20_gaeformer_20260819` | GAE × ConvFormer 合体 1.2M：700k 可加性判定 + 1.2M 挑战教师参照线 1117.7（AlphaZero 里程碑战） | 发射确认，running | 🔵 700k 里程碑评分 ≥1110=可加；1.2M 胜 bcrl14_600=里程碑达成 |
| 2026-08-19 | `exp21_defense_probe_20260819` | 防守测量套件（defense_iq=条件现物率差）+ 五谱系回填，各 800 局引擎直采 | **全谱系 defense_iq≈0（−0.04..+0.01）**，曝露放铳率 19-24%；H1 排序方向成立但幅度为零 | ✅ 防守未在任何谱系涌现；威胁供给必要非充分 ⇒ **exp22 对手联赛/场景课程升级为必要**；防守从此可测（本探针为其成功判据） |
