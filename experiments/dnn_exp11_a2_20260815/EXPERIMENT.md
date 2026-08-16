# exp11-A2：hazard-critic（共享完成-危险头，身份无关役种泛化）

- **Date**: 预注册 2026-08-15；发射待 mahjong-dnn-a0 跑完 A0 基线（看守进程自动接棒）  **Status**: pre-registered
- **Git**: eb91d8f（实现 commit；设计文档 docs/design_hazard_critic.md）
- **Env**: GCP g2-standard-32 spot（32 vCPU + L4），DLVM cu129，torch 2.13.0+cu129，mahjong==2.0.0
- **对照**: A0 = `dnn_exp11_a0_20260815r2`；A1 = `dnn_exp11_a1_20260815`

## Purpose & Hypothesis
完整设计（docs/design_hazard_critic.md）：V(s) = Σ_y P_y(s)·value_y + 残差。9 个役种族各表示为
身份无关动力学行 [d, u, closed_ok, value/32000, turns/18]，全部过**同一个** hazard 头 → 完成率曲面
从常见族迁移到国士/四暗刻（同 32000 价值、不同 (d,u) 形状）。P_y 走 BCE 监督通道（完成与否是
既成事实，不受 on-policy 复用限制）；value 走乘法通道（役满 32000 不经优势裁剪）。
假设：①hazard 头对常见族校准良好且能区分国士/四暗刻的 (d,u) 剖面；②γV(s′)−V(s) 把「鸣牌杀死
门清族」折算成当步负信号 → 竞技场强度 ≥ A0/A1。

## Method
- 唯一改动 vs A0：`--critic_feats hazard --hazard_coef 0.5`（45 维特征 rollout 传输 + BCE 通道 +
  V 分解头；策略纯净性由 `test_a2_logits_independent_of_cfeats` 锁定）。
- 完成标签：`completion_labels(result_summary)` 解析赢家役种段（双响分段、放铳不误配，测试锁定）。
- 从零训练，其余全同 A0/A1（受控三臂）。
- 已知边界（诚实声明，见设计文档）：通用族 u 是常数占位 8.0；closed 代理把暗杠算开门。

## Config
A0 全套 + `--critic_feats hazard --hazard_coef 0.5`（diff 仅此两项；快照 config.json 由 GCS 同步）

## Success Criteria（发射前定死）
1. **主判定**：600k 终点 vs A0-600k 的 200-deal 复式配对竞技场，超出 95% CI 才算非 null。
2. **机制指标**：hazard_bce 单调下降并收敛；按族校准（预测 P vs 实际完成频率）：常见族
   （riichi_menzen/tanyao/yakuhai）reliability 斜率 0.5–2.0；国士/四暗刻的平均 P 不得高于
   其实际基率 100×（防病态乐观）。
3. explained_var ≥ A0 同期；训练健康同 A1 标准。
4. **役种泛化探针**（结论性，跑完后做）：600k 模型对「国士 2 向听起手」构造局面的 V 必须显著高于
   把同局面改为已鸣牌版本的 V（≥1 个满贯当量差）——这是「知道鸣牌灭役」的直接检验。
5. **风格迁移预测（2026-08-16 发射前补注，源自用户提问「hazard-critic 能学到副露的缺点吗」）**：
   若「γV(s′)−V(s) 即时定价灭役损失」机制成立，A2-600k 相对 A0-600k（唯一差异=critic）应表现出
   **副露率显著下降、立直率显著上升**（eval_style_profile 各 4000 局，SE 按二项计）。背景基线：
   现役策略副露 ~88%、立直 ~3%（异常极端，人类高手 ~35-40%/~19%）。若风格无迁移而竞技场有增益，
   说明 critic 只优化了开门线内的细节；若风格迁移但竞技场 null，说明门清路线被高估——两种失败
   模式都有独立信息量。已知风险：P 是策略条件的，若共享曲面插值不足以打破「从不立直」的
   自我实现循环，此预测落空（设计文档「良性乐观闭环」的直接检验）。

## Progress
- [2026-08-15 ~15:30] 预注册。实现 eb91d8f；A2 冒烟通过（512 局端到端：45 维特征传输、BCE、
  GAE、里程碑、checkpoint 自描述 critic_feats 字段）。发射机制同 A1（看守进程）。
- [2026-08-15 ~15:50] 同 A1：us-central1-a spot 两次双机抢占 → 迁移 us-central1-b（b2）；
  A0 基线以 dnn_exp11_a0_20260815r3 从零重跑；--ckpt_every 25→10；runner 2 分钟 TB 轻量同步。
  本臂在 b2 跑完 A0-r3 后自动发射。

## Results
| Metric | This run | Baseline (A0) | Success criterion |
|---|---|---|---|
| （待运行） | | | |

## Conclusion
（待运行）

## Next Steps
- 国士场景重置试点（1000 局 2 向听开局）——设计文档遗留，另行预注册。

## Artifacts
| Path | Size | Description |
|---|---|---|
| gs://llm-mahjong-experiments/dnn_exp11_a2_20260815/ | — | 云端主目录（10 分钟增量同步） |
