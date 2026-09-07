# exp70 半庄顺位奖励从零（纯血线设定变更，用户 2026-09-06 批准）

- **Date**: 2026-09-06  **Status**: running（实现 + 冒烟）
- **Git**: 见 Progress；RunPod Secure L40S；纯血无核心资产、零人类数据（W 不用 exp55-D 的学习残差）
- **Env**: `train_dnn_ppo.py --hanchan_pure --hanchan_credit rank --arch cnn_m_rh`（本实验新增）；引擎字节不变（半庄层是驱动侧）

## Purpose & Hypothesis
exp68/69 证明从零 RL 的防守缺失不是感知问题：看得见暗手、预测得出待张都不多弃一张牌。诊断是"单局终局点差"设定下弃张不划算
（放铳几千分 ≪ 运气 σ≈5,000，四家同水平的均衡 = 都不防）。人类先验模型的防守来自人类数据内化的**多局顺位代价**。
本实验把 episode 改为整个半庄（E1–S4，连庄/本场/供托/飞），奖励 = 终局顺位 uma（±15k/±5k）+ 点差，每局按**解析的顺位势函数**
（当前名次 uma + 自身点差，`RankUmaCredit`，无学习、无人类数据）做精确 telescoping 的逐局归因。
**假设**：顺位刻度下放铳的代价被放大（尤其领先时），防守首次在纯血线涌现——H 臂暴露席放铳率 ≤ 单局配方产物 − 0.03、defense_iq ≥ 0.08，
半庄头对头对 P3（单局冠军配方）显著为正。**反假设**：horizon 变长 + 方差更大 → 同预算下学得更慢，强度反而低于单局配方；
或顺位压力只改变风险偏好（领先时保守）而不改进弃张选择。

## Method
- 生成器：`play_hanchan_gen`（exp46-D/55-D 既有）+ 新 `RankUmaCredit`（纯解析 W）；四席全为学习者（`--hanchan_pure`，dup_k 复式同 match 种子 = 组基线）。
- 编码：`cnn_m_rh` = cnn_m_r + 3 个场况标量（局数/4、本场/8、all-last），平面不变；单局评测时这些标量由 `randomize_round` 表提供，语义一致。
- 配方 = 钉死的冠军旗标：`--ppo_epochs 1 --adv_clamp 5.0 --target_kl 0.03 --batch 4096 --lr 1e-4 --entropy_schedule 0:0.03,600000:0.01
  --gamma 0.995 --gae_lambda 0.95 --dup_k 8`；`games_per_iter 256` 场（≈2,500 局/迭代，与单局 2048 局/迭代同阶）。
- **H 臂**：`--hanchan_credit rank`；**H0 臂**（消融）：`--hanchan_credit none`（逐局裸点差 + 终局 uma 一次性给，γ=0.995 下跨局信号基本衰减）。
  预算 150k 场/臂（≈1.5M 局）。单局对照 = exp68r3-P3 / G3 / exp27A（三个独立读数，噪声地板已知）。
- 终评（工作站）：①半庄头对头 H vs P3、H vs bc49（`play_pair_vector(hanchan=True)` n≥600 场，两向复式）；②单局配对牌山 H vs P3 / bc49
  （双段 20k 对，同刻度可比）；③defense_iq + 暴露席放铳率；④风格向量 vs bc49；⑤梯子 Elo（单局 + 半庄池）。

## Success Criteria（预注册）
1. **防守涌现**：H 暴露席放铳率 ≤ 0.17（P3 0.196、exp27A 0.156、bc49 0.126）且 defense_iq ≥ 0.08。
2. **强度（半庄刻度，主判据）**：H vs P3 半庄头对头 ≥ 0.55（n≥600 场）；H vs bc49 半庄 ≥ 0.30（P3 单局 0.33 作参照）。
3. **强度（单局刻度）**：H vs P3 ≥ 0.50（不因设定变更而毁掉单局能力）。
4. H vs H0：rank 归因是否优于裸 uma（≥0.53 → 归因有效）。
- 反假设成立（H vs P3 单局 <0.47 且放铳率无改善）→ 半庄设定不是纯血线的出路，记录并停。
- 预算：两臂 ≈ 4–6h ≈ **$6，上限 $10**。

## Progress
- [09-06 12:20] 用户批准立项。实现：`RankUmaCredit`（解析 W = 当前名次 uma + 自身点差；逐局 ΔW，末局付 true_uma − W(before)，
  测试验证对任意局序列 telescoping 到 true_uma − W(start)）；`--hanchan_pure`（四席学习者镜像，dup 复式 = 组基线；roles 统计改报 mean|uma|）；
  `--hanchan_credit {w,rank,none}`；`v1rh`/`cnn_m_rh`（+局数/本场/all-last 三标量）。测试 11/11；本机冒烟 96 场（4.6 场/s，8 worker）正常。
  发射：Secure L40S US-TX-4，H（rank）与 H0（none）并行，各 15 万场（≈1.5M 局），games_per_iter 256 场，熵 0.03→0.01@9 万场。
- [09-06 18:12 PDT] **发射 r1 夭折**：pod `2dwweiv0o0ywl4` 在 18:02–18:12 之间消失（ssh 拒连，RunPod API 404 "pod not found"，非 stop 而是删除；
  本会话脚本无任何 terminate 调用，删除来源外部/未知）。两臂各跑到 ~6,400 场（25 迭代，6.5 场/s，熵 1.89→1.72，H 臂 EV +0.02），
  首个 ckpt（iter 25）尚未拉回，本地只有 games_0.pt + 前 20 迭代 train_log。心跳按"发射死线"退出码 3 报警，拉取循环与 90k 探针已停。
  花费 ≈$0.45。r1 无可用结果；待用户确认后以同配置重发（r2）。
- [09-06 18:20 PDT] 用户确认非本人删除，批准重发。create-pod 返回 **402 账户余额不足** → r1 的 pod 消失原因确定为 RunPod 余额耗尽自动回收
  （不是故障）。r2 阻塞在充值，TB 三件套（exp70_H/H0_LIVE）与镜像目录仍在位，充值后同脚本直接重发。
- [09-06 19:16 PDT] **r2 发射**：用户充值后新建 Secure L40S US-TX-4 pod `rl9cqt2t8tncjb`（$1.09/h，ssh 195.26.232.163:49651），GPU 数值校验通过
  （ones matmul = 1e9、4096² 与 CPU 最大误差 2e-4）。同 repo.tar（代码自 r1 未变，只改文档）、同 exp70_pod_train.sh；H/H0 迭代 1 各 6.8/7.0 场/s。
  r1 残留目录移到 `experiments/exp70_r1_dead/`。心跳/拉取（含 TB 镜像 rsync 到 _cloud_mirror）/90k 探针重挂。预计 ~6h、≈$6.5，累计 ≈$7。
- [09-06 21:30 PDT] 100 min：两臂各 34,560 场（23%），吞吐降至 5.6–5.8 场/s（对局变长），ETA ≈02:40，pod 预计 ≈$8.2（累计 ≈$8.6，≤$10）。
  和牌率 22%/21%（已到单局配方水平），熵 1.41/1.32。**30k 早读探针**（T=0，400 局）：H defense_iq 0.073、H0 0.086；
  同阶段单局对照：P3@100k 局 −0.031（303 弱步）、P3@300k −0.029（仅 35 弱步）、exp27A@348k **+0.081**（92 弱步）。
  ⇒ 400 局探针 SE≈0.05，单局配方早期 ckpt 也能读出 0.08，30k 读数**不可判读**（训练阶段效应 vs 信号无法区分）。
  90k 探针改为 1,600 局并带 P3 终点同批对照（SE≈0.03）。教训：defense_iq 的样本量以暴露步数计，≥500 弱步才够 0.03 分辨率。

## Results

## Conclusion

## Next Steps

## Artifacts
