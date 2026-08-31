# exp55-D：半庄排位训练（用户设计，2026-08-30 深夜定稿）

- **Date**: 2026-08-30  **Status**: running（W 阶段）
- **Git**: 见各阶段 commit；引擎字节不变（半庄层驱动侧）
- **Env**: 云 G4（训练）；本地仅小任务

## Purpose & Hypothesis
最终目标 = 半庄排位（uma）。核心设计（用户洞察）：**每局点数精确可知 → 用排位价值函数 W 做势函数分解，把终局 uma 干净归因到每一局**：credit(局 i) = W(局后) − W(局前)，telescoping 精确、按排位语境加权（终盘接近时的点数 ≫ 领先时同额点数）。红利：**episode 保持单局尺寸**——dup-8/LOO/detach/锚全套已验证工具原样可用，方差不随 match 长度爆炸，目标却已是排位。

## Method（分阶段）
1. **W 阶段**：extract_placement_states.py 从 20,526 场人类半庄抽 (局间状态→终局uma) ~87 万行（无自举循环）；train_placement_value.py 小 MLP 双头（顺位 CE + uma 回归），按局哈希留出。**判据：holdout MAE 显著优于 rank-uma 基线**（按当前排名直接给 uma）——否则 W 无语境价值，降级用 rank-uma 解析式。
2. **生成器阶段**：play_hanchan_gen v2——每局末步奖励 = ΔW×0.001（替换裸点差），局间不跨 terminal；TrainHanchanTable 已零化 RANK_BONUS。
3. **编码器阶段**：scalar 追加局间语境（相对分×3、场次、本场、供托、亲位），bc49 热启动用零初始化投影行做形状手术。
4. **训练阶段**：四席 {bc锚 T0 / top(半庄Elo) T0 / 贪心自身 T0 / 学习者 T1}；I 配方全套（detach + 锚0.3 + 熵0 + 无裁剪）；贪心席 uma → hanchan_stats.jsonl = 训练内实时半庄 Elo。
5. dup：同 match 种子复式 + 席位轮转。

## Success Criteria（训练阶段，预注册）
①贪心席半庄 Elo 曲线持续正斜率；②终检半庄 n=300 vs bc49 ≥0.50；③T=0 族外梯子 ≥bc49_T0（1210.6）；④defense_iq ≥0.17；⑤bc_kl 走平（健康指纹）。

**判据③刻度重校（2026-08-30 深夜，发射前）**：原文 1210.6 是已作废的 12 锚外推刻度；
纪元 6 重锚后 bc49_T0 = **1189.0±7.9**（单局）/ 半庄 C 池 1443.1±11.0。判据③改读
「T=0 族外单局梯子 ≥ 1189.0」，实质不变（≥bc49 部署刻度）。终检加半庄口径
（vs bc49 双 T=0 n≥300，逐场计分——exp56 度量学）。

## 训练阶段 Config（首个 tranche，2026-08-30 深夜发射）
- 平台：RunPod 3090Ti community（exp58 已验证；runbook：cgroup <16 核 re-roll、
  bootstrap 补 tensorboard、kill 附清场、ckpt scp 逃逸、30min 自停保险）。
- 命令形态（= exp58 stage3 冒烟 + I 配方）：
  `--arch convformer_m_v3rh_m46 --init bc49_v3rh_init.pt --hanchan
   --hanchan_w_path w_resid.pt --league {bc49, exp46I} --value_detach
   --bc_anchor bc49_v3rh_init.pt --bc_kl_coef 0.3 --entropy_coef 0
   --gpu_infer --gpu_infer_opponents --games_per_worker 8 --ckpt_every 10`
  league 双元：entry0=bc49（bc 锚席），entry1=exp46I（top 席）——都是纪元 6 前二强。
- **规模：200,000 场半庄（≈2.1M 局）**；按 15.4 场/s（10 核）估 ~3.6h ≈ $1.0，
  23 核宿主更快。预算上限 **$3**（含 re-roll 与余量）。
- 心跳：发射相 15 分钟死线 + 训练相 20 分钟 STALL；TB 镜像 rsync-over-ssh 到
  `experiments/_cloud_mirror/exp55D_t1/`（两段式防 inode 换）。

## Progress
- [08-30 深夜] W 训练集抽取中；W 训练器就绪；MatchState/play_hanchan_gen 地基已合入（3672ba8）。
- [08-30 深夜] **训练阶段 tranche-1 发射**（exp58 验证链后首个 RunPod 正式 run）：
  pod `kkvad8c36c5l77`（3090Ti community $0.27/h，Threadripper 10.2 核宿主——runbook 的
  <16 核 re-roll 规则本次破例：库存 Low、成本无差、墙钟 ~5.8h 可接受，记录在案）。
  200k 场半庄，I 配方全套 + league={bc49, exp46I}，run 目录 pod:/workspace/exp55D_t1。
  首迭代 9.5 场/s（passes=4 完整口径），KL 0.0043 正常 → 预计 5.8h ≈ $1.58。
  心跳（STALL 20min/DONE/crash）+ TB 两段式镜像（exp55D_t1_LIVE 已挂 6006）+
  每 30min ckpt 回拉（损失上界 30min）全部就位。
- [08-30] **W v1 定稿（残差参数化）**：`W = rank_uma解析式 + MLP残差`。纯 MLP 版在 S4 比解析基线差 1043 点（光滑网络拟合不了排名不连续）→ 残差版八盘位全部 ≥ 基线（E1 +1936，S4 -32≈平），整体 15140 vs 15480。工件：experiments/placement_value/{states.npz, w_resid.pt}（87 万行人类局间状态）。教训入档：信用函数必须分盘位验收，全局 MAE 会骗人。
