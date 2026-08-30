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

## Progress
- [08-30 深夜] W 训练集抽取中；W 训练器就绪；MatchState/play_hanchan_gen 地基已合入（3672ba8）。
- [08-30] **W v1 定稿（残差参数化）**：`W = rank_uma解析式 + MLP残差`。纯 MLP 版在 S4 比解析基线差 1043 点（光滑网络拟合不了排名不连续）→ 残差版八盘位全部 ≥ 基线（E1 +1936，S4 -32≈平），整体 15140 vs 15480。工件：experiments/placement_value/{states.npz, w_resid.pt}（87 万行人类局间状态）。教训入档：信用函数必须分盘位验收，全局 MAE 会骗人。
