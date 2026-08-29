# exp50（预注册）— 真 Mortal 入池：北极星的外部刻度

- **Date**: 2026-08-28  **Status**: building  **Cost**: 本地（libriichi 已构建，权重 130MB 已下载）
- **权重**: VoidShine/mortal-298k（HF，AGPL-3.0，社区训练：192×40 SEres、obs v4、天凤高段位
  牌谱 2025-26、**maka 平均 S+**≈魂天级）。仅本地评估，不再分发。

## Purpose

北极星（简单输入模型综合能力超 Mortal）此前只有内部尺。本实验把一个头部 Mortal 模型
放进我们的 Elo 池，取得绝对参照线：旗舰 bc49（1191.4）距真 Mortal 还有多远。

## Method

**黑箱路线（版本无关、逐位忠实）**：不走我们的 mortal_obs 移植（该权重是 v4，含未移植的
sp 求解器特征），而是用 Mortal 自家 libriichi（Rust，已按 conda py3.10 构建）在进程内做
obs 编码与决策：我们的引擎经 `play_game_mjai` 事件流驱动 `libriichi.mjai.Bot`。

工作项：
1. `mjai_export.play_game_mjai` 加 observer=None 全信息模式（默认行为不变），runner 侧按座位分发掩码视图；
2. `scripts/arena_mortal_mjai.py`：MortalSeat（Bot + 两步立直合并 + 反应→引擎动作转换 + 合法性守卫）、
   锚点 net 政策、league 同款 2v2 复式对局，**复用 run_elo_league 的 deal_scores/fit_ratings/rating_se**
   逐字生成可比 Elo，入 history.jsonl（标签 mortal298k_ext）；
3. 保真门：转换非法率 <0.5%（超限=桥有 bug，不出分）。

## 已知偏置（如实记录，方向均使 Mortal 读数偏低）

- 单局引擎 vs 其半庄训练分布（顺位 EV 语境缺失）；
- 我们的雀魂规则 vs 其天凤规则（三家和了/头跳等细节）；
- Mortal 走自身贪心（其部署口径），池内我方模型为 T=1 口径。

## Success Criteria

1. 保真门过 + 9 锚 deals=100 出分 → **绝对参照线入池**（预期显著 >1191；若 <1191 则
   北极星第一阶段已名义达成，需在偏置澄清后谨慎宣称）；
2. 附防守探针（同 seed）：Mortal 的 defense_iq 对照我方 0.165/0.198；
3. 差距数字直接标定 exp46（RL-on-prior）与数据扩容的目标区间。

## Progress
- [2026-08-28] rustup + libriichi（py3.10 ABI 重编）构建通过；ckpt 结构确认（Brain 10.8M + DQN，v4，298k steps）。
