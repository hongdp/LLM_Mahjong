# 论文参照行：同判决牌山上的复式份额（Mortal / 随机 / 规则 / exp27A 1M）（2026-11-06）

- **Date**: 2026-11-06 06:27 PST  **Status**: running
- **Git**: 4ae29e1（worktree claude/rl-model-defense-learning-8fb5c0）；向量评测用 `/home/hongdp/Workspace/LLM_Mahjong_exp124`（f11278a，与 exp124 判决同一份代码）
- **Env**: 本机 RTX 4080 + 24 核；$0

## Purpose & Hypothesis
论文写作会话（local_1104c935，用户在该会话授权其调度本会话的 $0 本机实验）请求 L1：在 exp124 判决的**同一批牌山**上给出参照行，让份额刻度有锚点。
描述性测量，无 H1/H0。

## Method
- 牌山：10,500 个去重牌山 101112000–101122499（= 旧"2×10k"两段的并集，见 SKILLS 11-06），T=0 复式 2v2 对 bc49（A 坐 (0,2) 与 (1,3) 两个方位，B = bc49 T=0），份额 = 每牌山 sign(A−B 点差) 记 1/0.5/0。
- 参照：
  (a) Mortal 298k（mjai 黑盒桥，原生贪心）；
  (b) 均匀随机合法动作（每牌山每方位独立 RNG，不触碰引擎全局 RNG）；
  (c) 向听贪心规则 agent（新写）：能和就和；有立直就立直（立直牌同下述规则）；打牌按（打后向听最小，再按有效进张枚数最大——只扣自己手牌，不看场上）；从不吃碰杠、不九种九牌；
  (d) exp27A 1M（早期 CNN，`_cloud_ckpts/dnn_exp27_A_cnn_m_r_20260823/games_final.pt`）。
- (d) 走与判决相同的向量评测（`pairdiff_eval.py`，seed0 101112000，10,500 牌山）。(a)(b)(c) 走新写的逐局驱动 `dup_driver.py`：random.seed(wall) → `PyMahjongTable(randomize_round=True)` → `play_game_mjai`，与向量评测的牌山 / 场况 / 座位生成相同。
- **驱动校验（发射前必须过）**：用 exp124 A 32M 坐 A 方位跑前 300 牌山，逐牌山点差与存档 `pairdiff_exp124_A_n10000_games_32000000.pt_101112000.json` 比对；一致率 ≥ 99% 才接受（batch-1 与批量推理的极少数 argmax 平手可翻转）。
- 报告：份额 ± SE、每牌山点差 ± SE、一个 10k 牌山段在 4080 上的墙钟。

## Success Criteria
描述性；驱动校验一致率 ≥ 99%。

## Progress
- [2026-11-06 06:27 PST] 目录与预注册建立；写驱动。

- [2026-11-06 06:50 PST] **驱动校验通过**：exp124 A 32M 坐 A 方位、前 300 牌山，逐牌山点差与存档 **100% 一致**（300/300，份额 0.5333 = 0.5333）。10 牌山冒烟：随机 / 规则 / Mortal（572 决策零回退）均正常。发射：exp27A 向量读数 + 10k 计时 → Mortal / 规则 / 随机各 12 分片（每片 875 牌山）。

## Results

## Conclusion

## Artifacts
