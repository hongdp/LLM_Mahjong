# exp24_majsoul_live — 雀魂实战评估（预注册，未发射）

- **Date**: 2026-08-23 预注册  **Status**: planned（基建完成、离线验证通过；**发射需用户操作**：雀魂账号 + 本机浏览器）
- **Git**: 本 commit（tools/majsoul_bridge + src/agents/dnn/mjai_bridge.py）
- **Env**: 本机 CPU 推理（20M cnn 单步 <10ms）；MahjongCopilot 0.6.0 + mitmproxy；被测 = 纯自对弈冠军
  `experiments/_cloud_ckpts/dnn_exp17c_gae_20260818/games_final.pt`（Elo 池 1079.7）

## Purpose & Hypothesis
北极星 (a) 的外部标尺：Elo 池里只有自家谱系 + 教师参照线，从未和**人类**交手。把冠军接入雀魂，
拿到第一批「对人类」的顺位/放铳/和牌数据，回答两个问题：
1. 自对弈生态里「推牌近似最优」的策略（defense_iq≈0、曝露放铳率 19-24%）面对人类立直时放铳率是多少——
   预期显著高于人类同段位均值（雀魂玉之间放铳率 ~12-13%），是「防守未涌现」的外部确认。
2. 顺位期望：假说 H1 = 平均顺位 > 2.5（弱于对手池），H0 = ≈2.5。

## Method
- 引擎→MJAI 影子桌（`ShadowTable`）复用训练编码器与合法动作生成，离线保真度：400 局 7004 次决策张量逐位一致
  （`scripts/verify_mjai_bridge.py`）。
- MahjongCopilot 负责 liqi 抓包 → MJAI 事件 → 我方 HTTP bot（`scripts/serve_mjai_bot.py`）→ 自动点击。
- **对局模式（预注册）**：段位场 4 人东风/半庄均可，但统计按半庄计；友人房可用于首次联调（不计入）。
- 贪心（temperature 0），与竞技场评分口径一致。**计分只用自动打牌模式**；辅助模式（`/panel` 显示分布 +
  采样动作、用户手动操作）用于联调与观察，其会话**不计入**本实验（混入人类判断）。
- 已知失真（记录，不修）：红宝牌折叠为普通 5；本场数不可见；西场按南场编码；九种九牌不宣告。

## Config
`serve_mjai_bot.py --ckpt <exp17-C final> --temperature 0 --log experiments/exp24_majsoul_live_<ts>/mjai_session.jsonl`；
MahjongCopilot settings: model_type=LLM_Mahjong, enable_automation=true, ai_randomize_choice=0, delay 默认。

## Success Criteria（发射前定）
- 样本：≥ 30 半庄（≈250 局）才做结论；每 10 半庄用 `scripts/analyze_majsoul_session.py` 出一次中间报告。
- 主指标：平均顺位（SE 按半庄计）；副指标：放铳率 / 和牌率 / 立直率 / 副露率（与自对弈 0.24 / ~0.4 对照）。
- 「可用」门槛：整段会话里 bot 反应与雀魂可选操作不匹配（MahjongCopilot 日志 `no op list`）< 1%，
  超时自动摸切 < 2%；否则先修桥接再计分。
- 合规提示：使用第三方工具属雀魂 ToS 灰区，有封号风险——**只用可承受损失的小号**；这是用户决定。

## Progress
- [08-23] 基建完成：桥接 + HTTP 服务 + MahjongCopilot 插件安装器 + 会话分析器；离线 400 局保真通过；
  端到端（真实插件类 + factory 注册 + meta_options）6 局回放通过。**等待用户发射**。

## Results
| Metric | This run | Baseline | Success criterion |
|---|---|---|---|
| 平均顺位 | – | 2.5 | 报告 + SE |
| 放铳率 | – | 自对弈曝露 19-24%；人类玉之间 ~12% | 报告 |

## Conclusion
（待）

## Next Steps
（待）

## Artifacts
| Path | Size | Description |
|---|---|---|
| tools/majsoul_bridge/README.md | – | 运行手册 |
