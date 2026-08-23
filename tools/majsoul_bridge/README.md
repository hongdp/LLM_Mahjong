# 雀魂（Majsoul）实战桥接 —— 仿 MahjongCopilot 的接入方案

把本项目的 DNN agent 接入雀魂网页版做实战性能测试。方案完全仿照
[MahjongCopilot](https://github.com/latorc/MahjongCopilot)（以下简称 MC）：
MC 用 mitmproxy 抓雀魂 websocket（liqi protobuf）→ 翻译成 **MJAI 协议**事件 → 问 bot 要反应
→ playwright 在浏览器里自动点击。MC 原生只支持 Mortal 系 bot，我们加了第四种 bot 类型
`LLM_Mahjong`，通过本地 HTTP 调用我们的 agent。

```
雀魂网页 ⇄ mitmproxy(MC) ⇄ MC GameState(liqi→MJAI) ⇄ bot/llmmahjong(HTTP 客户端)
                                                       ⇅  http://127.0.0.1:8765
                              LLM_Mahjong: scripts/serve_mjai_bot.py
                                  └ MjaiDnnBot → ShadowTable(引擎影子桌) → encode_state → 策略网络
```

## 组件（本仓库）
| 文件 | 作用 |
|---|---|
| `src/agents/dnn/mjai_bridge.py` | `ShadowTable`（继承引擎 `PyMahjongTable`，由 MJAI 事件驱动，复用编码器与合法动作生成）+ `MjaiDnnBot`（MJAI bot） |
| `src/agents/dnn/mjai_export.py` | 引擎自对弈 → MJAI 事件流（测试用；也可导出 mjai 日志） |
| `scripts/serve_mjai_bot.py` | HTTP 服务：`/start` `/react` `/react_batch` `/health` `/last` `/state`；`/panel` 辅助面板；全量事件落 JSONL |
| `scripts/verify_mjai_bridge.py` | 用真实 checkpoint 做保真校验（影子桌张量 / 合法集 逐位对比引擎） |
| `scripts/analyze_majsoul_session.py` | 会话日志 → 顺位 / 和牌率 / 放铳率 / 立直率 / 副露率 |
| `tools/majsoul_bridge/bot_llmmahjong.py` | MC 侧插件（HTTP 客户端，实现 MC 的 `Bot` 接口） |
| `tools/majsoul_bridge/install.py` | 一键把插件装进 MC 检出（注册 factory / settings / 转发局结果） |
| `tests/test_mjai_bridge.py` | 单测：随机 + 偏副露自对弈 80 局逐决策保真、红五簿记、抢杠 |

## 两种模式
| 模式 | MC 设置 | 我方 | 说明 |
|---|---|---|---|
| **自动打牌** | 「自动打牌」开；`ai_randomize_choice=0` | `--temperature 0`（贪心，与竞技场口径一致） | MC 把 bot 反应点进浏览器；用于正式计分 |
| **辅助打牌** | 「自动打牌」关 | 打开 `http://127.0.0.1:8765/panel` | 面板实时显示手牌、宝牌、合法动作的策略分布（条形）、**采样/贪心选中的动作**（高亮）与 V(s)，以及本局决策历史；你在雀魂里自己操作。`--temperature 1.0` = 显示采样动作，`0` = 贪心 |

两种模式下 MC 自己的 GUI/悬浮层也会显示我们的建议（通过 MJAI `meta` 概率），面板信息更全。
服务端只由 `--temperature` 决定采样/贪心，是否自动点击只由 MC 的开关决定，可随时切换而不重启服务。

## 运行步骤
1. **准备 MC**（Python 3.12，独立 venv；Linux 上 tkinter 需要 `python3-tk`）
   ```bash
   git clone https://github.com/latorc/MahjongCopilot ~/MahjongCopilot
   cd ~/MahjongCopilot && python -m venv venv && source venv/bin/activate
   pip install -r requirements.txt && playwright install chromium
   ```
2. **安装插件**（幂等，可重复执行）
   ```bash
   python tools/majsoul_bridge/install.py ~/MahjongCopilot
   ```
3. **启动 agent 服务**（本仓库根目录，rlhf_mahjong 环境；先按 ml-experiment-tracking 规则建好实验目录）
   ```bash
   PYTHONPATH=. python scripts/serve_mjai_bot.py --ckpt experiments/_cloud_ckpts/dnn_exp17c_gae_20260818/games_final.pt --temperature 0 --log experiments/exp24_majsoul_live_<ts>/mjai_session.jsonl
   ```
   `curl localhost:8765/health` 应返回 ok。
4. **启动 MC**：`python main.py` → 设置 → 模型类型选 `LLM_Mahjong`（URL 默认 127.0.0.1:8765）→ 保存；
   主界面「启动浏览器」（MC 自带带代理的 Chromium，首次需装 mitm 证书，见 MC readme）→ 登录雀魂 →
   勾选「自动打牌」。首次联调建议在**友人房**看几局，确认 MC 日志里 `Bot out:` 与牌桌操作一致。
5. **计分**
   ```bash
   python scripts/analyze_majsoul_session.py experiments/exp24_majsoul_live_<ts>/mjai_session.jsonl
   ```
   （顺位/放铳依赖 install.py 对 `game_state.py` 的结果转发补丁。）

## 协议要点（实现时踩过的坑）
- MC 只在雀魂给出 operationList 时才问 bot；bot 自己算合法动作（影子桌复用引擎 `get_legal_actions` /
  `get_interrupt_actions`），返回的动作若雀魂没提供，MC 会记 `no op list` 并放弃 → 由超时自动摸切。
- 立直：反应是 `reach` + 附带 `reach_dahai`（MC 约定）；随后 MC 会回灌 `reach`+`dahai` 事件，影子桌以事件为准。
- 杠：MJAI 在杠后单独发 `tsumo`（岭上）；引擎是在杠里隐式摸牌。影子桌按「每个 tsumo 事件 wall−1」计，
  与引擎 70 张活牌一致（已在保真测试中逐局核对）。
- 抢杠：`kakan` 事件到达时先判定是否荣和，再把加杠写入影子桌。
- `end_kyoku`/`end_game` MC 原版不发给 bot，install.py 补丁转发 liqi 结果用于计分。
- 红五：引擎无赤宝牌，观测折叠为普通 5；出牌时优先打普通 5 留红 5（`ShadowTable.physical`）。

## 已知失真（不影响合法性，影响强度上限）
红宝牌价值不可见 · 本场数不可见 · 西场按南场编码 · 不宣告九种九牌 · 不支持三麻。

## 风险
使用第三方自动化工具违反雀魂服务条款，**存在封号风险**，只用可承受损失的账号；这是用户的决定。
