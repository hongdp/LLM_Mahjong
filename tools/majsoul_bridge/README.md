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
| `tools/majsoul_bridge/WINDOWS.md` + `mahjongcopilot_windows.patch` | **Windows 打牌机实录**：SxS Chromium 起不来 → 用系统 Chrome；mitm 缓冲 46 MB wasm → 流式透传；证书装用户存储免管理员；新版 Unity 客户端兼容情况；排障速查 |
| `tests/test_mjai_bridge.py` | 单测：随机 + 偏副露自对弈 80 局逐决策保真、红五簿记、抢杠 |

## 两种模式
| 模式 | MC 设置 | 我方 | 说明 |
|---|---|---|---|
| **自动打牌** | 「自动打牌」开；`ai_randomize_choice=0` | `--temperature 0`（贪心，与竞技场口径一致） | MC 把 bot 反应点进浏览器；用于正式计分 |
| **辅助打牌** | 「自动打牌」关 | 打开 `http://127.0.0.1:8765/panel` | 面板实时显示手牌、宝牌、合法动作的策略分布（条形）、**采样/贪心选中的动作**（高亮）与 V(s)，以及本局决策历史；你在雀魂里自己操作。`--temperature 1.0` = 显示采样动作，`0` = 贪心 |

两种模式下 MC 自己的 GUI/悬浮层也会显示我们的建议（通过 MJAI `meta` 概率），面板信息更全。
服务端只由 `--temperature` 决定采样/贪心，是否自动点击只由 MC 的开关决定，可随时切换而不重启服务。

## 牌局留底（两种模式相同）
服务端 `--log` 的 JSONL 里除原始事件外，每次决策在结果确定后写一条 `"kind": "decision"`；每个半庄结束
（`end_game`，或下一局 `/start`）再把整局写成 `<log目录>/games/game_<ts>_seat<N>.json`（`--games-dir` 可改）。
每条决策包含：
- `state`：决策前的完整观测——手牌（含红五标识）、摸牌、四家副露/牌河/牌河事件（巡目/摸切/立直宣言/被鸣）、
  宝牌、立直与立直巡、点数、供托、余牌、场风/局/本场、振听等；
- `actions` / `probs` / `value`：合法动作、策略概率分布、V(s)；`chosen`：策略选中（贪心或采样）；
- `reaction`：发给 MC 的 MJAI 反应；
- `executed`：**牌桌上实际发生的动作**（辅助模式你手动打的牌；或被雀魂拒绝后的超时摸切）、
  `executed_action`（引擎动作格式）、`override`（实际 ≠ 策略选择）。
- 每局附 `start`（start_kyoku）与 `result`（liqi 和牌/流局数据），整局附 `end_game` 终局。
`scripts/analyze_majsoul_session.py` 会额外汇报 recorded decisions / 人工覆盖率 / 平均 V / 平均 p(选中)。

## 模型机（model server）准备
模型机 = 放本仓库、checkpoint 和 torch 环境的那台；打牌机只需要 MC + 浏览器。两台机分工见下表，
单机部署时两者是同一台。

| 项 | 要求 / 做法 |
|---|---|
| 代码 | 本仓库（PR #5 之后的 master，或 `pr/majsoul-bridge` 分支） |
| Python 环境 | `conda activate rlhf_mahjong`（torch + `mahjong` 库 + numpy）；`python -m pytest tests/test_mjai_bridge.py` 应全绿 |
| checkpoint | 冠军 `experiments/_cloud_ckpts/dnn_exp17c_gae_20260818/games_final.pt`（23 MB，不入 git；无则 `gsutil cp gs://llm-mahjong-experiments/dnn_exp17c_gae_20260818/games_final.pt experiments/_cloud_ckpts/dnn_exp17c_gae_20260818/`） |
| 硬件 | CPU 即可（20M cnn 单步 <10 ms，默认 `--device cpu`）；GPU 不必要 |
| 保真自检（可选） | `PYTHONPATH=. python scripts/verify_mjai_bridge.py --ckpt <ckpt> --games 50` 打印 `OK {...}` |
| 实验目录 | 正式计分前按 ml-experiment-tracking 建 `experiments/exp24_majsoul_live_<ts>/`（预注册见 `experiments/exp24_majsoul_live_prereg/`）；日志与每局 JSON 都落这里 |
| 启动服务 | 见运行步骤 3；`--temperature 0`（贪心，exp25 证明比 T=1 强 +500 点/副）；`--name` 可设 MC 里显示的模型名 |
| 网络 | 服务默认只监听 `127.0.0.1:8765`（无鉴权，**不要**直接暴露到网络）。打牌机在另一台时用 SSH 隧道：打牌机执行 `ssh -N -L 8765:127.0.0.1:8765 <模型机>`，MC 侧 URL 保持 `http://127.0.0.1:8765`。可信局域网内也可 `--host 0.0.0.0` + MC `settings.json` 的 `llmmahjong_url` 指向模型机 IP |
| 心跳 | 按项目规则挂监控：`curl localhost:8765/health` 的 `decisions` 计数应随对局增长；`mjai_session.jsonl` mtime >40 min 不更新 = 已停止；日志里 `"kind": "error"` 行应为 0 |
| 常驻 | `nohup ... > server.log 2>&1 &` 或 tmux；重启服务会重置进行中对局的内存状态（JSONL 已写的决策行不丢，但当前半庄的 game JSON 会缺）——**在半庄之间重启** |
| 换模型 | 换 `--ckpt` 重启即可；`load_dnn` 自动识别 arch（cnn / vit / ConvFormer）与编码器版本（v1/v3） |

## 运行步骤
1. **准备 MC**（Python 3.12，独立 venv；Linux 上 tkinter 需要 `python3-tk`）
   ```bash
   git clone https://github.com/latorc/MahjongCopilot ~/MahjongCopilot
   cd ~/MahjongCopilot && python -m venv venv && source venv/bin/activate
   pip install -r requirements.txt && playwright install chromium
   ```
   Windows 打牌机还需要对 MC 打三处补丁（`git apply mahjongcopilot_windows.patch`），详见 [WINDOWS.md](WINDOWS.md)。
2. **安装插件**（幂等，可重复执行）
   ```bash
   python tools/majsoul_bridge/install.py ~/MahjongCopilot
   ```
3. **启动 agent 服务**（模型机，本仓库根目录，rlhf_mahjong 环境；准备工作见上节「模型机准备」）
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
