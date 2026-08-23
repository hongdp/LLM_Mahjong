# 雀魂实战：Windows 打牌机部署与排障记录

日期：2026-08-23。配套 [`README.md`](README.md)（通用流程）；本文只记 Windows 打牌机上**实际踩到的坑和修法**。首次完整跑通：exp24，友人房辅助模式一局，bot 决策 150 次，`no op list` 0 次。

## 1. 拓扑

```
打牌机 (Windows 11, build 26200)                 模型机 (ubuntu-server, 192.168.0.110)
  MahjongCopilot (MC) + 插件                        serve_mjai_bot.py  (127.0.0.1:8765)
    └─ mitmproxy :10999 ──► Chrome(雀魂)                 ▲
    └─ bot_llmmahjong ──► 127.0.0.1:8765 ──ssh -L 隧道──┘
```

模型机只监听本机，打牌机通过 `ssh -N -L 8765:127.0.0.1:8765 ubuntu-server` 转发。

## 2. 打牌机环境

| 项 | 实际情况 |
|---|---|
| Python | 机器上只有 miniconda (3.7) 且 conda 联网失败；用 `winget install Python.Python.3.12 --scope user` 装到 `%LOCALAPPDATA%\Programs\Python\Python312` |
| MC | `git clone https://github.com/latorc/MahjongCopilot` @ `31be3de` (0.6.0, 2025-03-22)，`python -m venv venv`，`pip install -r requirements.txt`，`playwright install chromium` |
| 插件 | `python ~/majsoul_bridge/install.py ~/MahjongCopilot` |
| 浏览器 | 本机 Chrome 151（见 §3.1，不用 Playwright 自带 Chromium） |

MC 启动命令（不要用 MC 的 `--help`/exe 版本）：

```powershell
cd $env:USERPROFILE\MahjongCopilot; .\venv\Scripts\python.exe main.py
```

日志在 `MahjongCopilot\log\majsoul_copilot_<ts>.log`（stderr 也可重定向到文件看）。

## 3. 对 MahjongCopilot 的三处补丁

补丁文件：[`mahjongcopilot_windows.patch`](mahjongcopilot_windows.patch)（基于 MC `31be3de` 生成，已验证 `git apply --check` 通过），在 MC 检出目录 `git apply <path>\mahjongcopilot_windows.patch` 即可。`install.py` 打的插件改动（`bot/factory.py`、`common/settings.py`、`game/game_state.py`）不在此 patch 内。

### 3.1 `game/browser.py`：用系统 Chrome，不用自带 Chromium

**现象**：点「启动浏览器」报 `playwright._impl._errors.Error: spawn UNKNOWN`。直接运行 `ms-playwright\chromium-1105\chrome-win\chrome.exe` 报 "side-by-side configuration is incorrect"；事件日志 SideBySide：`Dependent Assembly 123.0.6312.4 could not be found`。重新 `playwright install chromium` 无效——是这个 Windows build 与 Chromium 123 的 SxS 兼容问题，不是下载损坏。

**修法**：两处 `launch_persistent_context(...)`（一处是启用 Chrome 扩展的分支，一处是普通分支）加 `channel="chrome"`，Playwright 改用系统安装的 Google Chrome（本机 151），其余参数（持久化 profile、代理、视口）不变。没装 Chrome 的机器可改成 `"msedge"`。

### 3.2 `mitm.py`：大响应流式透传

**现象**：浏览器起来后雀魂黑屏 / 一直"正在下载网络资源"。用 Playwright 挂 request/response 事件对比：不走代理 207 个响应，走代理只有 12 个，卡住的那个是 `Build/chs_t-WebGL-release-4.0.46(46).wasm.gz`（10 MB，服务器还带 `Content-Encoding: gzip`，解压后 46 MB）。mitmproxy 默认把整个响应 body 读进内存、按 Content-Encoding 解码、再交给 addon，Chrome 在这期间收不到任何字节，超时后游戏报错。curl 走代理能下完只是因为它肯等。

**修法**：`_run_mitm_async` 里 `DumpMaster(...)` 构造**之后** `self.dump_master.options.update(stream_large_bodies="1m")`：超过 1 MB 的 body 不缓冲、不解码，边收边转发。MC 需要拦截的只是 WebSocket 对局消息（几百字节），完全不受影响。
顺序很重要：这个选项由 mitmproxy 的默认 addon 注册，只有在 `DumpMaster` 构造完成后才存在。写进 `options.Options(listen_port=..., stream_large_bodies="1m")` 会让 mitm 线程直接抛 `KeyError: 'Unknown options: stream_large_bodies'`，代理起不来，浏览器报 `ERR_PROXY_CONNECTION_FAILED`。

### 3.3 `common/utils.py`：证书装进当前用户存储

**现象**：游戏都进到登录界面了，MC 覆盖层左下角仍显示 **「主进程发生错误!」**。这不是游戏报错，是 `bot_manager._create_mitm_and_proxinject()` 里 `install_mitm_cert()` 返回 False 后置的 `main_thread_exception`（主循环其实照常跑）。根因：MC 用 `certutil -addstore Root` 往**系统**存储装 mitm CA，非管理员会 "requires elevation"。

**修法**：`is_certificate_installed` / `install_root_cert` 的 certutil 都加 `-user`，改用当前用户（CurrentUser）的 Root 存储：不需要管理员；Chrome 同样信任用户存储里的根证书（实测能正常过 TLS）；首次安装会弹 Windows 的"您即将安装来自某 CA 的证书"确认框，点"是"。**查询那一处的 `-user` 同样必要**：检查函数查的是本机存储，否则即使证书已在用户存储里，MC 也判定"没装"，每次启动都会尝试安装、失败、再置错误标志。手动装也可以：

```powershell
certutil -user -addstore Root $env:USERPROFILE\MahjongCopilot\mitm_config\mitmproxy-ca-cert.cer
```

（`Import-Certificate` 在非交互 shell 里会报 "UI is not allowed"，用 certutil。）

## 4. MC 设置（`settings.json`）

直接改文件比走 GUI 省事，改完重启 MC：

```json
"model_type": "LLM_Mahjong",
"llmmahjong_url": "http://127.0.0.1:8765",
"enable_automation": false,      // 辅助模式；自动模式改 true
"ai_randomize_choice": 0,
"enable_overlay": true
```

## 5. 新版雀魂客户端（Unity WebGL 4.0.46）与 MC 0.6.0 的兼容情况

2026-08 的雀魂网页版已换成 Unity 构建（`res 0.16.269 / client 4.0.46`），MC 0.6.0 的 `liqi_proto` 是旧表。实测：

- 登录 (`.lq.Lobby.oauth2Login`)、开局 (`.lq.FastTest.authGame`)、对局 (`.lq.ActionPrototype`、`inputOperation`、`inputChiPengGang`) **都能正常解析**，bot 正常出牌。
- 日志会大量刷 `Failed to parse liqi msg ... Error: 'Route'`（`.lq.Route.requestConnection` / `heartbeat`）以及 `'addRoomRobot'`、`'roomKickPlayer'`：都是新加的路由层 / 房间管理方法，MC 解析失败直接跳过，**无害**。想清净可把这些方法名加进 `bot_manager.METHODS_TO_IGNORE` 前的解析异常过滤，或更新 `liqi_proto`。

## 6. 验收清单（辅助模式一局）

1. `curl localhost:8765/health` → `ok: true`，`decisions` 计数随对局增长（本次 153，与 MC 日志 `Bot out` 150 条 + 若干 none 一致）。
2. MC 日志出现 `Lobby login done`、`Game Started. Game Flow ID=...`、`Bot in: {'type': 'start_game'...}`。
3. 每次 `Bot in: tsumo` 后有 `Bot out: {'type': 'dahai' ...}`；`no op list` 0 次。
4. 面板 `http://127.0.0.1:8765/panel` 推荐与 `Bot out` 一致（抽查：bot 出 `P`，游戏里打出 `5z` 白）。
5. 模型机 `experiments/exp24_majsoul_live_<ts>/mjai_session.jsonl` 在增长；打完跑 `scripts/analyze_majsoul_session.py`。

自动模式额外注意：MC 靠点击页面坐标出牌，Chrome 窗口保持 1280×720、缩放 100%，不要部分移出屏幕；覆盖层提示「检查缩放」时先修。

## 7. 排障速查

| 症状 | 看哪里 | 原因 → 修法 |
|---|---|---|
| `spawn UNKNOWN` | MC 日志 | 自带 Chromium 起不来 → §3.1 |
| `ERR_CERT_AUTHORITY_INVALID` | MC 日志 | 证书没装 → §3.3 手动 certutil |
| 游戏黑屏/卡进度条 | Playwright 抓 `requestfailed`/pending | `wasm.gz` 被 mitm 缓冲 → §3.2 |
| 覆盖层「主进程发生错误!」但游戏正常 | `bot_manager.py:473` | 证书自动安装失败置的标志 → §3.3 |
| `ERR_PROXY_CONNECTION_FAILED` | MC 日志开头 | mitm 线程起失败（如 `Unknown options`），看 Traceback |
| `Error: 'Route'` 刷屏 | — | 新客户端心跳，忽略 → §5 |

## 8. 便利脚本

开隧道（最小化窗口常驻）：

```powershell
Start-Process ssh -ArgumentList "-N","-L","8765:127.0.0.1:8765","ubuntu-server" -WindowStyle Minimized
```

复现页面加载问题时用的探针思路：用 `playwright.chromium.launch(channel="chrome", proxy={"server": "http://127.0.0.1:10999"})` 打开 `https://game.maj-soul.com/1/`，挂 `request`/`response`/`requestfailed`/`websocket` 事件，对比走代理与不走代理的响应数，就能定位到被代理卡住的那个请求。
