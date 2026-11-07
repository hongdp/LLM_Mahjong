# akochan 可行性（论文 L5）（2026-11-06）

- **Date**: 2026-11-06 17:40 PST  **Status**: done（只做可行性，未跑对局）
- **Source**: github.com/critter-mj/akochan @ 53188a0（2022-07-05），浅克隆到 `/home/hongdp/Workspace/akochan_ext/akochan`（独立目录，不入本仓库）；用户 11-06 在对话中批准克隆与编译
- **Env**: 本机 g++ 13.3；Boost 1.84 装在独立 conda 环境 `akochan`（conda-forge，不改系统）；$0

## 许可（LICENSE，Shift-JIS，已转码阅读）
- 非公开研究可自由使用；**非营利**前提下，注明使用了本程序即可转载 / 发表其输出（牌谱、局面评估值）；其他用途需作者许可。
- 改动 AI 部分（ai_src）后再分发原则上禁止；知识产权归作者。
- ⇒ 论文把 akochan 作为对照对手、不改 ai_src、不分发二进制：可行；发表时需写明使用了 akochan 并致谢作者；若论文涉及营利出版需另询作者。

## 构建
- `ai_src`：`make -f Makefile_Linux`（CFLAGS 加 `-I<conda>/include`，LIBS `-L<conda>/lib -lboost_system`）→ libai.so（16.9 MB）。
- 根目录：`make -f Makefile_Linux` → system.exe（11.1 MB）。两步均 rc=0（只有警告）。运行需 `LD_LIBRARY_PATH=.:<conda>/lib`。

## 接口
- `system.exe pipe <tactics.json> <seat>`：stdin 逐行读 mjai 事件；在本家 tsumo、他家 dahai / kakan 时向 stdout 写一行 JSON 动作列表（如 `[{"type":"dahai","actor":0,"pai":"F","tsumogiri":false}]`、`[{"type":"none"}]`）。无需 TCP 服务器。
- `start_game` 必须带 `kyoku_first` 与 `aka_flag`（缺失即断言崩溃）；我们用 `kyoku_first: 4, aka_flag: true`。
- 默认 tactics（setup_mjai.json）`jun_pt [90,30,-30,-90]` 含顺位价值——单局复式评测时应改成点差口径或注明。

## 冒烟
- 本引擎 bc49 自对弈一局（种子 101112000）按座位 0 遮蔽后喂入：133 事件、63 次应对、6.4 s ⇒ **≈0.10 s / 应对**，动作格式与 MortalSeat 处理的 mjai 一致。

## 结论与工作量
- **可行**。接入 = 写 `AkochanSeat`（仿 MortalSeat：子进程 pipe，喂遮蔽事件，把应对列表转成引擎动作；需处理立直两步、鸣牌后同列表内的打牌），≈ 半天，加 300 牌山逐牌山校验（回退率 / 非法率）。
- 算力：每局约 2 个 akochan 座位 × ≈60 次应对 × 0.1 s ≈ 12 s；2000 牌山复式（4000 局）单进程 ≈ 13 h，12 进程并行 ≈ 1–2 h（需确认 akochan 自身 OpenMP 线程数，避免超订）；10,500 牌山 ≈ 5×。

## Artifacts
| Path | Description |
|---|---|
| `/home/hongdp/Workspace/akochan_ext/akochan/`（含 build_ai.log、build_main.log） | 源码与二进制（第三方，不入库、不分发） |
