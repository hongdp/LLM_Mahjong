# majsoul_exp113C — 雀魂实战（纯血 RL 线最新最强：exp113 C 384M）

- **Date**: 2026-10-25 00:07  **Status**: running（服务已起，端口 8765）
- **模型**: `experiments/exp113_C/games_384000000.pt`（arch `cnn_l_v3r`，4.0M 参数，原生 374 槽动作空间，
  编码器 v3r）= exp113「数量级规模」对照臂 C，exp107 P 256M 起、同配方续训到 384M 局终点。
- **为什么选它（纯血线口径，依据记忆 pure-line-200usd-goal）**：纯血 T=0 平台 ≈0.47 vs bc49；exp113 窗口 360–384M
  C 0.4703 = L 0.4703（配对差 +0.00±0.20，容量轴 H0）；exp107 P 256M 窗口 0.4695——三者在噪声内并列纯血最高。
  取最新一条的终点（C 384M）；L（384×12，21M 参数）同分但 5× 大，C 小/快（CPU 2 ms）。
  **未选 exp116 G0/G1/G2**（0.476）：规则特征先验臂 + v3rg 编码 + 多训，用户裁定不进纯血榜。
  **单 ckpt 读数噪声 ±1 pp，判读以窗口均值为准**；这里只是窗口内的终点快照，不是单点最优。
- **代码来源**：当前 checkout 的 arch_zoo 不含 `cnn_l_v3r`（新代码在 runpod worktree 分支 1736dfe，
  只读使用，未改动）：`cd .claude/worktrees/runpod-training-cost-optimization-626ab2 && PYTHONPATH=. python scripts/serve_mjai_bot.py ...`
- **前置校验**: verify_mjai_bridge.py --games 20 --device cpu OK（348 决策张量/合法集逐位一致）；CPU 2 ms/决策。
- **模式**: 贪心 T=0，device cpu，端口 8765。
- **对照**: 人类先验谱系（exp24f bc49 家族 20 局 1.79、exp24g exp46I_recipe 54 局 1.87）；
  纯血 exp27-A 雀魂 maka C+、exp90_P4@96M B-（雀魂 2 局 3.00±0.71，未收尾）。

## Purpose
让纯血线当前最强模型（≈0.47 vs bc49，自对弈尺度）上雀魂真人局，取行为/顺位观察。强度不作判定（样本小）。

## Progress
- [00:07] 服务启动。
