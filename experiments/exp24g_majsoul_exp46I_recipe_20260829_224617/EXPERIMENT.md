# exp24g_majsoul_live — 雀魂实战（exp46I_recipe，仍在训练中的中间快照）

- **Date**: 2026-08-29 22:46  **Status**: running（服务已起，端口 8765）
- **Git**: decf106
- **模型**: `experiments/_cloud_ckpts/exp46I_recipe/games_final.pt`（从 GCS `gs://llm-mahjong-experiments/exp46I_recipe/` 拉取）
  = exp46 系列（强 BC 先验之上做 RL）第 I 个配方迭代，**云端仍在训练中**（拉取时 RUNNING.lock 在、
  train_log.json 2 分钟前刚写；ckpt 快照落后训练前沿 ~17 iter/14 万局，因 `ckpt_every=25`——
  正常，不是异常）。arch `convformer_m_v3r_m46`（与 exp49-B 同族），起点 = exp49-B 旗舰 BC，
  `bc_kl_coef=0.3` 锚定，warmup 150，恒定熵 0.0。
- **⚠️ 强度未定，且母实验此前判负**：`exp46_rl_on_prior_prereg`（A/B 裸臂+锚臂）已收官判定
  **RL 净毁值**（裸臂 −143 / 锚臂 −61 半庄 Elo，双双低于起点 BC 旗舰；防守侵蚀实锤）——
  exp46I 是该结论之后的后续配方搜索（C~J 系列之一），试图找到能让 RL 有正收益的配方。
  截至拉取时（iter 115，942k 局）：`gens.jsonl` 显示最近三代 gen_6/7/8 **均未通过晋级门**
  （gate_share 0.38-0.44，门槛未过）；`league_stats.jsonl` 里 learner vs "best"/"init" 的
  mean_diff 在正负之间大幅震荡（如 iter108 vs init +7.3，iter109 vs init −1102.1），
  **没有稳定跑赢自己起点的迹象**。换言之：**这不是已验证更强的模型，是训练中、母系列已判负
  背景下的探索性快照**。部署纯粹是用户明确要求，不代表强度判定。
- **模式**: 贪心 T=0，device cpu（同族 2.0M 参数，~4ms/决策）
- **前置校验**: `verify_mjai_bridge.py --games 20 --device cpu` OK（287 决策张量/合法集逐位一致）
- **切换时机**：上一部署（exp24f，exp49-B）当时空闲 75 分钟无对局，非中途打断。

## Purpose
用户明确要求换成 exp46I_recipe 的最新 checkpoint 实战观察，尽管其母实验/晋级门尚未给出正面证据。
后续可与 exp24f（同起点 exp49-B）的实战数据做行为对照。

## Progress
- [22:46] 服务启动，替换 exp24f（exp49-B）。
