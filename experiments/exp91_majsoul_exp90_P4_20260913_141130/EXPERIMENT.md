# exp91_majsoul_live — 雀魂实战（exp90_P4，~97.5M 局，96M 判读点附近）

- **Date**: 2026-09-13 14:11  **Status**: running（服务已起，端口 8765）
- **Git**: 2c96562
- **模型**: `experiments/_cloud_ckpts/exp90_P4/latest.pt`（从 GCS `gs://llm-mahjong-experiments/exp90_P4/` 拉取）
  = 纯血 RL 主线 exp90 的 P4 臂，**云端仍在训练中**（拉取时 RUNNING.lock 在，train_log.json 实时更新；
  拉取时 games=97,484,800/iter=47600，当前训练前沿已到 97,771,520——run 本身只保留一份滚动
  `latest.pt`，没有按 milestone 单独存档，这是能拿到的、离 96M 判读点最近的快照，偏差约 +1.5M 局）。
  arch `cnn_m_v3r`（**原生 374 槽动作空间**，非 exp46/49 系的 46 槽 mortal 空间——无需桥接改造）；
  `resume: /workspace/pool/M.pt`（从锚点 M 续训）；`milestones` 含 84M/88M/92M/**96M**/100M/104M/108M；
  entropy_coef 0.01、houjuu_extra −4.0、shaping 已在 400k 局退火到 0、无 hanchan/league，
  单局纯自对弈继续训练配方（exp88/89/90 系）。
- **背景（见 memory pure-line-200usd-goal / give-models-enough-training-time）**：这是 $200 预算内
  「纯血从零 RL 追平 bc49」目标下的在跑序列；exp87 PSRO 最佳响应判空（M 在本谱系不可剥削）、
  exp88（熵 0.03+KL-to-M）判读点 92M；用户规则「给够训练时间，判读用终点，不要过早截断」。
  **本 checkpoint 尚无 Elo/竞技场判定**——纯粹是训练中间快照，雀魂部署仅作行为观察，不代表已验证强度。
- **前置校验**: `verify_mjai_bridge.py --games 20 --device cpu` OK（314 决策张量/合法集逐位一致）
- **模式**: 贪心 T=0，device cpu

## Purpose
观察 exp90_P4（纯血 RL 主线，96M 判读点附近）在真人雀魂对局中的行为，与此前部署过的
人类 BC 谱系（exp24f/exp49-B、exp24g/exp46I_recipe）以及纯血自对弈谱系（exp24b/exp27-A）对照。
**强度不作判定**：checkpoint 未评级，且训练仍在继续。

## Progress
- [14:11] 服务启动。
