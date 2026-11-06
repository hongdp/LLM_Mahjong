# exp58 — RunPod 迁移验证：3090Ti 训练全链路 + resume 演练

- **Date**: 2026-08-30 20:40  **Status**: done（22:10 收官，pod 已 terminate，实花 ~$0.36）
- **Git**: 876164d（master 已含 exp57 裁决）
- **Env**: RunPod 3090 Ti community（$0.27/h）；对照 = 本机 4080 / g4-48 的训练口径历史数字

## Purpose & Hypothesis
exp57 判了「换机」，但留了三个迁移前提。本实验补掉可实测的两个：
1. **训练全链路**（不只 rollout）：PPO 更新步在 3090Ti 上吞吐正常、显存不炸、ckpt 能逃逸出 pod；
2. **resume 演练**：训练中途 kill，`--resume` 续跑，games 计数连续、无 NaN、指标曲线无跳变
   ——community 机器会坏，这是迁移的生死前提（SKILLS：spot/抢占式 run 的先例）。
第三前提（长跑 26h 可靠性）只能在真迁移中观察，本实验记录 pod 稳定性观察窗即止。
**顺带产出**：exp55-D 形态（--hanchan + W 信用 + v3rh）在 3090Ti 的实测训练吞吐 → 直接用于 exp55-D 报价。

## Method
3090Ti community pod（同 exp57 协议），git clone @876164d + scp 少量工件
（bc49_v3rh_init.pt、w_resid.pt、bc49.pt）：
1. **deal 模式训练冒烟**：`train_dnn_ppo --arch cnn_m_r --gpu_infer` 短 run（~40k 局），
   记训练口径 局/s、GPU util、显存；
2. **kill→resume 演练**：训练中途 kill 训练器进程（不碰容器），`--resume` 最新 ckpt 续到目标局数；
   校验 train_log.json 的 games 序列连续、resume 前后指标无断崖、无 NaN；
3. **exp55-D 形态冒烟**：`--hanchan --hanchan_w_path w_resid.pt` + v3rh 编码器 + bc49_v3rh 热启动，
   短 run 记 半庄/s 训练口径；
4. **ckpt 逃逸**：ckpt scp 回工作站 + sha256 比对；GCS 直推若无凭证则记 gap（不阻塞判定）。

## Config
- 预算：**≤$2**（约 3-7h 余量；预计实用 ~1.5h ≈ $0.5）。用完 terminate。
- 心跳：发射相 15 分钟死线；训练相 log 静默 20 分钟报 STALL；pod 侧 30 分钟自停保险。

## Success Criteria（预注册）
1. deal 训练口径 ≥ **85 局/s**（= exp57 rollout 214.5 的 40%，历史训练/rollout 比的下沿）；
2. resume 演练：games 连续 + 无 NaN + 续跑后 loss/EV 与 kill 前同量级（无断崖）；
3. exp55-D 形态跑通且吞吐可测（给出 半庄/s 数字即可，无阈值——首个数据点）；
4. ckpt sha256 一致地逃逸出 pod。
**全过 ⇒ 解除 exp57 的「正式长训练仍走 g4」红线（长跑可靠性单列观察）；
任一不过 ⇒ 红线保持，缺陷立项。**

## Progress
- [20:40] 预注册；pod 待开。
- [20:50] pod `0na74mzeppqd9g` 就绪（$0.27/h 计费自 03:46:53Z）。**宿主异构性实锤**：
  这台是 Threadripper PRO 5965WX，cgroup 只保证 **10.2 核**（exp57 那台 EPYC 给 23.8）——
  同价同卡不同核。**迁移规则草案：pod 起来先读 `/sys/fs/cgroup/cpu.max`，低于 16 核 re-roll**。
  本实验判据 1 按每核归一解读（预注册值假设了 ~23 核）。
- [20:55] stage1 发射：cnn_m_r 40k 局，workers=10，K=32，gpu_infer，ckpt_every=5；
  pod 侧 30 分钟静默自停保险已挂。坑：镜像缺 tensorboard（trainer 硬依赖），pip 补装。
- [21:10] **stage1 完成：训练口径 189.7 局/s**（40,960 局 3.6 分钟，10 核宿主）——判据 1
  的 85 局/s 假设 23 核，实测 10 核就翻倍过线；≈ 本机 4080 训练口径（~100）的 1.9×。
- [21:25] **stage2 resume 演练通过**：200k 局 run 在 81,920 局处 kill（残留 worker 需手动
  kill -9 清场——正式迁移脚本要带清场步），`--resume games_80000.pt` 续起
  （"81920 games, 40 log rows kept"），iter 编号/games/熵曲线无缝（H 1.615→1.569→1.556），
  收官 200,704 局、全程 191.7 局/s、无 NaN。判据 2 过。
- [21:35] **stage3 exp55-D 完整形态跑通**：convformer_m_v3rh_m46 + bc49_v3rh 热启动 +
  `--hanchan` + w_resid W 信用 + league 四席，**15.4 场/s（≈165 局/s）训练口径**，
  首迭代 KL/EV 正常。判据 3 过（首个吞吐数据点到手）。

## Results

| 判据 | 预注册目标 | 实测 | 判定 |
|---|---|---|---|
| 1. deal 训练吞吐 | ≥85 局/s（按 ~23 核设） | **189.7–192.2 局/s（10 核宿主）** | ✅ 翻倍过线 |
| 2. resume 演练 | games 连续+无 NaN+无断崖 | 81,920 局 kill→续起，日志保留 40 行，H/win 曲线无缝，收官 200,704 局 | ✅ |
| 3. exp55-D 形态 | 跑通+吞吐可测 | v3rh 热启动+W 信用+league 四席全通，**15.4 场/s（≈165 局/s）** | ✅ |
| 4. ckpt 逃逸 | sha256 一致 | scp 回工作站，sha256 完全一致 | ✅ |

宿主：Threadripper PRO 5965WX，cgroup 10.2 核（vs exp57 那台 EPYC 23.8 核——同价同卡不同核）。
花费：~80 分钟 ≈ **$0.36**（预算 $2）。

## Conclusion

**四判据全过 ⇒ 解除 exp57 的「正式长训练仍走 g4」红线**（长跑 26h 可靠性单列观察，首个正式
run 挂紧心跳即可）。训练口径性价比比 exp57 的 rollout 口径更惊人：10 核宿主训练 190 局/s
= **~700 局/s/$，g4 训练口径（~$2.25/h、按 rollout 279 的 45% 折算 ≈126 局/s ≈56 局/s/$）的
~12×**。exp55-D 报价：15.4 场/s ⇒ **100k 半庄 ≈ 1.8h ≈ $0.50**（10 核宿主口径）。

**迁移 runbook 增补（正式发射脚本必须带）**：
1. pod 起来先读 `/sys/fs/cgroup/cpu.max`，保证核 <16 则 terminate re-roll（宿主异构实锤：
   同价拿到过 23.8 核与 10.2 核）；`--workers` 按保证核设定；
2. bootstrap 依赖：`pip install --break-system-packages numpy mahjong tensorboard`；
3. kill/重启训练必须附带**残留 worker 清场**（kill 主进程后 worker/infer server 会孤儿存活，
   握着共享内存与 GPU）；
4. ckpt 逃逸走 scp/rsync 回工作站（已验证）；GCS 直推未配凭证（**gap**，正式迁移时补
   HMAC 或 service-account key，或沿用工作站中转）；
5. 30 分钟静默自停保险（`kill 1`）+ 工作站侧分相心跳照旧。

## Artifacts
| Path | Size | Description |
|---|---|---|
| experiments/exp58_runpod_migration_prereg/logs/stage{1,2,2_resume,3}.log | 40KB | 四阶段完整训练日志 |
| scratchpad/exp58_games_final.pt | 8MB | 逃逸验证用 ckpt（会话结束可弃） |
