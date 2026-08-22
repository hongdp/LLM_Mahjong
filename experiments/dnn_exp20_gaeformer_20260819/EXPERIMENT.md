# exp20：GAE × ConvFormer 合体（正交增益可加性 + 教师参照线挑战）

- **Date**: 预注册 2026-08-19  **Status**: launching
- **Git**: ada70f3  **Env**: mahjong-dnn-c5（us-east1-c）on-demand g2
- **对照**: 同协议家族三基线——exp18-cnn 1013.4（无增益）/ exp17-C 1079.7（仅 GAE）/
  exp19-r2 1065.7（仅 ConvFormer）；教师参照线 bcrl14_600 = 1117.7

## Purpose & Hypothesis
exp17-C 与 exp19 分别以正交机制（信用分配 vs 表征）打破 1012 平台，且 700k 终点都仍在
爬升。假说：①两增益（+66 / +52）至少部分可加；②合体 + 拉长到 1.2M 可冲击并超越教师
参照线 1117.7——**若达成，即「纯自对弈超过教师先验系」的 AlphaZero 里程碑**。

## Method / Config
exp18 共享协议 + 三件套合体：`--arch convformer_m --gae_lambda 0.95 --warmup_updates 150`，
`--total_games 1200000`（熵台阶仍在 600k 降 0.01，与家族严格同相位；700k 里程碑供
同局数三方对表）。exp_dir `dnn_exp20_gaeformer_20260819`。

## Success Criteria（发射前定死）
1. **可加性判定（700k 里程碑）**：正式评分 ≥1110 ⇒ 近似可加；≈1080（CI 内含 exp17-C）⇒
   不可加（GAE 主导）；<1065 ⇒ 负交互（合体有害，同样定论）。辅以 vs exp17-C-700k
   200 副直接对打。
2. **冠军/里程碑判定（1.2M 终点）**：正式评分 + vs bcrl14_600 直接对打 200 副；
   **点差显著正或评分 >1117.7+2SE ⇒ AlphaZero 里程碑达成**。
3. 风格：立直率 ≥20%（GAE 风格效应在新表征下保留）。
4. 健康：吞吐 ≥10 局/s、KL/熵正常、跑满。

## Progress
- [2026-08-19] 预注册，c5 发射。
- [2026-08-21 23:20 复盘，运维事故] 8/19 首发射落在 c5 上 exp19-r2 刚结束、runner EXIT trap
  正在收尾的窗口；trap 的 `shutdown -h now` 在 ~2 分钟后把刚起步的 exp20 一并关掉（首 iter
  哨兵恰在关机前看到 1 个 iter，假阳性）。VM 随后关停两天（无闲置计费）。8/21 14:47 被重新
  启动并以**与预注册逐字一致**的配置从零重发（非本会话操作）。当前 run 即正式 run，
  8.3h 处 379k 局、12.7 局/s、Elo 370k@978，ETA 8/22 ~17:00。
  **教训**：①复用 VM 前必须等 TERMINATED（exp11 看守的条件，手动发射时被跳过）；
  ②完成哨兵只盯 games_final 对死 run 是盲的——已加 train_log 心跳告警（30 min 无更新即报）。

## Results
| Metric | This run | Baselines | Success criterion |
|---|---|---|---|
| **M1 可加性（700k 正式评分）** | **1066.7 ± 13.3** | GAE 1079.7 / ConvFormer 1065.7 / cnn 1013.4 | ≥1110 可加 ⇒ **未达；CI [1040,1093] 含 GAE ⇒ 预注册「不可加（GAE 主导）」分支** |
| 700k vs GAE-700k 对打（seed0=20260828） | +554 ± 1062（176:166）null | — | 与评分一致：同量级 |
| 饱和点防守探针 @700k | defense_iq **0.001**（fold 0.219/0.217/0.218）；曝露放铳 0.224；V 跌落 −2018/−1948/−2393 | 全谱系≈0 | 仍无条件折牌；critic 更怕了（V 跌幅史上最大），策略仍不折 |
| M2（1.2M 终点 vs 教师参照线） | 待跑满（84%，ladder 700k 后已爬到 1035-1097） | bcrl14_600 1117.7 | — |

### M1 解读（2026-08-22 12:50）
700k 处合体 ≈ ConvFormer 单臂，未叠上 GAE 的 +66——两增益**在 700k 不可加**。但 ladder 在 700k 后
继续爬（900k@1097），ConvFormer 的「慢热」特性（exp19 教训）意味着 700k 对它偏早；真正的
可加性裁决留给 1.2M 终点与 M2。防守：700k 处策略仍零折牌而 critic 的危险感知达史上最强
（V 跌 2000+ 点当量）——「知道危险但不改行为」的脱节进一步坐实；由于 700k 时 exp20 尚未饱和
（仍在爬升），梯度竞争 vs 生态均衡的裁决推迟到 1.2M 饱和点的探针。

## Conclusion
（待运行）

## Artifacts
| Path | Description |
|---|---|
| gs://llm-mahjong-experiments/dnn_exp20_gaeformer_20260819/ | 云端主目录 |
