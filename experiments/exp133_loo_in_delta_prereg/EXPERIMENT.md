# exp133：副本留一基线进入优势（配对牌山臂 P），从零到 32M（2026-11-08）

- **Date**: 2026-11-08 14:00 PST  **Status**: done（2026-11-08 14:52 PST 起训；慢机 16:00 自行恢复；2026-11-09 18:06 PST 完成，pod 已自毁）
- **Git**: 代码 perf 分支 `claude/perf-rollout-overlap` 43c6a28（`--loo_in_delta`；`apply_group_baseline` 存 `loo_d`），运维 `experiments/exp133_ops/`（由 exp127_ops 派生，`repo.tgz` = perf 43c6a28 的 git archive）
- **Env**: RunPod Secure RTX 3090（或 A40，≤ $0.59/h），仅 22/tcp，pod 上无凭据；完成后自毁
- **Approval**: 本会话用户 11-08"都开"（exp132 + exp133，合计约 $30）；论文会话转达其用户"跑"
- **Budget**: ≈ 32M ÷ 300 局/s ≈ 30 h × $0.50 ≈ **$15**，上限 **$20**；超过 1.3× 预估时长报用户

## Purpose & Hypothesis
论文 RQ3"纯自对弈需要什么、能否学得更快"。现行配方（exp124）里，同一牌山同一座位 8 个副本的留一基线 b 只进了 value 目标，GAE 的 δ_t 用原始 r_t，所以 8 份副本对策略梯度不降方差（论文 §Training 的 \pending）。本臂把 δ_t 改成用 r'_t = r_t − b/n，其余全同。
- 已知先验（exp131 阶段 0，论文审阅会话）：b 项占优势方差 ≈13%、符号翻转 17–18%，但修正后优势方差只降 ≈4%、PG 梯度方差比 1.02–1.03、均值梯度余弦 0.96–0.99 ⇒ **预期 H0**。

## Method
- 唯一改动：`--loo_in_delta`。其余 = exp124 A 第一段配方（与 exp127 A 臂相同的 perf 代码与同步调度）：`convformer_m_v3r_m46`、lr 1e-4、seed 1241、warmup 3000、batch 4096、1 epoch、dup_k 8、γ 0.995、λ 0.95、entropy 0.03、target_kl 0.03、adv_clamp 5、shaping 0→400k、`--rollout_values --fast_trunk --compile --bf16_rollout`，0 → 32M，里程碑 8 / 16 / 24 / 32M。
- **对照**：exp124 A 32M（0.4810，同主干同配方同 seed；注意 exp124 A 用的是 exp124 时的代码，本臂用 perf 分支——exp127 A 臂同样如此）。
- **主判据**：32M 对 bc49 份额，标准 10,500 去重判决牌山（101112000 起 1 万 + 101112500 段末 500），与 exp124 A 32M 逐牌山配对。
- **次要**：8 / 16 / 24 / 32M 在前 2000 牌山上的曲线（与 exp124 A 同点配对），以及到达 0.48 所需局数。

## Success Criteria（发射前写定）
Δ = 臂 − 对照（同牌山配对）：**Δ ≥ +1.5 pp ⇒ 加速器**；**Δ ≤ −1.5 pp ⇒ 有害**；其余 ⇒ **H0**。
1.5 pp ≈ 配对牌山差 3 SE；**不含种子方差**（单种子）——判读时注明。

## Progress
- [2026-11-08 14:00 PST] 预注册。代码：单测（`tests/test_loo_in_delta.py`：以 r − b/n 折现求和逐步复现减基线后的 returns）通过；真实 Rust rollout 512 局全部带 `loo_d`（SD 0.21，与 exp131 测得的 b/n SD 0.19–0.20 一致）；本机 ConvFormer46 1,024 局训练冒烟通过。

- [2026-11-08 15:05 PST] **发射**：pod `c30kwr09vn1ag2`（exp133-P），Secure RTX 3090 ×1、32 vCPU（cgroup v1，31 核）、EU-CZ-1、**$0.50/h**，仅 22/tcp，SSH 213.192.2.92:40112，pod 上无凭据。编译门 / 臂门通过；22:52 UTC 从 0 局起训，标志 `--rollout_values --ckpt_every 400 --loo_in_delta`。**慢机迹象**：前 52 次迭代每次 ≈7.3 s（≈280 局/s），之后 `concat_h2d`（回合数组拼接 + 拷到 GPU）从 0.6–0.7 s 跳到 4–15 s，每次迭代 11–22 s（≈100–180 局/s）；臂 O 同代码同阶段 concat_h2d 0.15 s ⇒ 是这台主机的问题（内存压力 PSI ≈4–5%，与其他租户共享的页缓存），不是 `--loo_in_delta`（GAE 一步 0.012 s）。按规则不擅自动手，已报用户。

- [2026-11-08 15:10 PST] 用户决定"先观察一小时"再定是否换机；16:11 复查（每轮耗时、concat_h2d、内存压力、按当前速度的总时长与花费）。

- [2026-11-08 16:12 PST] **一小时复查：已恢复**。第 53–288 轮慢（每轮 10–22 s，concat_h2d 4–15 s）；**第 289 轮（≈59 万局，约 16:00）起回到每轮 ≈6.8–7.1 s、concat_h2d 0.13 s**，与臂 O（6.7 s / 0.13 s）相同。主机内存压力 PSI avg10 从 ≈4–5% 降到 0.29%。当前 76.8 万局；过去一小时平均 158 局/s（含慢段），现速 ≈300 局/s ⇒ 余下 ≈29 h，总 ≈30.5 h ≈ **$15.3**（上限 $20 内）。保留本 pod，继续观察。

- [2026-11-08 16:30 PST] 用户确认"保留继续跑"（不换机）。

- [2026-11-08 22:25 PST] 8M 里程碑（md5 与 pod 一致）：2,000 牌山（101112000 起）份额 0.4438 ± 0.0104，同牌山对 exp124 A 8M（0.4413）配对 **+0.25 ± 1.38 pp**，在轨读数、不作判决（判决在 32M）。吞吐 ~300 局/s、每轮 ~6.0 s。

- [2026-11-09 05:30 PST] 16M 里程碑（md5 与 pod 一致）：2,000 牌山份额 0.4598 ± 0.0104，同牌山对 exp124 A 16M（0.4353）配对 **+2.45 ± 1.35 pp**（z ≈ 1.8），在轨读数、不作判决。基线 exp124 A 在 16M 处偏低（低于它自己的 8M 0.4413），exp132 同点也是 +2.97，所以差值有相当部分来自基线下凹；判决看 32M。

- [2026-11-09 12:00 PST] 24M 里程碑（md5 与 pod 一致）：2,000 牌山份额 0.4800 ± 0.0104，同牌山对 exp124 A 24M（0.4708）配对 **+0.92 ± 1.43 pp**，在轨读数。三点 +0.25 / +2.45 / +0.92（16M 受基线下凹影响）；判决在 32M。

- [2026-11-09 18:35 PST] **32M 完成并判决**。18:06 心跳 `✅ DONE md5 verified`（games_final.pt f0a0d0df…，games_32000000.pt 9b123fe7…），PULLED_OK 已发；18:30 `get-pod c30kwr09vn1ag2` 返回 404（pod 已自毁）。本地 pull / eval 循环自行退出，孤儿 watchdog 按 PID 结束。墙钟 27 h 14 min × $0.50/h ≈ **$13.6**（上限 $20 内；按时长估算，未核对账单）。

## Results
判决：T=0 单局复式 2v2 对 bc49，标准 10,500 去重判决牌山，与 exp124 A 32M 逐牌山配对。

| 指标 | 臂 P（LOO 进 δ） | 对照 exp124 A 32M | 配对差 | 预注册线 |
|---|---|---|---|---|
| 对 bc49 份额（32M） | 0.4840 ± 0.0045 | 0.4810 | **+0.31 ± 0.61 pp（z 0.5）** | ±1.5 pp |
| 点差 / 牌山 | — | — | +186 ± 178 | — |

在轨曲线（前 2,000 牌山，同点配对）：8M +0.25 ± 1.38、16M +2.45 ± 1.35（基线 16M 下凹）、24M +0.92 ± 1.43 pp。到达 0.48：臂 P 24M 读数 0.4800 ± 0.0104（2,000 牌山，恰在线上）、32M 0.4840；对照 24M 0.4708、32M 0.4810——在噪声内没有可分辨的提前。

## Conclusion
- **按预注册判读：H0**。把副本留一基线放进 GAE δ（r − b/n）对 32M 强度没有可测影响（+0.31 ± 0.61 pp），与 exp131 阶段 0 的先验预期一致。
- **范围戳**：架构 ConvFormer46（exp124 A 配方）；信号源 = 纯血自对弈 PPO、单局点差目标，唯一改动 `--loo_in_delta`；规模 0–32M；评测 T=0 单局复式对 bc49、10,500 牌山；**单种子**；对照 exp124 A 用 exp124 时的代码、本臂用 perf 分支；前 289 轮在慢机上（只影响墙钟）。失效条件：更大规模（方差缩减的收益可能随规模变化）、其他主干、半庄目标未测。
- 纯血线下一步：降方差这一类在 ConvFormer46 0–32M 上再记一次 H0；主线看 Stage 2 与 XL 续训。

## Artifacts
| Path | Description |
|---|---|
| `experiments/exp133_P/games_{0,8000000,16000000,24000000,32000000}.pt`、`games_final.pt`、`milestones.md5`（主 checkout） | 里程碑检查点（各 ≈24 MB）与 md5；`games_final.pt` = 32M 终点 |
| `experiments/exp133_P/config.json`、`train_log.json`、`yaku_counts.json`、`tensorboard/` | 配置快照、逐迭代日志、役种计数、TB（面板 `_tb_qlearning/exp133_P_looindelta_0-32M_LIVE`） |
| `experiments/probes/pairdiff_exp133_P_n10000_games_32000000.pt_{101112000,101112500}.json` | 32M 判决逐牌山 share / diff |
| `experiments/probes/pairdiff_exp133_P_n2000_games_{8,16,24}000000.pt_101112000.json` | 在轨读数逐牌山 |
| `experiments/probes/pairdiff_exp124_A_n10000_games_32000000.pt_{101112000,101112500}.json` | 对照逐牌山 |
| `experiments/exp133_pull_P/`（heartbeat / autoeval / train_P.log / exp.log / watchdog.log） | 运维日志 |
| `experiments/exp133_ops/` | 发射与监控脚本 |
