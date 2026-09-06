# exp68 Oracle policy guiding（Suphx 式）从零：防守能否在纯血线涌现

- **Date**: 2026-09-05  **Status**: running（实现 + 冒烟）
- **Git**: 见 Progress；训练上 RunPod **Community**（纯血线无核心资产，允许；3090Ti $0.27 或 L40S community $0.79）
- **Env**: `train_dnn_ppo.py --arch cnn_m_ro --oracle_hide_schedule`；设计见 experiments/designs/design_pure_line_v2.md

## Purpose & Hypothesis
纯血冠军 exp27-A 的 defense_iq = 0.011（bc49 0.184），对 bc49 头对头 0.203：从零 RL 从未学会防守（design_pure_line_v2 §2）。
Suphx 的 oracle guiding 让策略在训练前期看见对手暗手/待张，先把"何时弃张"学成显式映射，再随 dropout 退火把它迁移到公开线索上。
**假设**：同 1.0M 局预算下，退火到全盲后的 G 臂 defense_iq ≥ 0.10 且对 P 臂（冠军配方原样重跑）头对头 ≥ 0.53；
**反假设**：退火后防守随隐藏信息一起消失（策略学到的是"看暗手"而非"读线索"），defense_iq 回到 ~0，头对头平。
**纯度声明**：隐藏平面是引擎状态（自对弈模拟器特权），不含任何人类知识；播放/评测/桥接路径永远为零（编码器默认盲，
只有训练 rollout 按概率置 `table.oracle_visible`），因此 `_ro` 检查点不可能在评测中作弊。这与 AlphaZero 的"完美信息"同性质，
按 memory 规则属"纯度分岔"，在此明示供用户否决。

## Method
- **G 臂**：`cnn_m_ro`（cnn_m_r + 8 平面：3 家暗手计数、3 家待张掩码、牌山构成、里宝指示牌），`--oracle_hide_schedule 0:0.0,600000:1.0`
  （隐藏概率 0→1 线性，60 万局后全盲），其余 = exp27-A 配方（PPO，lr 1e-4，熵 0.03→0.01@600k，dup_k 8，games_per_iter 2048→8192，
  gamma 0.995，gae 0.95，batch 2048，无联赛/无锚，1.0M 局）。
- **P 臂**：`cnn_m_r` 同配方同种子（同批控制，exp62 教训）。
- 里程碑 ckpt 每 10 万局；每臂 TB 记录 oracle_hide_p、熵、EV。
- 终评（工作站）：G vs P、G vs exp27-A、G vs bc49 各双牌山段（67M n=4000 + 68M n=16000）；defense_iq（800 局 T=1 与 T=0 各一次）；
  风格向量 vs bc49；梯子 Elo（纪元 6 池，13 锚 × 200 对）。
- 额外诊断：G 臂里程碑 ckpt 在 30/50/60/80/100 万局的 defense_iq 曲线（隐藏概率 0.5→1.0 期间防守是否保住）。

## Success Criteria（预注册）
1. **防守涌现**：G（全盲，final）defense_iq ≥ **0.10**（P 预期 ≈0.01）。
2. **强度**：G vs P 双段合并 ≥ **0.53**；G vs exp27-A ≥ 0.53 → 纯血新冠军候选（走纯血谱系加冕：LEADERBOARD + champion_model §8 纯血行）。
3. 中间档（defense_iq ≥0.10 但强度平）：防守学到了但被进攻退化抵消 → 下一步调退火长度/熵。
4. 反假设成立（defense_iq <0.03）：oracle guiding 在本设定下不迁移，记录里程碑曲线（看到隐藏时是否曾有防守）。
- 预算：两臂并行 1.0M 局各 ≈ 3h（L40S community $0.79/h）≈ **$2.5**；若 3090Ti community 可得则更低。上限 $6。

## Progress
- [09-05 21:30] 差距诊断（`experiments/probes/pure_vs_prior_style_vs_bc49.json`，各 600 局 T=0 坐 bc49 桌）：
  exp27-A 和牌 14.7% / **放铳 20.7%** / 立直 23.0% / 副露 23.0%；bc65 21.8% / 13.7% / 14.7% / 32.3%；bc49 22.5% / 13.7% / 16.8% / 29.0%；
  凤凰卓人类 21.2% / 12.5% / 18.2% / 33.8%。纯血冠军对 bc49 桌放铳率高 7pp、和牌率低 7pp；defense_iq 0.011 vs 0.184。
- [09-05 21:50] git 2671f2f：v1ro 编码器 / cnn_m_ro / `--oracle_hide_schedule`；测试 20/20；冒烟 768 局 hide_p 0→0.5→1.0 生效；
  向量化 rollout 验证：hide 1.0 → 0/2865 步含 oracle；0.0 → 2890/2890；0.5 → 17/32 局（按局抽签）。
- [09-05 19:30] 首发 Community L40S（24 vCPU，$0.79/h）：宿主驱动 CUDA 12.4，cu128 镜像下 `torch.cuda.is_available()=False`，
  两臂静默落到 CPU 推理（日志只有一行 `_cuda_getDeviceCount` 警告）→ terminate（~$0.15）。`allowedCudaVersions` 参数未改变分配结果
  （仍 12.4 宿主）→ 改用 `runpod/pytorch:2.4.0-py3.11-cuda12.4.1` 镜像重建（同一宿主），发射前加 `torch.cuda.is_available()` 守卫。
- [09-05 19:36] cu124 镜像在同一宿主上仍 `CUDA unknown error` → 宿主 GPU 本身坏/配置错，不是版本问题；Community 两次都分到同一台
  （60.249.37.148）→ 放弃 Community，改 **Secure L40S US-TX-4**（$1.09/h；纯血无资产，Secure 只是可靛性选择）。三次建机浪费 ≈ $0.3。
- [09-05 19:40] **发射** pod `l53wqg84un9454`：CUDA 守卫通过；G 79.6 局/s、P 85.4 局/s（8192 局/迭代 ≈ 1.7 分钟），显存 16GB，GPU 98%；
  起点熵 1.89（≈ 6.6 个等价动作），KL 0.008–0.009。预计 100 万局 ≈ 3.5h，两臂并行 ≈ **$4**。
- [09-05 21:44] **60 万局里程碑（G 刚退火到全盲）防守探针**（800 局 T=0 镜像自对弈）：G600 defense_iq **0.021**（fold 0.226/0.209/0.205）、
  P600 **−0.039**（0.175/0.211/0.214）、exp27A（同批复测）0.055；暴露步数 G/P 只有 ~630 vs exp27A 8,300——60 万局的自对弈里立直还很少，
  防守读数噪声大。早读：G 与 P 无差别，等 100 万局终评。

- [09-05 22:50] **两臂完成**：各 123 迭代 / 1,007,616 局 / 175–185 分钟（88–94 局/s）。pod terminate（204），≈3.3h ≈ **$3.6**（+ 三次废机 $0.3）。
  轨迹：G 熵 1.89 → 1.06、迭代胜率 0.605；P 熵 1.90 → 1.17、胜率 0.557；G 的 hide_p 按计划 0 → 1.0@62 万局。

## Results（第 1 轮，games_per_iter 8192）

| 判据 | 目标 | 实测 | 判定 |
|---|---|---|---|
| 1 防守涌现：G defense_iq（800 局 T=0） | ≥ 0.10 | **G −0.002**、P 0.025；同批 exp27A 0.055、bc49 0.209 | ❌ 无涌现，G ≈ P |
| 2 强度：G vs P 双段合并 20k 对 | ≥ 0.53 | 67M 0.5178、68M 0.5141 → **0.5148±0.0035（z=+4.2）** | ❌ 未达 0.53，但显著为正（+1.5%） |
| 2 强度：G vs exp27A | ≥ 0.53 | **0.3733±0.0034**；P vs exp27A 0.3720 | ❌ 两臂都远弱于 exp27A |
| G/P vs bc49 | — | 0.2112 / 0.2078 | 与 exp27A 的 0.203 同层 |
| 暴露席放铳率（探针） | — | G 0.230、P 0.243、exp27A 0.156、bc49 0.126 | G 的 +1.5% 不来自防守 |

风格向量（坐 bc49 桌 600 局 T=0）：G 和牌 4.7% / 放铳 25.3% / 立直 2.0% / **副露 86.2%**；P 5.3% / 23.3% / 2.2% / 87.8%——
两臂都还在从零 RL 的"见牌就碰"早期阶段（exp27A 终局：14.7% / 20.7% / 23.0% / 23.0%）。

**配方混淆**：为吞吐把 `games_per_iter` 从 exp27A 的 2048 提到 8192 → 100 万局只有 123 次 PPO 迭代（exp27A 488 次），
两臂都处在学习曲线更早的位置（对 exp27A 0.37），因此"对 exp27A ≥0.53"判据在本轮不可判；A/B（G vs P）本身公平。

## 第 2 轮预注册（2026-09-05 23:00）：忠实 exp27-A 配方（games_per_iter 2048）
唯一改动 = `--games_per_iter 2048`（其余同第 1 轮），G2/P2 各 100 万局同批。判据不变（defense_iq ≥0.10；G2 vs P2 ≥0.53；
P2 vs exp27A 应 ≈0.50 作为复现检查，若 <0.45 则配方仍有差异需查）。预算 ≈ 4–5h ≈ $5，上限 $8。
- [09-06 01:45] 第 2 轮 pod `8xdqwz6ght9m4v`（CUDA 13.0），G2 65–73 局/s、P2 74 局/s。60 万局探针：G2 defense_iq **−0.036**、P2 −0.032
  （暴露席放铳率 G2 0.192 vs P2 0.226）；两臂 66–68 万局时熵 1.25、迭代胜率 39–42%。

## Conclusion（第 1 轮）
Oracle policy guiding 在同预算下给了**小而显著的强度增益（+1.5% share，z=4.2）**，但**没有让防守涌现**：全盲后 defense_iq 回到 0，
暴露席放铳率与对照臂相同。也就是说，退火期间策略学到的"看暗手弃张"没有迁移成"看公开线索弃张"；留下的增益更像是
更快的进攻学习（G 熵更低、迭代胜率更高）。第 1 轮因 games_per_iter 的混淆无法回答"是否越过 exp27A"，第 2 轮修正。

## Next Steps
- 第 2 轮（进行中）。若 G2 仍无防守：oracle guiding 作为"加速器"保留（+1.5%），防守涌现需要别的机制
  （候选：对手待张作为**辅助预测目标**而非输入——策略从公开线索预测隐藏待张的辅助头，纯、可迁移；或半庄级顺位奖励）。

## Artifacts
| Path | Size | Description |
|---|---|---|
| experiments/exp68_{G,P}/ + gs://llm-mahjong-experiments/exp68_{G,P}/ | 12 ckpt ×~2MB 各 | 里程碑 ckpt、latest.pt、train_log.json、TB |
| experiments/exp68_pull/*.log | — | pod 日志 |
| experiments/probes/exp68_arms.json、exp68_defense_600k.json、exp68_defense_final.json、exp68_style_vs_bc49.json、pure_vs_prior_style_vs_bc49.json | — | 终评、防守探针、风格向量 |
| src/agents/dnn/encoder.py（v1ro）、arch_zoo `cnn_m_ro`、train_dnn_ppo `--oracle_hide_schedule`、tests/test_oracle_guiding.py | — | 基建 |
