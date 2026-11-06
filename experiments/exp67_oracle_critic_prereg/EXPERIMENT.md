# exp67 Oracle critic：让 critic 看见隐藏信息，PPO 从 bc65 起步能否第一次越过 BC 水位

- **Date**: 2026-09-05  **Status**: done（判负：oracle critic 零增益，原因已量化）
- **Git**: 见 Progress；训练上 RunPod Secure（冠军谱系 bc65）；评测工作站双牌山段
- **Env**: `train_dnn_ppo.py --critic_feats oracle --arch convformer_m_v3r_m46_oc`（本实验新增）；引擎指纹不变（规则审计见 docs/engine_known_issues.md 2026-09-05）

## Purpose & Hypothesis
文献对照（docs/literature_review_2026-09.md §2/§4）：所有在麻将里越过模仿水位的系统都让 critic 看见隐藏状态
（Suphx oracle guiding、RVR 相对价值网 +4.9% 胜率、PerfectDou 完美信息 critic），并用 GRP/配对复式压方差。
本项目 exp46/59/60 的 RL 全部"能学不毁值不超越"，exp66 量出病根：单局回报 σ≈3,700 分，动作价值差几十到几百分，
(σ/Δ)² ≈ 10²–10³。**假设**：把对手暗手、牌山构成、隐藏宝牌/里宝喂给 value head（策略不看），critic 的解释方差大幅上升、
优势估计噪声下降，PPO（bc65 init + KL 锚）在同样局数下对同批"盲 critic"控制臂显著为正，并首次对 bc65 本体为正。
**反假设**：方差瓶颈不在 critic 而在策略梯度本身（每决策 1 个样本），oracle critic 只让 value loss 好看，强度仍平。

## Method
1. **oracle 特征（critic 专用，319 维）**：三家对手暗手计数、赤五、听牌待张、向听；牌山 34 类计数与长度；下一张杠宝牌指示牌、
   首张里宝牌指示牌（`src/agents/dnn/oracle_features.py`）。策略 logits 路径与 bc65 逐位一致（测试保证）。
2. **网络**：`convformer_m_v3r_m46_oc` = 同 trunk/头 + value 路径拼接 oracle 投影（policy 键从 bc65 加载，value 头重新初始化）。
3. **训练配方**（沿用 exp46I 的最强 RL 配方，唯一变量 = critic 信息）：PPO，`--init bc65 --bc_anchor bc65 --bc_kl_coef 0.3
   --value_detach --entropy_coef 0 --dup_k 8`（配对牌山组基线 = 控制变量）、`--value_warmup`（先只训 critic）、league 对手池含 bc65/bc49。
   - **O 臂**：`--critic_feats oracle`；**C 臂**：`--critic_feats none`（同批、同种子、同局数）。
4. **诊断指标**（TB）：value loss、critic 解释方差 EV = 1 − Var(R−V)/Var(R)（两臂对比，O 应显著高）、优势方差、KL 到锚、熵。
5. **终评**（工作站）：O vs C、O vs bc65、C vs bc65，各两个牌山段（67M n=4000 + 68M n=16000，若 z>2 再补到 50k/段）；A/A。

## Config（预注册）
- 局数：与 exp46I 同量级 1.0M 局/臂（RunPod L40S 13 核演员+学习者同 pod ≈ 65 局/s → 两臂并行 ≈ 8.5h ≈ **$9–10**，上限 $15）。
  先本机冒烟（2k 局）验证管线与 EV 指标。
- value_warmup：前 20% 更新只训 critic。

## Success Criteria（预注册）
1. **诊断**：O 臂 critic EV ≥ C 臂 + 0.2（否则 oracle 特征没被用上，后续结论无效）。
2. **主判据**：O vs C 双段合并 ≥ 0.51（z ≥ 2.5 在 20k 对 SE 0.0035 下 = 0.509）**且** O vs bc65 ≥ 0.51 → 首次 RL 越过 BC 水位，
   下一步扩局数 + 演员多 pod（exp60 设计）。
3. 0.50–0.51：critic 方差不是瓶颈；记录 EV 增益与强度零增益的组合 → 病根转向策略梯度样本数（每决策 1 样本），
   剩余方案只有 exp60 式大规模演员或搜索（需 oracle critic 本身作为价值网 —— 本实验产物可复用）。
4. < 0.495：RL 毁值，检查锚系数。

## Progress
- [09-05 13:30] 规则审计完成（docs/engine_known_issues.md 2026-09-05）：引擎零改动，指纹不变，无需重校。bc65 复制为 `experiments/_anchors_epoch6/bc65.pt`。
- [09-05 14:10] git 53a0679：oracle 特征 / `_oc` 网络 / 训练器支持；测试 2/2；本机冒烟（768 局，batch 1024 因 16GB 显存）：
  EV −0.000 → 0.006 → 0.012 三迭代上升，KL 到锚 0.001–0.004，策略 logits 与 bc65 逐位一致（测试保证）。
- [09-05 14:20] 发射准备：L40S Secure US-TX-4，两臂并行（各 6 worker，`games_per_worker 32`，gpu_infer），配方 = exp46I
  （lr 6e-5、warmup 150、KL 锚 0.3、value_detach、熵 0、dup_k 8、league {bc65,bc49} frac 1.0 学习席 1、对手 T=0）+ `--value_warmup 300`。
  预计每臂 ~30 局/s → 100 万局 ≈ 9.3h，两臂并行 ≈ $10.5（上限 $15）；里程碑 ckpt 每 10 万局。
- [09-05 13:28] pod `jkw919o635e5rl`，首发 batch 4096：两臂各占 21GB，L40S 44GB 打满，O 臂推理服务器重启时 OOM（try1 保留）。
  改 **batch 2048**（两臂同改，A/B 仍公平）重发：两臂各 **90 局/s**（比预估快 3×，GPU 推理服务器承担了演员），
  每迭代 8192 局 1.5 分钟 → 100 万局 ≈ **3.1h**，两臂并行 ≈ **$3.5**；显存合计 24GB。
  重发过程中远程 `pkill -f` 再次自匹配杀掉 ssh 会话（SKILLS 老坑第二次），改为"先列 pid、再单独 kill"。
- [09-05 17:42] **两臂完成**：各 123 迭代 / 1,007,616 局 / 244 分钟，后段 67 局/s。pod terminate（204），≈4.3h ≈ **$4.7**。
  **判据 1（诊断）不满足**：末 20 迭代均值 EV O **0.102** vs C **0.107**，value_loss 16.24 vs 16.57，KL 到锚 0.0124 vs 0.0126，
  熵 0.440 vs 0.428——oracle 特征没有带来任何可见的解释方差增益。逐段 EV 曲线两臂重合（0.09–0.12）。
- [09-05 17:50] 终评发射（O/C latest.pt 双段）；追加"oracle 信息量"离线探针（见 Results 后补）以区分"特征无信息"与"critic 没学会用"。

## Results

| 判据 | 目标 | 实测 | 判定 |
|---|---|---|---|
| 1 诊断：critic EV（末 20 迭代均值） | O ≥ C + 0.2 | **O 0.102 vs C 0.107**（曲线全程重合 0.09–0.12） | ❌ oracle 特征零增益 |
| 2 主判据 O vs C，双段合并 20k 对 | ≥ 0.51 | 67M 0.4949、68M 0.5033 → **0.5016±0.0035** | ❌ 平 |
| 2 O vs bc65 | ≥ 0.51 | 67M 0.5044、68M 0.5049 → **0.5048±0.0035（+1.4σ）** | 平（噪声地板内） |
| C vs bc65 | — | 67M 0.5105、68M 0.5017 → 0.5035±0.0035 | 平 |
| A/A | — | 0.500 | ✅ |
| 训练轨迹 | — | 两臂 KL 到锚 0.0125、熵 0.44→0.43、pg −0.007；100 万局 244 分钟/臂 | 与 exp46I 同形 |

**追加探针：隐藏状态对本局回报的解释力**（`scripts/oracle_info_probe.py`，bc65 T=1 自对弈 2048 局 = 125,673 步，按局切分 80/20，
紧凑特征岭回归 / 小 MLP 的留出 R²；回报 std 5.15 ≈ 5,150 分）：

| 决策位置 | 公开标量（20 维） | 公开 + 隐藏摘要（对手待张数/听牌/向听/牌山，30 维） | 仅隐藏（10 维） |
|---|---|---|---|
| 全部步 | 5.3% | **6.0%（+0.7pp）** | 1.4% |
| 最后 5 步 | 8.4% | **9.6%（+1.2pp；MLP 9.1→10.7%）** | 2.8% |
| 最后 15 步 | 5.5% | 6.3% | 1.5% |
| 前 15 步 | 4.1% | 3.8% | ≈0 |

（2,000 维 one-hot 平面 + 12 万样本的岭回归/MLP 全部过拟合到 R²≤0，不作证据。）

**第三牌山段 n=50,000 对补评（seed 69M）**：O vs bc65 **0.4995±0.0022**；C vs bc65 0.5042±0.0022；O vs C **0.4969±0.0022**。
三段合并 70k 对：O vs bc65 **0.5010±0.0019**、C vs bc65 0.5040±0.0019（+2.1σ）、O vs C **0.4982±0.0019**。

## Conclusion
**Oracle critic 判负，且给出了原因的直接测量。**
1. 让 critic 看见对手暗手、待张、牌山构成、隐藏宝牌，训练 100 万局后 critic 解释方差与盲 critic 完全一样（0.102 vs 0.107），
   策略强度也一样（O vs C 70k 对 0.4982±0.0019，略负）。
2. 离线探针解释了为什么：在决策点上，**即使知道全部隐藏牌，本局回报的可解释方差也只从 5.3% 升到 6.0%，最后 5 步从 8.4% 到 9.6%**。
   本局结局 ≥90% 由尚未发生的摸牌顺序决定，这部分对任何 critic 都不可见。文献里 oracle critic 的收益（RVR +4.9%、PerfectDou +0.04）
   来自从零训练的样本效率，不是从 Mortal 水位再往上的信噪比。
3. 因此 exp59 的病根 (σ/Δ)² 是**内生的**：σ≈5,000 分/局里 ~4,700 分是摸牌顺序，Δ 是几十到几百分。exp46/59/60/67 四条 RL 线都平在 BC 水位，
   不是实现或算法问题。
4. RL 从 bc65 起步 100 万局（两种 critic）仍为 0.500–0.504 vs bc65 —— bc65 = 目前这套设定（单局、终局奖励、配对牌山组基线、KL 锚）下 RL 的不动点。

**文献路线 ⑤（RL 前置条件）到此关闭**：oracle critic（本实验）、控制变量（dup_k 已在）、10⁵ 级评测（70k 对已达）三项都做了，
GRP 对单局设定不适用。剩余唯一未测的 RL 形态是 Suphx 式 **oracle policy guiding**（策略输入带隐藏信息 + dropout 退火的课程），
但按本实验的 R² 测量，其上限同样受摸牌顺序主导，且 Suphx 只给出了箱线图级证据；不建议在没有新信号源前投入。

## Next Steps
- 本方向不再追加实验。bc65 作为 BC 线终点（候选未加冕，见 exp65）；下一步的决策在用户：
  （a）加冕 bc65；（b）改变问题设定（半庄级 episode + GRP 顺位奖励，重做 reward 架构）；（c）接受"凤凰卓 BC ≈ Mortal 水位"为项目终态。
- `oracle_features.py` / `_oc` 网络 / `oracle_info_probe.py` 保留为基建（若做半庄级 GRP 训练仍可复用）。

## Artifacts
| Path | Size | Description |
|---|---|---|
| experiments/exp67_{O,C}/ + gs://llm-mahjong-experiments/exp67_{O,C}/ | 11 ckpt ×8MB 各 | 里程碑 ckpt（每 10 万局）、latest.pt（GCS 亦有）、train_log.json、league_stats、TB |
| experiments/exp67_pull/*.log、try1/ | — | pod 日志（含 batch 4096 OOM 的首发） |
| experiments/probes/exp67_arms.json、exp67_arms_n50000.json | 1KB | 终评（三段） |
| experiments/probes/exp67_oracle_info.json | 2KB | 隐藏状态解释力探针（分桶 R²） |
| src/agents/dnn/oracle_features.py、arch_zoo `convformer_m_v3r_m46_oc`、scripts/oracle_info_probe.py、tests/test_oracle_critic.py | — | 基建 |
