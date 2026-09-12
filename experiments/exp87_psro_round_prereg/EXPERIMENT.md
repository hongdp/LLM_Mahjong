# exp87 完整 PSRO 一轮：对着冻结的 M 训最佳响应剥削者（×2），再让 M 对着剥削者人口训（2026-09-12）

- **Date**: 2026-09-12 PDT  **Status**: running
- **Git**: 6322f5c（Rust 引擎 + 联赛路由 + `--houjuu_extra`；uncommitted: none）
- **Env**: 社区 RTX 3090 Ti / 3090（$0.22–0.27/h）；纯血谱系；`train_dnn_ppo.py --engine rust --arch cnn_m_v3r`
- **起点**：M = exp85_M（vs bc49 0.4461，vs E 0.534），E = exp85_E（防守者，vs bc49 0.429），W = exp81_W

## Purpose & Hypothesis
exp85 的剥削者 E 是在**镜像**里加放铳惩罚训出来的，没有针对主体做最佳响应：主体对 E 的胜份 0.50→0.51，M vs E（T=0）0.534，即池子里的对手弱于主体，主体没有"必须改变"的压力。exp82/85 一起说明"单成员、frac 0.5、非最佳响应"的人口机制不移动均衡。
真正的 PSRO 步是：(A) 对着**冻结的** M 训最佳响应 BR(M)；(B) 让 M 对着 {M, E, BR₁, BR₂} 的均匀混合（frac 1.0）续训。假说：
- **H-A（可剥削性）**：M 存在同谱系可学的弱点（对立直不弃和、放铳 0.15、南场不收手），8M 局的 BR 能把对 M 的胜份推到 ≥0.53（T=0，2v2 配对 4k 对）。若两个 BR 都 <0.52 ⇒ M 在本人口可达范围内近似不可剥削 ⇒ PSRO 无梯度，本轮在阶段 A 后停止（省下阶段 B）。
- **H-B（主体响应）**：M 在含真实剥削者的池中续训后，vs bc49 ≥ **0.46**（20k 对，+1.4 pp，z≈2.8）且行为迁移（默听 >3% 或撞立直 <0.40 或南场领先放铳 <0.15）。**H0**：≤0.45 且无行为迁移 ⇒ 完整 PSRO 一轮也不动，纯血线训练侧关闭。
- 反解释：BR 若只是"消极不进攻"则对 M 胜份不会 >0.52（消极者 vs 主体 ≈0.47，exp85 E vs W 0.466）；阶段 B 的 M2 若 vs bc49 掉但 vs 池升 ⇒ 过拟合人口（记录，不算正）。

## Method
- **阶段 A（两台 pod 并行，各 8M 局）**：`--league pool87A.json`（{M}）`--league_frac 1.0 --league_learner_seats 1 --league_opp_temp 0`（对手贪心 = 评测形态），lr 3e-5，熵 0.01，塑形关（`--shaping_schedule 0:1.0,400000:0.0`，已过 400k）。
  - BR₁：从 W 起，`houjuu_extra 0`（纯最佳响应）。games 64M→72M，milestone 68M。
  - BR₂：从 E 起，`houjuu_extra −8`（防守偏置最佳响应）。games 80M→88M，milestone 84M。
  - 在轨每 4M：ckpt vs M（T=0，1,000 对）+ vs bc49 600 对 + 风格探针。学习者对池胜份 `league_stats.jsonl` 每迭代。
  - 门 A：max(BR vs M) ≥ 0.53（4k 对，两段 67M/68M 种子）。
- **阶段 B（一台 pod，24M 局）**：M 在池 {M, E, BR₁, BR₂} 中续训，`--league_frac 1.0 --league_learner_seats 1 --league_opp_temp 1.0`（T=1 对手保持与 exp85 一致的采样多样性）。games 80M→104M，milestones 88/96M。学习者席 1/4 ⇒ 24M 局 = 24M 学习者席局（exp85 为 48M）。
  - 在轨每 8M：vs bc49 600 对 + 和牌类型探针（默听）+ 学习者对各池成员胜份。
  - 终评：M2 vs bc49 20k 对、vs M 20k 对、vs BR₁/BR₂ 4k 对、半庄 n=1,200（seed 59300000）、风格/和牌类型/半庄行为探针。
- **第二轮**（条件）：仅当阶段 B 出现 +1 pp（≥0.456）或行为迁移，才预注册 exp88 复用同一流程（BR₃/BR₄ 对 M2）。

## Config
同 exp85（P3 + 退火塑形），差异：`--league_frac 1.0 --league_learner_seats 1`；阶段 A `--league_opp_temp 0`；`--games_per_iter 2048 --dup_k 8 --batch 4096`。seeds：BR₁ 11，BR₂ 12，M2 13。

## Success Criteria（预注册）
1. 门 A：max(BR₁, BR₂) vs M ≥ 0.53（4k 对 T=0）；否则 H0-A，停止本轮。
2. H-B：M2 vs bc49 ≥ 0.46（20k 对）且行为迁移之一成立。≥0.50 = 总目标。
- **预算**：阶段 A 2×(8M/≈900 局/s ≈ 2.5 h + 开机 0.5 h) ≈ 2×$0.8 = $1.6；阶段 B 24M ≈ 7.5 h ≈ $2.0 + 开机 $0.15；合计 ≈ **$3.8，上限 $7**。累计已用（09-07 起，billing API）≈ $67，剩 ≈ $133。

## Progress
- [09-12 11:10 PDT] 社区 3090/3090Ti/4090 全部无货或不可用（4090 宿主 64411a5a 无公网 TCP 口，且账号未注册 SSH 密钥 ⇒ 代理 SSH 不可用；两次试机 ≈$0.03，已终止）。改用 **Secure RTX 2000 Ada**（16GB，$0.24/h）×2：
  BR1 `lszhp6ts2wvzmu`（EUR-IS-1，8 vCPU，157.157.221.29:57102），BR2 `3s3noy34tubywz`（EU-RO-1，6 vCPU，213.173.110.197:20322）。vCPU 少，吞吐预计 250–400 局/s（exp85 3090/32 vCPU 为 925）。
  GPU 守卫过（cu12.8）；start.pt / pool/M.pt md5 与本地一致；bootstrap（rustup + maturin）启动中。

- [09-12 11:18 PDT] **阶段 A 发射**：BR1（W 起，houjuu 0）**≈500 局/s**，BR2（E 起，houjuu −8）**≈810 局/s**——学习者席只占 1/4，更新相很轻，吞吐比预期高。8M 局 ≈4.5 h / 2.7 h，阶段 A ≈$1.8。
  首迭代：熵 0.55/0.65，KL 0.001–0.003，EV 0.11 / 0.03–0.07（BR2 critic 在适应新分布），显存 1.2 GB（GPU 在用）。心跳 ×2 / 拉取+GCS（`gs://llm-mahjong-experiments/exp87_BR1|BR2`）/ 在轨每 4M（vs M 1,000 对 + vs bc49 500 对）/ TB `exp87_BR1_LIVE`、`exp87_BR2_LIVE`。
  教训重犯一次：TB 重启用 pgrep 模式含目标字串把自己杀了（exit 144）——改用已知 PID 单独 kill 后恢复。

## Results

## Conclusion

## Next Steps

## Artifacts
