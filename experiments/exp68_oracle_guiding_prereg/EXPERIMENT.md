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

## Results

## Conclusion

## Next Steps

## Artifacts
