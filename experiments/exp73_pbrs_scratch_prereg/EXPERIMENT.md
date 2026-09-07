# exp73 引擎势函数塑形（PBRS）从零（纯血线 $200 计划阶段 B-2 修订项，2026-09-07）

- **Date**: 2026-09-07  **Status**: running（实现 + 冒烟 + 发射）
- **Git**: 见 Progress；RunPod Secure L40S 单 pod 双臂；纯血：势函数只用引擎向听数（零人类数据、零外部模型）
- **Env**: `train_dnn_ppo.py --shaping --shaping_scale`（本实验新增旗标；`selfplay.potential` 改为表内缓存向听，0.58 ms/步）

## Purpose & Hypothesis
exp69/70/71/72 依次排除了感知、目标、估计器、归因四个瓶颈；exp72 的诊断把纯血线对 bc49 的差距定在**进攻效率**（bc49 桌上和牌 13% vs 22.5%）。
从零 PPO 的终局结算奖励对"手牌效率"没有逐步梯度：一次好切牌的价值要经过十几步、四家运气才在结算里出现。
势函数塑形 F_t = γΦ(s_{t+1}) − Φ(s_t)，Φ = −2×向听数（引擎自算、纯度不变），按 Ng et al. 1999 不改变最优策略，却给每一步切牌一个即时的效率信号。
LLM 时代 exp1/exp2 在 SFT 先验上测过 PBRS（无差异，64 局 arena，CI ±1,700 分）；**DNN 纯血从零从未测过**，而 `apply_shaping` 的注释本身写着这正是它应发挥作用的场景。
**假设 H1**：S1（scale 1.0）或 S2（scale 0.3）同预算 T=0 对 P3 ≥ 0.53；机制读数：和牌率、平均和牌巡目、向听下降速度优于 P3。
**反假设 H0**：≤0.50——稠密效率信号对 1M 局尺度的 PPO 没有增益（sparse 结算已足够学到牌效），或塑形把策略推向"只做向听不做打点/不做防守"的偏置。

## Method
- 配方 = P3 钉死旗标（`--ppo_epochs 1 --adv_clamp 5.0 --target_kl 0.03 --batch 4096 --lr 1e-4 --entropy_schedule 0:0.03,600000:0.01 --gamma 0.995 --gae_lambda 0.95 --dup_k 8 --games_per_iter 2048`），cnn_m_r，随机初始化，100 万局。
- 塑形：每步 φ = −2×最佳可达向听（14 张取各弃牌后的最小向听；13 张取手牌向听），`apply_shaping(steps, γ, scale)`，终局 Φ:=0；结算奖励不变。
  组基线（dup_k 8 同牌山）自动抵消 −Φ(s0) 常数。S1 scale 1.0（一向听 ≈ 2,000 分等价），S2 scale 0.3。
- 势函数实现改为表内 memoised 向听（原版含受入枚举 7.6 ms/步 → rollout 17×，不可用；新版 0.58 ms/步，向听一致率 1.0，rollout +34%）。
- 对照：P3（同配方同预算）+ exp27A/G3 独立读数；噪声地板 ±0.5–0.9%。
- 终评（工作站）：S1/S2 vs P3、exp27A、bc49 双段 20k 对；S1 vs S2；防守探针 1,600 局；风格向量 @bc49 桌（和牌率/巡目/打点）。

## Success Criteria（预注册）
1. **主判据**：S1 或 S2 vs P3 合并 share ≥ **0.53**（z≥8）→ 杠杆成立，进入阶段 C；0.50–0.53 中性；<0.50 判负（塑形有害）。
2. 机制：@bc49 桌和牌率 ≥ P3 + 2pp 或 平均和牌巡目 ≤ P3 − 0.3 巡（风格向量）。
3. 在轨：每 ckpt 对同局数 P3 n=600；任一臂连续三点 <0.40 ⇒ 早停该臂。
- 预算：双臂同 pod ≈ 6–7 h ≈ **$7–8，上限 $10**。

## Progress
- [09-07 05:55 PDT] 实现 `--shaping/--shaping_scale`、`apply_shaping(scale)`、快速势函数；`tests/test_shaping_scale.py` 1/1、`test_cf_rollout.py` 2/2（断言放宽：九种九牌流局可少席位）。
  冒烟 512 局：rollout 4.7 s → 6.3 s（+34%），ret_std 0.9→2.3（含 −Φ(s0) 常数，组基线抵消）。

## Results

## Conclusion

## Next Steps

## Artifacts
