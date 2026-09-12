# exp88 高熵 / QRE 正则续训：IIG 文献里唯一没测过的训练侧杠杆（草稿，未发射）

- **Date**: 2026-09-12（草稿）  **Status**: draft（等 exp87 门 A 判决后定稿发射）
- **Git**: 待填  **Env**: Secure/社区 24GB 或 RTX 2000 Ada；纯血谱系（锚 = 自家 M，不含人类数据）

## Purpose & Hypothesis
文献（Reevaluating PG for IIGs, ICLR 2026；MMD, Sokota 2022；ACH, ICLR 2022）一致指出：PPO 自对弈在不完全信息博弈里不收敛到 NE，但**熵系数取足够大（QRE 温度）并配 KL-to-magnet 正则**后，通用策略梯度 ≥ NFSP/PSRO/R-NaD。
我们 32M–96M 刻度只测过**调低**熵（exp80-C2 0.01→0.003，✗），从未测过调高；exp87 显示最佳响应对 M 完全不动（M 是当前优化器可达范围内的不动点），提示需要更强的探索/正则而非更多数据。
- **H1**：熵系数 ×3–×10 或 KL-to-magnet 续训 16M 局后 vs bc49 ≥ **0.46**（20k 对，+1.4 pp）且行为迁移（默听 >3% 或撞立直 <0.40）。
- **H0**：≤0.45 ⇒ QRE 正则在此刻度也不移动平台，纯血线训练侧最终关闭。

## Method（两臂，各 16M 局，从 M 80M 续）
- **Q1**：`--entropy_coef 0.03`（3×），lr 3e-5，镜像自对弈，其余同 exp85。
- **Q2**：`--entropy_coef 0.03 --bc_anchor M.pt --bc_kl_coef 0.1`（moving-magnet 的固定磁铁近似：KL(π‖π_M)），其余同 Q1。
- 熵值本身作为在轨监控：预期从 0.55 升到 0.8–1.0；若 Q1 熵 >1.2 且 vs bc49 在轨下跌 >3 pp ⇒ 早停。
- 终评：vs bc49 20k 对、vs M 20k 对、半庄 n=1,200、行为探针。

## Success Criteria（预注册）
1. H1 如上；2. 预算 ≈$2.5（两臂 16M @ 700–900 局/s），上限 $5。

## Progress
