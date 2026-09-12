# exp88 高熵 / QRE 正则续训：IIG 文献里唯一没测过的训练侧杠杆

- **Date**: 2026-09-12（草稿）  **Status**: running
- **Git**: 5692b36（uncommitted: none）  **Env**: Secure/社区 24GB 或 RTX 2000 Ada；纯血谱系（锚 = 自家 M，不含人类数据）

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
1. H1 如上；2. 预算：Secure RTX 2000 Ada（6–8 vCPU，$0.24/h）镜像自对弈预计 400–600 局/s ⇒ 每臂 16M ≈ 8 h ≈ $2；两臂 ≈**$4.5，上限 $7**。

## Progress
- [09-12 15:20 PDT] **Q2 发射**：Secure RTX 2000 Ada `s3bci45ayfkz4o`（EU-RO-1，6 vCPU，213.173.98.203:22989），从 M 80M 续（熵 0.03 + KL(π‖π_M) 0.1）。
  首迭代 **≈290 局/s**（镜像 4 学习者席 ⇒ 每局数据 4×，更新相在小 GPU/6 vCPU 上成瓶颈；exp87 BR 阶段 1 席时 930 局/s）；bc_kl 0.021–0.023（×0.1 ≈ 0.002，正则项很轻）、熵 0.53–0.56（起点，系数 3× 后应上行）、KL 0.002、EV 0.11。
  **成本修订（结果前）**：16M 局 ≈15 h ≈ $3.7/臂，两臂 ≈$7.4 > 上限 $7 ⇒ **判读点改为 92M 里程碑（+12M 局，≈11.5 h，≈$2.8/臂）**，96M 终点只在 92M 已显示 ≥+1 pp 时才等。心跳 / 拉取+GCS（`exp88_Q2`）/ 在轨每 4M（vs M 1,000 对 + vs bc49 500 对）/ TB `exp88_Q2_LIVE`。Q1 待 BR3 完工后复用其 pod。
- [09-12 16:18 PDT] **Q1 发射**：复用 exp87 BR3 的 pod `3s3noy34tubywz`（EU-RO-1，6 vCPU，213.173.110.197:20322），熵 0.03、无锚，从 M 80M 续。心跳 / 拉取+GCS（`exp88_Q1`）/ 在轨每 4M / TB `exp88_Q1_LIVE`。

