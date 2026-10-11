# exp106 — 优化器热重启（Adam 矩重置）是不是纯血线的杠杆

- **日期**：2026-10-11 预注册  **状态**：running
- **Git**：`7371bed`（`--reset_optimizer`）+ 本文件
- **Env**：纯血线；Rust 引擎镜像自对弈；`cnn_l_v3r`（无宏、无新输入）从 `exp102_A/games_final.pt`（224M，α=0.003 退火终点，20k 对 vs bc49 0.4509/0.4511）`--resume`；RunPod Secure RTX 4090 $0.74/h × 3
- **用户批准**：2026-10-11 "三臂都跑，exp106 发射"

## Purpose & Hypothesis
exp105 的 $0 复评（10-10）发现：exp103 的两个 widened 臂（加载时 Adam 矩被重置）在 228–232M 的 8 个读数全部 ≥0.464，M1 均值 0.471 / M2 0.468，而原样 resume（矩保留）的 M0 0.461、谱系窗口 0.4605；M2 232M 0.4706/0.4703 是纯血线最高的 20k 对读数。
三臂的系统性差别只有：①Adam 矩重置；②前 ≈4M 局 ≈1% 的弃和入口行；③训练 seed。日志已排除"重置 = 瞬时大步"（重置臂前 1M 局 KL 反而更低）与"评测时结构不同"（logits 位级一致、EV 一致）。
**H1（热重启）**：从同一 ckpt 续训时丢弃 Adam 的 m/v（预条件器重排，总步长不变、方向改变）能让平台上的策略再升 ≥ +0.5 pp。
**H0**：exp103 的读数是 ckpt 噪声 + seed 运气（我们从未测过同配置两条 8M 续训的 seed 间方差）。
纯度：零人类结构、零外部模型，仅优化器状态；成立则下一步是周期性重启（SGDR 式）。

## Method
三臂，全部 `cnn_l_v3r`，从 exp102 A 224M 续 **16M 局 → 240M**，配方 = exp103 M0（lr 3e-5、α 0.003 常数、gpi 2048、dup_k 8、γ 0.995、λ 0.95、target_kl 0.03、clip 0.2、batch 4096、1 pass、adv_clamp 5、amp_update、shaping 已退火）：
- **R**：`--reset_optimizer`（权重保留、Adam 矩清空），seed 1061；
- **C**：原样 resume（矩保留），seed 1061；
- **C′**：原样 resume，seed 1062 —— 测 seed 间方差。
里程碑 228/232/236/240M；每臂一台 Secure 4090（≈950 局/s ⇒ 16M ≈ 4.7 h ≈ $3.5/臂）；**合计 ≈$10.5，上限 $14**。
评测（本机 $0）：每里程碑 ckpt 10k 牌山 × 2 段（seed0 102224000/102224500）T=0 vs bc49，与 exp102/exp103/exp105 全部同牌山可配对；脚本 scratchpad `pairdiff_eval.py`。
对照数据已在手：exp103 M0 228/232M、M1/M2 228/232M、exp102 A 200–224M 窗口。

## Success Criteria（预注册）
1. **健康**：KL < 0.03、H 0.2–0.4、吞吐 ≥ 800 局/s；三臂首迭代 EV ≈ 0.10（critic 完整）。
2. **seed 方差**（C vs C′，4 里程碑 × 2 段配对差的均值与散布）：给出 σ_seed；若 |C − C′| 窗口均值 ≥ 0.8 pp ⇒ 单臂判决不可信，一切结论需 ≥2 seed。
3. **主判决（窗口均值 228–240M，4 ckpt × 2 段）**：R − mean(C, C′) ≥ **+0.5 pp** 且点差 > 0 且 R 窗口均值 ≥ 0.465 ⇒ **H1**，热重启是杠杆（下一步：R 续到 256M 看是否回落 + 周期性重启臂）；
   |R − mean(C,C′)| < 0.5 pp ⇒ H0，exp103 的 0.470 归为噪声/运气（并把 σ_seed 写进 SKILLS 作为今后判决的下限）；
   R − mean(C,C′) ≤ −0.5 pp ⇒ 重置有害。
4. **早停**：任一臂 KL > 0.03 持续或 H < 0.15 ⇒ 停该臂；无里程碑早停（在轨读数只做健康）。
5. **运维**：TRAIN_DONE ⇒ 本机 md5 握手 ⇒ 自毁；拉取循环主 checkout 绝对路径、rsync 退出码报警；发射相 15 min 死线 / 30 min STALL。

## Progress
- [10-11] `--reset_optimizer` 已加（`7371bed`）；本机冒烟（2 迭代）打印 "Adam moments discarded (warm restart)" 并正常训练。
- [10-11 14:16 PDT] 三台 Secure RTX 4090 开机（EU-RO-1，16 vCPU，**$0.89/h**——比 exp105 时的 $0.74 涨价，库存 Low；三臂 ≈4.7 h ⇒ ≈$12.6，仍在 $14 上限内）：
  R `kon8ezplbs5g29` 213.173.102.212:28384（CUDA 12.8）/ C `nsikefnliqirk4` 213.173.110.42:14615（13.0）/ C′ `kr6uym6xiee3ct` 213.173.110.38:10801（13.0）。
  起点 ckpt md5 `cf66bae9…` 三端一致，repo.tgz `aa297f06…`（HEAD `fb75e4b`）。运维脚本：主 checkout `experiments/exp106_ops/`（`pod_run.sh ARM SEED [flags]`、`pull_loop.sh HOST PORT ARM`）；TB `exp106_{R,C,Cp}_LIVE`。
- [10-11 14:20 PDT] 三臂启动，首迭代 EV 0.094–0.110（critic 完整）、KL 0.001–0.002、H 0.28–0.31。吞吐：C **1256 局/s**、C′ **993**、R 只有 **577**（48 核宿主，核慢）⇒ R 预计 7.7 h ≈ $6.9，总额顶到上限。
  再开一台 R2 `j8gdrtqoqs6qu4`（EU-CZ-1，36 vCPU，213.192.2.77:40035）试机：**928 局/s**，前 15 迭代与旧 R 逐位一致（pg/H/KL/EV 全同 ⇒ 续训确定性） ⇒ R 切到 R2，旧 R pod 终止（≈12 min ≈ $0.2）。
  修正预计：R 4.8 h $4.3 + C 3.5 h $3.1 + C′ 4.5 h $4.0 + 试机损耗 ≈ **$11.6**（上限 $14）。拉取循环三条（主 checkout）、TB 三面板已挂。

## Results
（待）

## Conclusion
（待）

## Artifacts
（待）
