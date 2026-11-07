# exp127 — 平台来源拆分：主干 vs 动作头（A）、优化信号（B）

- **Date**: 2026-11-05（用户 11-05"同意做这些消融"）  **Status**: launching
- **Git**: 代码 `claude/perf-rollout-overlap` e5fbbfc（新增 `cnn_l_v3r_m46`；流式/one_forward/value_detach 已在 exp126 与本机冒烟验证）；运维 `experiments/exp127_ops/`
- **Env**: RunPod Secure RTX 3090（≥32 vCPU，$0.50/h）×4；评测本机 4080

## Purpose & Hypothesis
exp124（ConvFormer-M + 46 槽头）在每个里程碑领先 exp101（cnn_l + 374 槽头）4–5 pp，但同时换了主干和动作头，两变量从未分离；且 exp124 第二段（lr 3e-5）自 ≈72M 起平台在 0.50–0.51，112M→160M 已不涨（cnn_l 同期 +1.0 pp）。用户问"平台和主干关系很大吗"。
- **A（主干 vs 动作头，2×2 析因）**：H_A：+5 pp 主要来自主干（ConvFormer），而非 46 槽动作空间。
- **B（优化信号）**：H_B：ConvFormer-M 的第二段平台部分源于优化信号（批内数据落后、价值梯度主导共享主干），换流式调度、再加价值梯度截断能在 160M→192M 内推高平台。

## Method
- **A 臂**（从零，exp124 第一段配方：lr 1e-4、seed 1241、warmup 3000、batch 4096、1 epoch、dup_k 8、shaping 0→400k、bf16、compile，**同步调度**，0→32M）：
  - A1 `cnn_l_v3r_m46`（CNN 主干 + 46 槽，3.84M，rollout_values）
  - A2 `convformer_m_v3r`（ConvFormer 主干 + 374 槽原生头，2.00M；原生路径不支持 rollout_values，价值在打包阶段计算——与 exp101 一致）
  - 已有格子：CNN+374 = exp101 cnn_l 32M **0.4305**；ConvFormer+46 = exp124 A 32M **0.4808**（均为 2×1 万牌山、exp101 种子）。
- **B 臂**（从 exp124 A `games_160000000.pt` 续训到 192M，第二段配方 lr 3e-5、seed 1341、warmup 0；与 exp124 自身 160M→192M 同步续训逐点对照）：
  - B1 流式（K=512、块 128、队列 1，`--m46_one_forward`）
  - B2 流式 + `--value_detach`（价值头在截断梯度的特征上训练，主干只受策略梯度）
  - 对照：exp124 A 192M 判决读数（同起点、同 lr、同种子，同步调度）。
- 评测：A 臂 8/16/24M 各 2000 牌山、32M 判决 2×1 万；B 臂 168/176/184M 各 2000、192M 判决 2×1 万；T=0 复式对 bc49，exp101 种子。

## Success Criteria（发射前定）
- **A**：32M 判决下 主干效应 = ½[(A2−0.4305)+(0.4808−A1)]，动作头效应 = ½[(A1−0.4305)+(0.4808−A2)]，交互 = (0.4808−A2)−(A1−0.4305)。每项 SE ≈0.33–0.47 pp。≥ +1.0 pp 判有效；主干效应 ≥ 动作头效应 + 1.0 pp ⇒ H_A 成立。
- **B**：192M 判决下 B1 − exp124 ≥ +1.0 pp ⇒ 流式在规模上抬平台；B2 − B1 ≥ +1.0 pp ⇒ 价值截断再抬；≤ −1.0 pp ⇒ 有害。
- 局限：每格单种子；B 只有 32M 续训窗口。

## 预算
3090 Secure $0.50/h：A1 ≈20 h、A2 ≈31 h、B1/B2 ≈27 h 各 ⇒ ≈ **$53**，上限 **$65**；超过 1.3× 预估时长的臂报用户。

## 范围戳（判决须带）
ConvFormer-M 2.0M / cnn_l 4.0M（含 46 槽变体 3.84M），v3r 编码，exp124/exp101 配方，纯血自对弈；A 为 0–32M、B 为 160M→192M；T=0 复式对 bc49。

## Progress

- [2026-11-05 20:10 PST] 本机四臂冒烟全部通过（A1/A2 从零 1.2 万局，B1/B2 从 160M 续训 8k 局）；运维脚本 `exp127_ops/` 就绪（里程碑只拉一次 + md5 清单；每臂完成标记）。开 4 台 Secure 3090：两次 create-pod 被权限分类器拦截（真实交易），两次 ≥32 vCPU 的 3090 无货 ⇒ **未开出任何 pod，等用户放行/重试**。架构梳理见 `experiments/designs/architecture_ceiling_20261105.md`。

- [2026-11-06 00:55 PST] 用户在对话中批准方案 1（A 同步、B 流式）。Secure 3090（32 vCPU）与 A40（32/24 vCPU）均无货，create-pod 返回容量不足；改为每 30 分钟自动重试（cron 141adb69，仅限 Secure、≤$0.59/h），开出即部署。

- [2026-11-06 01:20 PST] **A1 发射**：pod n4zajmh86epzfd（exp127-A1，Secure RTX 3090，32 vCPU，$0.50/h，EU-CZ-1，213.192.2.111:40178，仅 22/tcp）；compile gate / arm gate 通过，首迭代 179 局/s；pull/watchdog/eval 循环已起，TB `exp127_A1_cnnl46_0-32M_LIVE`。A2/B1/B2：同轮 3090 与 A40 均无货，cron 141adb69 继续重试。

- [2026-11-06 01:43 PST] **A2 发射**：pod 83u5mfggd84ys4（exp127-A2，Secure RTX 3090，32 vCPU，$0.50/h，EU-CZ-1，213.192.2.83:40108，仅 22/tcp）；gates 通过，首迭代 131 局/s；循环 + TB `exp127_A2_convformer374_0-32M_LIVE` 已起。A1 20 min 时 93 万局、751 局/s。B1/B2：本轮 3090/A40 无货，继续重试。

- [2026-11-06 03:13 PST] **B1 发射**：pod v4onbm14zjcs0d（exp127-B1，Secure RTX 3090，32 vCPU，$0.50/h，EU-CZ-1，213.192.2.74:40168，仅 22/tcp）；start.pt = exp124_A_p2 games_160000000.pt（md5 校验通过），确认从 160,002,048 局续训，流式落后 ≈11 步；循环 + TB `exp127_B1_stream_160-192M_LIVE` 已起。B2：同轮 3090/A40 无货，继续重试。

- [2026-11-06 03:44 PST] **B2 发射，四臂全部在跑**：pod 9499s5utscnk4p（exp127-B2，Secure RTX 3090，32 vCPU，$0.50/h，EU-CZ-1，213.192.2.68:40188，仅 22/tcp）；start.pt md5 校验通过，确认从 160,002,048 局续训，`--value_detach` 已生效（config）；循环 + TB `exp127_B2_stream_vdetach_160-192M_LIVE` 已起。重试 cron 已删除。
  在跑：A1 cnn_l+46（≈735 局/s）、A2 ConvFormer+374（≈310 局/s）、B1 流式（≈405 局/s）、B2 流式+价值截断。四台合计 $2.00/h。

- [2026-11-06 04:07 PST] **B2 首台 pod 宿主慢，已终止重开**：9499s5utscnk4p 稳定在 95 局/s（B1 同期 414），每迭代 update/rollout/pack 全部慢 4×，宿主 CPU 为 EPYC 7H12（B1 为 7763），同时跑训练时 matmul 30.7 vs 48.4 TFLOPS。本机同配置 4 min A/B：流式 489 局/s，流式+value_detach 488 局/s ⇒ 价值截断本身不降速，是宿主问题。按 95 局/s 跑完 32M 需 ≈94 h ≈ $47，会突破 $65 上限 ⇒ 停本地循环、04:05 终止该 pod（运行 ≈40 min，≈$0.35，产物丢弃）；重开时 3090/A40 无货，cron 继续重试 B2。

- [2026-11-06 06:56 PST] **B2 重开成功**：pod r7lvhb32p3re45（exp127-B2，Secure RTX 3090，32 vCPU，$0.50/h，EU-CZ-1，213.192.2.90:40119，仅 22/tcp，宿主 EPYC 7C13）；从 160,002,048 局续训，`--value_detach`；9 min 时 **404 局/s**（B1 ≈413）。旧慢机镜像改名 `exp127_B2_slowhost_0642` / `exp127_pull_B2_slowhost_0642`。四臂全部在跑，重试 cron 已删。

- [2026-11-06 13:58 PST] **A1（cnn_l + 46 槽）到 32M**：13:37 TRAIN_DONE，镜像 md5 10a523b4… 一致，pod n4zajmh86epzfd 已自毁（get-pod 404），≈12.5 h × $0.50 ≈ **$6.3**；残留 watchdog 已按 PID 停。
  32M 判决（10,500 去重牌山，见 SKILLS 11-06 两段重叠勘误）：**A1 0.4384 ± 0.0046**（点差 −2114 ± 142）。同牌山配对：A1 − exp101 cnn_l+374 32M（0.4270，11-06 重读）= **+1.13 ± 0.61 pp**（动作头效应，CNN 主干上）；A1 − exp124 ConvFormer+46 32M（0.4810）= **−4.26 ± 0.63 pp**（主干效应，46 槽头上）。2×2 的第四格 A2（ConvFormer+374）≈ 12 h 后到 32M。

## Results

## Conclusion

## Artifacts
