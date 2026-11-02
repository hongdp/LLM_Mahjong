# exp125 ConvFormer-L 尺寸扫描（纯血从零，bc49 同结构只放大主干）

- **Date**: 2026-11-01 23:10 PST 预注册；**Status**: planned（闸门：exp124 A 64M 判决读数）
- **Git**: 代码 `claude/exp124-convformer-m46` @ zoo 提交（a8fef66 之后，含 `convformer_l_v3r_m46` / `convformer_ml_v3r_m46`）；上云脚本 `experiments/exp125_ops/`（exp124 套件，架构经 `ARCH` 环境变量，Stage 0 到 64M 即停、第二段需 `EXTEND_OK` 标记）
- **Env**: RunPod 4090（Secure $0.89/h；Community $0.34/h 有货优先），镜像 `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`；评测本机 4080 $0

## Purpose & Hypothesis
exp124 证明 bc49 的结构（ConvFormer46，2.0M）在纯血自对弈 32–45M 局就压过 384M 的 CNN（+5 pp vs bc49 份额；四方桌 +2.3–3.5k uma/场），
但 ConvFormer-M 的尺寸是按 BC 任务和本机显存定的，从未在纯血 RL 上扫过尺寸（10-28 定论"更大=更强不成立"只在 CNN 与 BC 上测过，范围戳不覆盖此处）。
**H1**：同配方只放大主干（6.6M，Mortal 298k 主干 10.8M 的 60%），同里程碑份额高于 M 版（结构红利随尺寸放大）。
**H0**：同里程碑不高于 M 版（每局样本效率不变，更大只更慢）。

## Method
- 臂 **L** = `convformer_l_v3r_m46`（256 通道 × 8 块 × 8 头，6.63M）；备选臂 ML = `convformer_ml_v3r_m46`（192×10×6，4.65M）仅在额度允许时加。
- 配方与 exp124 A **逐项相同**（lr 1e-4、warmup 3000、batch 4096、ppo_epochs 1、clip 0.2、熵 0.03、γ 0.995、λ 0.95、dup_k 8、2048 局/迭代、PBRS 塑形 0→400k、fast_trunk+compile、bf16 rollout、rollout_values）；只换 `--arch`。种子 1251。
- **对照 = exp124 A 自己的曲线**（同种子评测：4/8/16/24/32/40/48/56/64M，其中 32M/64M 为 2×10k 判决口径）。
- Stage 0：0→64M 单段即停（`pod_run.sh` 无 `EXTEND_OK` 不进第二段）；延长到 192M 另批。

## Config
`pod_run.sh L 1e-4 1251`，`ARCH=convformer_l_v3r_m46`；里程碑 4M×8 + 40/48/56/64M；`--ckpt_every 400`。

## Success Criteria（发射前定）
1. **发射闸门（用户 11-01 批准）**：exp124 A 64M 判决读数（2×10k）**≥ 0.46 且 32M→64M 斜率 ≥ 0**；不满足则不发，等 192M 再议。
2. **健康**：L 的 8M 读数 ≥ exp124 A 8M（0.441）− 2 pp；熵 ≥ 0.6；KL 常态。
3. **主判据（64M 判决口径，2×10k 同种子）**：L − M(exp124 A 64M) ≥ **+1.5 pp** ⇒ H1，申请延长到 192M；+0.5～+1.5 ⇒ 弱，报用户；< +0.5 ⇒ H0（尺寸不是杠杆，关闭）。
4. 在轨读数只用于健康/早停；32M 处 L ≤ M − 1.5 pp ⇒ 早停。

## 预算（本机 4080 实测，随机初始化，exp124 配方；4090 换算用 exp124 实测比 444/337 = 1.32）
| 架构 | 参数 | 本机 局/s（rollout/打包/更新 s） | 4090 估 局/s | 64M 墙钟 | Secure | Community |
|---|---|---|---|---|---|---|
| M（exp124，参照） | 2.00M | 337（2.5/0.18/3.4） | 444（实测） | 40 h | $36 | $14 |
| ML | 4.65M | 203（3.4/0.2/6.4） | ≈268 | ≈66 h | ≈$59 | ≈$23 |
| **L（采用）** | 6.63M | 184（3.7/0.2/7.2） | ≈243 | ≈73 h | **≈$65** | ≈$25 |
compile 闸门（FC_DEBUG 逐调用对照 eager）：ML maxdiff 0.0252 / max|dV| 0.0059；L 0.0239 / 0.0039（阈 0.06 / 0.05）。
**申请额度**：L 到 64M ≈ $65（Secure）；上限 **$80**。ML 第二臂另加 ≈$59（可选，用户定）。

## Progress
- [2026-11-01 23:10] zoo 条目 + 闸门 + 吞吐基准完成（$0）；ops 脚本就绪；等 exp124 A 64M 读数（预计 11-02 09:30 PST）。

## Results

## Conclusion

## Next Steps

## Artifacts
| Path | Size | Description |
|---|---|---|
| `experiments/probes/exp125_bench/` | — | 本机闸门与基准日志（gate_*.log / bench_*/train_log.json） |
