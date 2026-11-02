# exp125 ConvFormer-XL 尺寸扫描（纯血从零，bc49 同结构只放大主干到 Mortal 参数量）

- **Date**: 2026-11-01 23:10 PST 预注册；**Status**: approved（用户 11-01 23:55 批 $120 上限；闸门：exp124 A 64M 判决读数 ≥0.46 且 ≥ 32M 均值 0.4808，过则由半小时巡检自动发射）
- **Git**: 代码 `claude/exp124-convformer-m46` @ zoo 提交（a8fef66 之后，含 `convformer_xl_v3r_m46` / `convformer_l_v3r_m46` / `convformer_ml_v3r_m46`）；上云脚本 `experiments/exp125_ops/`（exp124 套件，架构经 `ARCH` 环境变量，Stage 0 到 64M 即停、第二段需 `EXTEND_OK` 标记）
- **Env**: RunPod 4090（Secure $0.89/h；Community $0.34/h 有货优先），镜像 `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`；评测本机 4080 $0

## Purpose & Hypothesis
exp124 证明 bc49 的结构（ConvFormer46，2.0M）在纯血自对弈 32–45M 局就压过 384M 的 CNN（+5 pp vs bc49 份额；四方桌 +2.3–3.5k uma/场），
但 ConvFormer-M 的尺寸是按 BC 任务和本机显存定的，从未在纯血 RL 上扫过尺寸（10-28 定论"更大=更强不成立"只在 CNN 与 BC 上测过，范围戳不覆盖此处）。
**H1**：同配方只放大主干到 Mortal 参数量（XL 11.37M，Mortal 298k 主干 10.82M +5%；用户 11-01 要求『match Mortal 略超』），同里程碑份额高于 M 版（结构红利随尺寸放大）。
**H0**：同里程碑不高于 M 版（每局样本效率不变，更大只更慢）。

## Method
- 臂 **XL** = `convformer_xl_v3r_m46`（288 通道 × 11 块 × 9 头，11.37M，每头 32 维与 M 一致）；中间点 L = `convformer_l_v3r_m46`（256×8×8，6.63M）/ ML（4.65M）仅在额度允许时加，构成 M→L→XL 尺寸曲线。
- 配方与 exp124 A **逐项相同**（lr 1e-4、warmup 3000、batch 4096、ppo_epochs 1、clip 0.2、熵 0.03、γ 0.995、λ 0.95、dup_k 8、2048 局/迭代、PBRS 塑形 0→400k、fast_trunk+compile、bf16 rollout、rollout_values）；只换 `--arch`。种子 1251。**显存**：XL batch 4096 在本机 16 GB OOM（PyTorch 已分配 15.2 GiB 时爆），pod 24 GB 由 pod_boot 闸门实测；若 pod 也 OOM 则退到 `--batch 2048`（每迭代 2 个 minibatch，记为配方偏差）。
- **对照 = exp124 A 自己的曲线**（同种子评测：4/8/16/24/32/40/48/56/64M，其中 32M/64M 为 2×10k 判决口径）。
- Stage 0：0→64M 单段即停（`pod_run.sh` 无 `EXTEND_OK` 不进第二段）；延长到 192M 另批。

## Config
`pod_run.sh XL 1e-4 1251`，`ARCH=convformer_xl_v3r_m46`（ops 默认）；里程碑 4M×8 + 40/48/56/64M；`--ckpt_every 400`。

## Success Criteria（发射前定）
1. **发射闸门（用户 11-01 批准）**：exp124 A 64M 判决读数（2×10k）**≥ 0.46 且 32M→64M 斜率 ≥ 0**；不满足则不发，等 192M 再议。
2. **健康**：L 的 8M 读数 ≥ exp124 A 8M（0.441）− 2 pp；熵 ≥ 0.6；KL 常态。
3. **主判据（64M 判决口径，2×10k 同种子）**：XL − M(exp124 A 64M) ≥ **+1.5 pp** ⇒ H1，申请延长到 192M；+0.5～+1.5 ⇒ 弱，报用户；< +0.5 ⇒ H0（尺寸不是杠杆，关闭）。
4. 在轨读数只用于健康/早停；32M 处 XL ≤ M − 1.5 pp ⇒ 早停。

## 预算（本机 4080 实测，随机初始化，exp124 配方；4090 换算用 exp124 实测比 444/337 = 1.32）
| 架构 | 参数 | 本机 局/s（rollout/打包/更新 s） | 4090 估 局/s | 64M 墙钟 | Secure | Community |
|---|---|---|---|---|---|---|
| M（exp124，参照） | 2.00M | 337（2.5/0.18/3.4） | 444（实测） | 40 h | $36 | $14 |
| ML | 4.65M | 203（3.4/0.2/6.4） | ≈268 | ≈66 h | ≈$59 | ≈$23 |
| L | 6.63M | 184（3.7/0.2/7.2） | ≈243 | ≈73 h | ≈$65 | ≈$25 |
| **XL（采用，Mortal 量级）** | 11.37M | 116（5.2/0.25/12.3，batch 2048） | ≈153 | ≈116 h | **≈$103** | ≈$40 |
compile 闸门（FC_DEBUG 逐调用对照 eager）：ML maxdiff 0.0252 / max|dV| 0.0059；L 0.0239 / 0.0039；XL（batch 2048）0.0264 / 0.0156（阈 0.06 / 0.05）。
**申请额度**：XL 到 64M ≈ **$103**（Secure）/ ≈$40（Community 有货时）；上限 **$120**。延长到 192M 另批（Secure ≈ +$210，Community ≈ +$80）。L 中间点另加 ≈$65（可选，用户定）。

## Progress
- [2026-11-01 23:10] zoo 条目 + 闸门 + 吞吐基准完成（$0）；ops 脚本就绪；等 exp124 A 64M 读数（预计 11-02 09:30 PST）。
- [2026-11-01 23:55] **用户批准额度 $120 上限**；巡检 cron 接入闸门判读与自动发射（Community 4090 优先，否则 Secure；pod 闸门 OOM 则 BATCH=2048 重部署）。
- [2026-11-01 23:45] 用户要求主臂改为 Mortal 参数量：加 XL（288×11×9，11.37M）；本机 4096 OOM、2048 闸门过、116 局/s；预算改为 ≈$103 / 上限 $120；ops 默认 ARCH=xl。
- [2026-11-02 10:05 PST] **闸门通过**（exp124 64M 均值 0.4890 ≥ 0.4808）。Community 4090 全部缺货（含 vCPU 过滤与否），Secure 4090 带 vCPU≥12 过滤也拒，去掉过滤成功：pod **`ih7vwfbj4dxg15`**（exp125-XL，Secure $0.89/h，EUR-IS-1，64 vCPU 宿主，host CUDA 13.2，ssh root@157.157.221.29 -p 33087）。10:08 开始部署（bootstrap + compile 闸门 batch 4096）。TB：`_tb_qlearning/exp125_XL_convformerXL_0-64M_LIVE` → `experiments/exp125_XL/tensorboard`。

## Results

## Conclusion

## Next Steps

## Artifacts
| Path | Size | Description |
|---|---|---|
| `experiments/probes/exp125_bench/` | — | 本机闸门与基准日志（gate_*.log / bench_*/train_log.json） |
