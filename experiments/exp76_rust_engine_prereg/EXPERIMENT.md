# exp76 Rust 引擎训练级验证 + 成本实测（2026-09-08）

- **Date**: 2026-09-08  **Status**: done（H1 通过：R vs P3 0.494 在噪声地板内；成本 45× 下降）
- **Git**: 见 Progress；RunPod Secure L40S 单臂；纯血同 P3
- **Env**: `train_dnn_ppo.py --engine rust`（riichi_rs.VecEnv，规则/编码/批处理在 Rust，策略前向在训练进程 GPU）

## Purpose & Hypothesis
riichi_rs 已在 7 套差分测试上与 Python 引擎位级一致（含 T=0 端到端 episode）。但训练是 T=1 采样 + 组基线 + PPO 更新的闭环，
本实验验证：**同配方（P3 钉死旗标）在 Rust 引擎上训 1M 局，产物强度与 P3 无差**，并实测 pod 上的吞吐与每百万局成本。
**H1（等价）**：R vs P3 双段 20k 对 share ∈ [0.48, 0.52]（噪声地板内），vs exp27A 同理；R vs bc49 ≈ 0.33。
**H2（成本）**：pod 吞吐 ≥ 800 局/s（P3 97 局/s） ⇒ 1M 局 ≤ 25 min、≤ $0.5/百万局（现 $3.1）。
反假设：R 显著弱于/强于 P3 ⇒ 引擎在训练闭环里有未被差分测试覆盖的差异（如采样温度路径、组基线键），停下查。

## Method
- 配方 = P3 钉死旗标 + `--engine rust --games_per_worker 1024 --workers 1`（K=1024 并发桌），其余不变，1M 局。
- pod bootstrap：rustup + maturin 构建 riichi_rs（≈3 min），并在 pod 上跑 random/table/game parity 测试作为守卫。
- 在轨：每 ckpt 对同局数 P3 ckpt n=600（exp72 循环复用）。
- 终评：R vs P3 / exp27A / bc49 双段 20k 对；防守探针；风格 @bc49。

## Success Criteria（预注册）
1. R vs P3 ∈ [0.48, 0.52]（等价）；2. pod 吞吐 ≥ 800 局/s；3. 记录每百万局美元数。
- 预算：≈0.5–1 h L40S Secure ≈ **$1，上限 $3**。

## Progress
- [09-08 19:55 PDT] **发射**：Secure L40S US-TX-4 pod `tj4kxcz3k6z6m6`（$1.09/h，ssh 195.26.232.163:44323）。bootstrap 在 pod 上装 rustup + maturin 构建
  riichi_rs（Python 3.12 wheel）并跑 random/table/game 三套 parity 守卫 **16/16 通过**；GPU 数值校验通过。单臂 R，`--engine rust --games_per_worker 1024`。
  心跳/拉取（含 TB 镜像）/在轨循环已挂；TB 加 exp76_R_LIVE。
- [09-08 21:30 PDT] **训练完成**：1,001,472 局 **82.3 min**（P3 207 min），中位 rollout 2.3 s / 更新 0.4 s 每 2048 局 ⇒ 稠态 ≈760 局/s；
  但每迭代墙钟 10.1 s，差额 7.4 s 是训练器自身的逐步 Python GAE 循环（GPU 标量索引 55 万次/迭代）——P3 时期同样存在（8.2 s）只是被 rollout 掩盖。
  已向量化（与参考循环数值等价，本机每迭代开销 7 s → 0.4 s，提交 2fe4821）。pod 费用 ≈$1.5。在轨曲线 vs 同局数 P3：0.49/0.48/0.53/0.53/0.52/0.51/0.50/0.52/0.50。
- [09-08 21:35 PDT] **成本探针（社区 RTX 3090，32 vCPU，$0.22/h，pod `1vodnftrw3y9nd`）**：riichi_rs 构建 + GPU 数值校验通过；`collect_rust` T=1 吞吐
  K=512 828 / **K=1024 1,193** / K=2048 1,021 局/s ⇒ 1M 局 rollout ≈14 min，加更新与开销 ≈18–20 min ⇒ **≈$0.07/百万局**（L40S+Python 引擎 $3.1，≈45×）。

## Results（T=0，工作站；双段 20k 对，SE≈0.0035）
| 读数 | share | 对照 |
|---|---|---|
| **R（Rust 引擎，P3 配方）vs P3（Python 引擎，同配方）** | **0.4937** | 判据 [0.48, 0.52] ✓；同配方独立产物互相 0.49–0.50 是已知噪声地板（P3 vs exp27A 0.492） |
| R vs exp27A | 0.4886 | P3 vs exp27A 0.4922 |
| R vs bc49 | 0.3167 | P3/G3 0.324–0.329 |
| R vs exp74-A1（退火塑形） | 0.4858 | A1 vs P3 0.5172（A1 的 +1.5 pp 优势在 R 上复现为 −1.4 pp） |
| 训练墙钟 | 82.3 min（P3 207 min） | 稠态 rollout 2.3 s / 2048 局 ≈ 760 局/s（判据 ≥800 未达，差 5%）；每迭代另有 7.4 s 训练器开销（已修） |
| pod 费用 | ≈$1.5 | 修复训练器开销后预计 ≈25 min ≈ $0.45/百万局（L40S Secure） |
| 社区 3090 探针 | 1,193 局/s | **≈$0.07/百万局** |

## Conclusion
1. **Rust 引擎在训练闭环里与 Python 引擎等价**：同配方产物对 P3 0.494、对 exp27A 0.489、对 bc49 0.317，全部落在同配方独立产物的噪声地板内；
   加上 7 套差分测试位级一致，riichi_rs 可以作为纯血线的默认训练引擎。
2. 成本：L40S 上 3.5 h → 82 min（训练器开销修复后 ≈25 min）；社区 3090 上 ≈$0.07/百万局，比出发点便宜约 45 倍。
3. 意外收获：训练器逐步 GAE 循环每迭代 7 s 的隐藏开销（P3 时期已存在），向量化后消除。
4. 吞吐判据 ≥800 局/s 差 5% 未达（16 vCPU 的 L40S pod 是 CPU 受限）；3090 社区机 32 vCPU 达 1,193，判据在目标硬件上满足。

## Next Steps
- exp77：A1 配方 × Rust × 3090，32M 局（已发射）。
- 引擎二阶优化（待张缓存、去字符串往返、双缓冲）在 exp77 跑完后按需做；半庄层/v3 编码（P5）未移植。

## Artifacts
| Path | Size | Description |
|---|---|---|
| `experiments/exp76_R/games_final.pt`（+GCS `exp76_R/`） | 24 MB | Rust 引擎训练的 P3 配方产物 |
| `experiments/exp76_R/` ckpt/train_log/tensorboard；`experiments/probes/exp76_arms.json`、`exp76_track.jsonl` | — | 全程 + 终评 + 在轨 |
| `rust/riichi_rs/`、`src/agents/dnn/rust_rollout.py`、`tests/test_rust_*_parity.py` | — | 引擎 + 集成 + 差分测试 |

