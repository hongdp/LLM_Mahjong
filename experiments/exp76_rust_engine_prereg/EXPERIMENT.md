# exp76 Rust 引擎训练级验证 + 成本实测（2026-09-08）

- **Date**: 2026-09-08  **Status**: running（发射）
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
