# perf — rollout 第二轮优化（发射第一批前，用户 03:15 指令）

- **Date**: 2026-08-23 03:15–04:10 PDT  **Status**: done
- **Git**: master 52318ad + 本提交（`_worker` 的 K 分派修复）
- **Env**: 本机 24 核 + RTX 4080，GPU rollout 基建，`scripts/bench_rollout_infer.py --gpu`

## Purpose
第一批纪元 3 训练（5 个 run）前再榨一轮 rollout 吞吐：找当前瓶颈并消除，直接按比例省 GPU 小时。

## Method / Findings
1. **HandSet 分支 GPU 受限**：`handset_xl`（20M）137 个候选槽位里只有 ~14–18 个真 token，手写 fp32 注意力 →
   前向 7.5→14 ms/批随批线性增长，15 局/s。改为先把存在的牌压成 ≤32 个 token + fused SDPA：**15 → 42 局/s**（K=1）。
2. `legal_mask` 每决策正则解析 + torch 逐项赋值 ~0.1 ms → 字符串→索引 memo + numpy 构建。
3. **向量化 worker**：每进程交错 K 局（`play_game_gen` 生成器版 play_game，逻辑等价，有测试），一轮把 K 局的
   待决策一次性 RPC；服务端槽位 = workers × 3K，凑批窗口改为按"请求源数"而非槽位数判定。
   第一版漏接了 `_worker` 的 K 分派（脚本没保存），基准看起来反而变慢——diag 打印 rows/round 才发现。

## Results（本机，24 workers；局/s）
| 架构 | K=1（旧） | K=8 | K=16 | **K=32** | 服务端状态 |
|---|---|---|---|---|---|
| cnn_m_r（2M） | 78 | 137 | 146 | **204** | 前向 3.3 ms/批 291 行，仍 CPU 侧封顶 |
| handset_xl（20M） | 42（压缩前 15） | — | 85 | **97** | 前向 27 ms/批 358 行 → **GPU 封顶** |
训练器冒烟（K=32 + 混合温度）：155.8 局/s（含 PPO 更新）。

## Conclusion
- 云端发射配置：`--games_per_worker 32 --infer_max_batch 512`（L4 ≈ 4080 的 0.6×：cnn 臂预计 CPU 封顶 ~100 局/s，
  handset_xl 臂预计 ~55–60 局/s，GPU 封顶）。每 run 1.0M 局 ≈ 3–5 h，比 exp22-r2 的 38.8 局/s 省 2.5–5×。
- 下一步若还要提：handset 分支 bf16 推理（GPU 侧 ×2）；cnn 臂 CPU 侧已是 0.2 ms/决策（引擎 shanten 占大头）。

## Artifacts
| Path | Description |
|---|---|
| scratchpad bench 输出（本文件表格） | 基准数字 |
