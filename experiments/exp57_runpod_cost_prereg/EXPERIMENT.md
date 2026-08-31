# exp57 — RunPod 性价比裁决（扩展曲线 + 实查目录 + 在 pod 实测每核速率）

- **Date**: 2026-08-30 17:45  **Status**: running
- **Git**: 3964de8（工作树干净；scaling_bench.py 在会话 scratchpad，正文已抄录本目录 `scaling_bench.py`）
- **Env**: 本机 RTX 4080 16GB + 24 核桌面 CPU；rlhf_mahjong conda env；RunPod 目录 = 2026-08-30 实查

## Purpose & Hypothesis
`docs/runpod_cost_and_ops.md` 的整张性价比表压在「5.8 局/s/核、线性到 48 核」的未验证外推上。
本实验分三步裁决「RunPod 上是否存在比 g4-standard-48 flex（$2.25/h，48 vCPU，实测 279 局/s）更划算的训练选项」：

1. **H1（扩展形状）**：rollout 吞吐随 worker 数的曲线在 24 核内近线性（每核速率衰减 <20%）。
   若在 ~16 核就饱和 → 核数不再是采购的主变量，整张表按曲线重算。
2. **H2（目录）**：以 GraphQL `lowestPrice.minVcpu` 实查为准重建候选表（文档旧表已知至少
   3090 community 的「16 核」与实查 8 核不符）。
3. **H3（每核速率）**：候选宿主的实测每核速率 ≥ g4 的 5.8 局/s/核（文档 §1.2 预注册的 ~$0.25 探针）。

## Method
- **本地扩展基准**（上会话排队，chain3 等机器空闲自动开跑）：bc49 ckpt，deal/hanchan 两模式 ×
  workers ∈ {1,2,4,8,12,16,20,24}，gpu_infer=on，产物 `experiments/probes/exp57_rollout_scaling.json`
  （含每点 GPU 利用率以定瓶颈）。
- **目录实查**：REST v2 list-gpu-types（价格/库存/上限）+ GraphQL lowestPrice（每卡保证 vCPU/RAM，
  分 secure/community、gpuCount 1–4）。快照存本目录 `catalog_snapshot/`。
- **RunPod 探针**：对目录+曲线联合裁决出的最优候选开一台 community/secure pod，跑同一个
  `scaling_bench.py`（git 同 SHA），得实测每核速率与实配核数；用完 terminate（不 stop）。
- 基线对照：g4-standard-48 flex 实测 279 局/s（docs/runpod_cost_and_ops.md §1，cnn_m_r K=32）。
  注意基准模型不同（本实验用 bc49 conv），跨文档比较以「同脚本同 ckpt 的本机 24 核点」为换算桥。

## Config
- scaling_bench.py：GAMES_PER_WORKER_TOTAL=24，games_per_worker=16（向量化 K=16），
  infer_max_batch=128，wait_ms=0，temperature=1.0，seed 70000000。
- 探针 pod 预算：**≤ $0.60**（1 台 ≤1h，候选价 $0.22–0.50/h；超出即中止并记录）。

## Success Criteria
- **裁决产出**：一张以实测曲线（非线性外推）+ 实查价格计算的 局/s/$ 表，明确「换/不换 g4」。
- H3 探针判「候选胜」需：实测 局/s/$ ≥ g4 基线（124 局/s/$，cnn_m_r 口径需按 bc49 换算桥折算）
  的 **1.5×**（低于 1.5× 不值得引入 community 可靠性风险与迁移成本）。
- 副产物（§3.2 误差棒）：确定性回放 12 对逐字节一致 ⇒ 判「二项 SE 正确，2.2σ = 牌山运气 +
  跨锚共享牌山相关性」；不一致 ⇒ 立项查非确定性来源。

## Progress
- [17:30] chain3 已排队（等 spec_probe + mortal 链结束后自动跑本地扩展基准）。
- [17:39] 目录实查完成：REST 46 卡 + GraphQL vCPU 表。要点：3090 community 最低价宿主
  仅 8 vCPU/卡（文档写 16，前提 3 被否）；3090 secure 32 vCPU/卡（×2=$1.00/64 核）；
  3090 Ti community 28 核/$0.27；A4000 community 16 核/$0.17；A5000 断货。
- [17:42] §3.2 确定性回放发射（12 对，seed0=51000000，对照探针批存档逐字节比对）。
- [17:44] §3.2 中间结论:两批温度均 0、无 fallback、无平局、代码/ckpt/池无变动;跨锚共享
  牌山相关 r=+0.2~0.3（bc49↔exp46I，策略相近所致），bc51 与两者不相关 ⇒ 「两锚同向偏」
  非独立复现。
- [17:50] **§3.2 结案**：12 对确定性回放与探针批存档逐字节一致（24 场 uma/顺位/局数全同，
  不同 GPU 负载下重放）。T=0 对局 = 种子的确定函数 ⇒ 二项 SE 数学正确（无平局时还严格等于
  经验方差）。判决：**2.2σ = 牌山运气 + 选择效应**，n=4000 结论不受影响，n≤400 读数按
  标称 SE 使用。新发现：五锚 Mortal 评分共用 seed0=49900000 同段牌山 ⇒ 相似策略锚
  （bc49↔exp46I r≈+0.2~0.3）读数正相关，池拟合 SE 偏乐观、三角闭合残差不可当独立证据；
  **以后每锚用不重叠种子段**。回放件：determinism_replay.json（已拷本目录）。

## Results
（待扩展曲线 + 探针）

## Conclusion
（待）

## Next Steps
（待）

## Artifacts
| Path | Size | Description |
|---|---|---|
| experiments/probes/exp57_rollout_scaling.json | 待 | 本地 1→24 worker 吞吐曲线 |
| experiments/exp57_runpod_cost_prereg/catalog_snapshot/ | 待 | 2026-08-30 RunPod 目录快照 |
