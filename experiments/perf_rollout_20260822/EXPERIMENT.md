# perf：rollout 引擎优化（scale-up 前置工程，用户指令 2026-08-22）

- **Date**: 2026-08-22  **Status**: done（引擎侧三轮完成；GPU 批推理为下一阶段）
- **Git**: 4d7a187 起  **Env**: 本地单线程 CPU 剖析（cnn_m，exp18-cnn 权重）

## Purpose
SKILLS 记录 rollout 时间构成为引擎 77%/网络 23%；scale-up 与 GPU 批推理的收益上限受引擎份额
封顶（Amdahl），故先砍引擎。目标：单线程局/s ×2 以上，**对局语义逐字节不变**。

## Baseline（cProfile，30 局，温度 1.0）
- 2.47 局/s 单线程；75 决策/局；5.40 ms/决策；总 12.14 s。
- **向听计算 7.73 s（64%）**：`do_discard` 后对 3 家各查 `_waits`，每次 miss 对 34 张牌各算一次
  shanten（60,740 次 shanten 调用 / 1,780 次打牌 ≈ 34/打牌）。`_waits` 虽有缓存但每家每巡
  手牌变化即 miss。
- `get_legal_actions` 1.62 s（立直候选逐张 `_shanten`）；`_can_ron`/`_win_result` 0.9 s。
- 编码 `encode_state` 1.95 s（16%）：`_counts_plane` 纯 Python 逐牌累加 1.36 s。
- 网络前向 ~1.7 s（14%）。

## Method（按收益排序，每步跑等价性快照 + 单测）
1. `_waits` 早退：13 张手牌先算一次 shanten，非听牌直接返回空（省 34→1）；听牌时只扫描
   「与手牌同花色 ±2 或同牌」的候选（国士形全扫）。
2. `_shanten` 进程级 LRU（按排序牌组+副露数键）：同手牌在立直候选枚举/复式重放中大量重复。
3. `_counts_plane` 向量化（预建 tile→34 索引，torch.bincount）。
4. 复测剖析，记录新时间构成；30 workers 云端吞吐换算。

## Success Criteria
单线程 ≥5 局/s（≥2×）；`traj_before.json` 与改后快照 sha256 一致（60 局贪心轨迹+结算）；
engine/encoder 单测全过。

## Progress
- [2026-08-22 13:10] 基线剖析完成；贪心轨迹快照 sha256 `0b8eb9e567ecea2e`（60 局）；pytest 安装，62 测试基线通过。
- [2026-08-22 13:40] **第一轮（方法 1-3）**：5.91 局/s（2.39×），2.43 ms/决策；shanten 调用 60,740→15,487；
  等价性 hash 一致；62 测试通过。新构成（5.1 s）：shanten 1.9 s（37%，其中 mahjong 库纯 Python
  `_run` 1.4 s）、网络 1.6 s（31%）、其余编码/合法动作/结算 ~1.6 s。
- [2026-08-22 14:20] **第二轮**：①立直候选枚举先看 14 张向听（非 0 跳过逐张扫描）；②荣和判定用
  memoized waits 预门控（同一条件，免逐次 shanten）；③DNN 路径关闭每步 LLM 文本观测拼装
  （`table.text_obs=False`）；④编码器改 numpy 一次成型。结果 **9.77 局/s（累计 3.96×）**，
  1.31 ms/决策；shanten 调用 4,327；hash 一致；62 测试通过。剩余构成（3.1 s）：网络 conv/linear/bn
  ~1.05 s（34%）、shanten 0.3 s、tile_to_34 字符串解析 0.15 s、legal_mask 0.17 s。
  引擎份额 77%→~45%：GPU 批推理的 Amdahl 上限相应抬高。

- [2026-08-22 14:45] **第三轮**：tile_to_34 查表（160k 次/30 局）。无 profiler 实测 **14.76 局/s**；
  等价性 hash 一致；全套 130 测试通过。为公平对比，用临时 worktree 跑 pre-perf 代码（b38ce3a^）
  同口径无 profiler 基线 = **4.65 局/s**（此前 2.47/9.77 等数字均含 cProfile 开销，只可互比）。

## Results
| 指标 | 优化前 | 优化后 | 判据 |
|---|---|---|---|
| 单线程吞吐（无 profiler，30 局） | 4.65 局/s | **14.76 局/s（3.17×）** | ≥2× ✅ |
| shanten 调用 / 30 局 | 60,740 | 4,327（−93%） | — |
| 贪心轨迹 sha256（60 局） | 0b8eb9e567ecea2e | 0b8eb9e567ecea2e | 一致 ✅ |
| 测试 | 130 passed | 130 passed | ✅ |

## Conclusion
1. 引擎份额从 77% 降到 ~45%，rollout 单线程 3.17×；全部改动语义零变化（轨迹逐字节一致 +
   全套测试），可直接用于后续所有 run（云端 30 workers 预期 25 → ~60-80 局/s，下次发射实测）。
2. 剩余构成以网络前向为主（conv/linear/bn ~1/3）+ mahjong 库纯 Python shanten（~10%）。
   下一杠杆即 SKILLS 预定的 **GPU 批量推理服务**（worker 只跑引擎、GPU 凑 batch），
   引擎份额下降使其 Amdahl 上限同步抬高；这是 scale-up（192×40 级）的前置工程。
3. 方法论沉淀：先剖析再动手；每轮改动以「固定种子贪心轨迹 hash + 全套测试」护栏。
