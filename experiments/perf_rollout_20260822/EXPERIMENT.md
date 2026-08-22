# perf：rollout 引擎优化（scale-up 前置工程，用户指令 2026-08-22）

- **Date**: 2026-08-22  **Status**: running
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

## Results
（待运行）

## Conclusion
（待运行）
