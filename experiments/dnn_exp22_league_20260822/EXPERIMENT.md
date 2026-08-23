# exp22：纯谱系对手联赛——防守涌现的生态手术

- **Date**: 预注册 2026-08-22  **Status**: launching
- **Git**: （发射时记录）  **Env**: mahjong-dnn-c3（us-east1-b），**首个 `--gpu_infer` 云端 run**
- **对照**: exp17-C（同协议镜像自对弈，1079.7；defense_iq 0.011）
- **设计**: docs/design_league_exp22.md

## Purpose & Hypothesis
exp21/20/23 已排除感知、信用、采样、输入、攻击饱和五项嫌疑，唯一剩余解释=生态均衡：镜像种群
里推牌近似最优、折牌无回报。假说：把训练对手换成**自己谱系的冻结历史池**（风格跨度立直 3→27%、
副露 37→88%、强度 944→1080），威胁多样性与惩罚真实度上升 ⇒ **折牌首次有正回报 ⇒ 条件折牌涌现**。
纯度：池内 7 个模型全部零教师知识（bcrl14/bc_* 不入池，只留评分池）；不改奖励/目标。

## Method / Config
exp17-C 协议（cnn_m + GAE 0.95 + 熵台阶 0:0.03,600000:0.01 + 700k + seed 42）
+ `--league experiments/league_pool_v1/pool.json --league_frac 0.5` + `--gpu_infer`。
对局构成：每副牌山按种子确定性决定镜像/联赛（50/50）；联赛局学习者占 1-2 座、其余座位从池均匀
采样；8 复本同分配（组基线键不变）；对手轨迹丢弃（学习样本约为镜像的 ~62%）。池静态（v1，
无自快照刷新——作为后续变量）。

## Success Criteria（发射前定死）
1. **防守判定（主）**：defense_iq ≥ +0.03（800 局探针，SE≈0.015）且曝露放铳率 < 0.15（基线 0.19-0.22）
   ⇒ 防守首次涌现。
2. **强度不退化**：正式评分 ≥ 1079.7 − 2SE（联赛可能牺牲镜像强度；显著低于则记为代价）。
3. **可利用性粗测**：对池内各对手 100 副对打的点差分布；对最强对手（exp17c）不应显著为负。
4. 吞吐 ≥ 30 局/s（对手 CPU 推理 + 学习者 GPU 服务的混合路径）；健康同标准。

## Progress
- [2026-08-22] 预注册；本地 smoke（8 deals × 4 replicas）：学习者座位独占轨迹、联赛比例 ~0.5、
  镜像路径贪心 hash 不变；池上传 GCS `league_pool_v1/`。

## Results
| Metric | This run | Baseline (exp17-C) | Success criterion |
|---|---|---|---|
| （待运行） | | | |

## Conclusion
（待运行）

## Artifacts
| Path | Description |
|---|---|
| gs://llm-mahjong-experiments/dnn_exp22_league_20260822/ | 云端主目录 |
| gs://llm-mahjong-experiments/league_pool_v1/ | 对手池（7 个纯谱系 checkpoint + pool.json） |
