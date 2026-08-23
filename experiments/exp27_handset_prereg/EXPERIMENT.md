# exp27 — 手牌集合模型（HandSet）：实例级无位置编码注意力 vs 34 轴 CNN（预注册）

- **Date**: 预注册 2026-08-23 03:40 PDT  **Status**: planned（发射条件：纪元 3 重校完成 + 云端 L4 吞吐基准 ≥ 25 局/s）
- **Git**: 分支 `engine/majsoul-rules`（本提交）：`HandSetEncoder` / `HandSetCnn`（arch_zoo）、`scripts/probe_decomposition.py`、v1r 役牌平面
- **Env**: GCP g2-standard-32（L4）×3，统一 GPU rollout 基建（`--gpu_infer`）；纪元 3 引擎（雀魂规则 + 赤宝牌）

## Purpose & Hypothesis
用户提出：把手牌当**集合**（牌实例 token、无位置编码）用 transformer 编码，是否比依赖 34 轴"局部顺序"的
CNN 更擅长复杂手牌拆分。既有证据：vit_small（34 牌型 token + 身份 embedding，等价集合+身份）输 CNN ~85 Elo；
ConvFormer 加回 rank 相对偏置后差距缩到 −14。因此假设拆成两部分：
- H1（实例级表示有用）：**实例 token**（同一张牌两个副本是两个 token）+ rank 相对偏置（内容相关、保持置换等变）
  的手牌分支，在拆分密集的决策上更优——探针 `probe_decomposition`（牌效最优弃牌一致率，按向听×单色块长度分层；
  冠军基线：总 78%，"一向听+长块"最弱 73%）先于 Elo 显出差异，Elo 不低于参数匹配的 CNN 对照。
- H2（规模）：用户判断 attention 需要 scale——分支 d=384×10 层×12 头（20M，L4 上 GPU 临界点）。
- H3（相邻偏置不可少）：去掉 rank 偏置的纯集合版显著更差（仅在 H1 有信号时跑）。
风牌/役牌：役牌平面已加入 v1r（冠军客风碰 40% vs 役牌风 52%，ConvFormer/vit 的注意力在 700k 局内并未自行形成
该对齐），所有臂同享，不作为变量。

## Method
四臂同配方（exp17-C 配方：PPO ppo_epochs=1, clip 0.2, GAE λ=0.95, dup_k=8, 熵 0.03→0.01, T=1），从零，
700k 局，编码 v1r，动作头 374，纪元 3 引擎，GPU rollout：
| 臂 | 架构 | 参数 | 角色 |
|---|---|---|---|
| A | `cnn_m_r` | 2.0M | 纪元 3 纯血基线（兼作新谱系起点） |
| B | `handset_xl_cnn_m_r` | 20.0M（分支 17.8M） | 主实验：实例集合 + rank 偏置 |
| C | `cnn_xl_r` | 6.6M | 规模对照（34 轴卷积再大无意义，取 192ch×6） |
| D | `handset_xl_pure_cnn_m_r`（待加） | 20M | 消融：无 rank 偏置（仅 H1 有信号时发射） |
发射前：云机 `scripts/bench_rollout_infer.py` 测 B 的吞吐 10 min（L4 首个实测校准，写入 perf 记录）。
每臂阶梯评分（纪元 3 锚点）、`probe_decomposition`（300 局）、`probe_defense`、风牌探针；A vs B 复式牌 1000 副。

## Config
`--arch <臂> --total_games 700000 --games_per_iter 2048 --dup_k 8 --gpu_infer --infer_max_batch 128 --warmup_updates 150`（B/C/D）；其余同 exp17-C 记录。

## Success Criteria
1. H1：B 的 `shanten1_long` 一致率 ≥ A + 0.05 **且** B 的 Elo ≥ C − 20（参数匹配对照不显著更强）。
2. 强度主判：B − A 复式牌 1000 副 CI 下界 > 0 ⇒ 集合模型成为新冠军候选（进入 O(3) 晋升）。
3. H3：D 比 B 低 ≥ 30 Elo ⇒ "相邻偏置不可少"成立。
4. 健康：B 吞吐 ≥ 25 局/s（否则先调批/worker，不改模型）；KL/熵正常。
5. 纯度：全部从零、无教师、对手=自身（纪元 3 无联赛池，先镜像自对弈）。

## Progress
- [03:40] 预注册。架构与探针本地测试通过（`tests/test_handset.py`）；冠军探针基线见上。

## Results
| 臂 | Elo（纪元 3） | 拆分探针 总/一向听长块 | defense_iq | 吞吐 |
|---|---|---|---|---|

## Conclusion
（待）

## Next Steps
（待）

## Artifacts
| Path | Description |
|---|---|
| gs://llm-mahjong-experiments/dnn_exp27_{A,B,C}_<ts>/ | 云端主目录（发射后填） |
