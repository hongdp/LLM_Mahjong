# ConvFormer：为击败 cnn 设计的注意力架构（exp19）

2026-08-18，源自用户要求「设计一个能够 beat cnn 的 attention 架构」。
上游证据：exp18 受控对照 cnn 渐近线高 vit_small ~85 Elo；归因分析（三条病根）见当日讨论。

## 设计原则
保住 vit 赢的，修掉 vit 输的——每个组件对应一条病根：

| exp18 病根 | ConvFormer 对策 |
|---|---|
| ①局部先验缺失（顺子/搭子=rank 轴 k3 局部模式，attention 要花容量重新发明相邻性） | **花色内共享卷积 stem**（k=3 两层，花色段内滑动、绝不跨 9m\|1p 假边界——这点比 cnn_m 的全轴卷积还干净；字牌走 1×1）+ **rank 相对注意力偏置**（Swin/T5 式：每头对 20 个桶学偏置——同花色 Δrank −8..+8 共 17 桶、跨花色、字牌对、global 对——相邻性变成查表，不再消耗容量） |
| ②容量差 2.4×（0.82M vs 1.94M） | d=160 × 6 层 × 5 头（head_dim 32）= **1.97M，与 cnn_m 1.94M 精确对标** |
| ③RL 优化娇气（vit 曲线反复冲高回落） | 全程 pre-LN + **策略头零初始化**（开局逐 token 均匀分布，实测起始熵=ln(合法数)）+ trainer 新增 `--warmup_updates`（线性 LR 预热，转体 RL 标配） |

保留的 vit 优势：per-token 8 类动作头（与 272 维动作空间精确对齐，BC 89.3% 与早期 3× 爬速的来源）、global token 承载标量、注意力通道为将来防守/对手建模留路由能力。

## 冒烟实测（2026-08-18）
1.97M 参数；forward/backward/act 通过；masked logits 正确；零初始化起始熵 3.69=ln(40)；
b1 延迟 3.47ms（vit_small 3.24 / cnn_m 0.64——参数 2.4× 但几乎不变慢，35×35 注意力矩阵太小）。

## 评估路线
1. **BC 保真度** ✅（2026-08-18 完成）：**90.5%，全 zoo 新纪录**（vit_small 89.3% /
   cnn_m 81.4%）；CPU b1 4.3ms；checkpoint `experiments/arch_sweep/models/convformer_m.pt`。
   注意 exp10 教训：BC 保真度不兑现强度——这只是及格线检查，判决在 RL。
2. **exp19 RL 配对**（预注册，待发射）：exp18 协议原样（从零 700k、熵台阶、同种子），
   convformer_m vs cnn_m。诚实声明：convformer 臂带 `--warmup_updates 1000`——这是
   「注意力系统包 vs 卷积系统包」的对照，不是纯架构消融（warmup 对 cnn 中性偏无用，
   对 transformer 是已知必需品；把它算进架构包是公平的系统级比较）。
3. 判定：700k 对打 + 240k 对打 + 双 ladder；赢=注意力路线复活并直通防守方向，
   输=「局部模式识别游戏里卷积先验不可替代」结案。

## 实现
`src/agents/dnn/arch_zoo.py`：`RelBiasBlock` / `_tile_buckets` / `ConvFormer`，
注册名 `convformer_m`；`scripts/train_dnn_ppo.py` 新增 `--warmup_updates`（默认 0 不影响任何现有 run）。
