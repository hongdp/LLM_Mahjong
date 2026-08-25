# exp40（预注册）— Mortal 型 backbone 对照：深度 SE-ResNet 能否在我们的从零配方下存活

- **Date**: 2026-08-24  **Status**: designed（等 exp37/38 释放机器后发射）
- **Cost**: 每臂 G4 flex ~$15（1.0M 局）

## 出处与纯度声明
Backbone 转写自 Equim-chan/Mortal `mortal/model.py`（`ResNet` / `ResBlock` / `ChannelAttention`），
尺寸取自其 `config.example.toml`（`conv_channels=192, num_blocks=40`）。
**只借架构，不借任何人类先验**：不用人类牌谱、不用 CQL/DQN 目标、不用 GRP 奖励网、不用 dueling head；
head 与编码器仍是我们自己的（ACTION_DIM=374 策略头 + 标量价值头）。
架构不是教师知识 ⇒ 从零自对弈训练的该臂**留在纯血谱系内**（对照 [[mahjong-alphazero-purity]]）。

## 动机
我们的 scale 尝试连续失败（cnn_xl 6.6M→851.7、HRF 23.2M→852~896），而 Mortal 证明了
**14M 级、40 层深的 ResNet 在麻将上可以很强**。这让它成为高信息量的对照：
- 若 Mortal 型 backbone 在我们配方下**也崩** ⇒ 问题在**训练配方**，不在我们挑的架构；
- 若它**活下来** ⇒ 我们此前的大模型设计本身有问题（深度/归一化/SE 的组合才是关键）。

关键观察：我们的 `cnn_xl_r` 与 Mortal **同宽（192ch）**，只是 **6 层 vs 40 层**、
**无 SE**、**后激活**——然后崩了。三个差异正是本实验要拆开的。

## 已实现（本次提交）
`src/agents/dnn/arch_zoo.py`：`SEChannelAttention` / `MortalResBlock`（预激活 + SE，
BN momentum 0.01 / eps 1e-3 同 Mortal）/ `MortalBackbone`（stem → N×block → BN+Mish →
Conv(32ch) → Flatten → Linear(1024) 颈部，同 Mortal；之后接我们的 head/value）。
注册：`mortal_bb_xl_r`（192×40，11.2M）、`mortal_bb_m_r`（192×6，3.5M，深度对照臂）。
已通过前向 + 训练循环冒烟测试。

## 臂设计（单变量可拆）
| 臂 | arch | 对照的问题 |
|---|---|---|
| A | `mortal_bb_xl_r`（192×40） | Mortal 型 backbone 在我们配方下能否存活 |
| B | `mortal_bb_m_r`（192×6） | 与 `cnn_xl_r`（192×6，无 SE/后激活）同深同宽 → **隔离出"预激活+SE"的贡献** |
配方：冠军 schedule（与 cnn 线可比）+ `--lr 6e-5`（exp37 的深栈步长结论）+ warmup 150
+ `--bf16_infer --amp_update`，1.0M 局，seed 42。

## 未控制的变量（诚实声明）
Mortal 输入是 **1012×34** 特征平面（`libriichi::consts::obs_shape(4)`），我们只有 **21**（v1r）——
**差约 48 倍**。本实验只换 backbone，特征保持我们的，因此**不能**用它回答"Mortal 强是否主要靠输入表示"。
那是独立的一支（候选 exp41：特征扩展），必须分开做，否则归因不清。

## Success Criteria
1. **主判据**：A 臂 T1 > 851.7（`cnn_xl_r`，同宽度的我们版本）→ 深度 SE-ResNet 在我们配方下可训练；
   若 A ≥ 1031.9（cnn_m 冠军）→ scale 路线复活，优先级最高。
2. **归因**：B 臂 vs `cnn_xl_r`（同宽同深，唯一差异=预激活+SE）→ 分离 block 设计 与 深度 的贡献。
3. 若 A、B 双双 ≲850 ⇒ **判定问题在训练配方而非架构**，后续重心转向配方（样本效率/目标函数），
   不再试新架构。
