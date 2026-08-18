# exp18：cnn vs vit 全同配置从零配对对照（架构效应的干净测量）

- **Date**: 预注册 2026-08-17 ~23:25 本地  **Status**: launching
- **Git**: 6620305 + 本预注册
- **Env**: 双 g2 on-demand——cnn 臂 mahjong-dnn-c3（us-east1-b）、vit 臂 mahjong-dnn-c5（us-east1-c）
- **臂目录**: `dnn_exp18_cnn_20260817` / `dnn_exp18_vit_20260817`

## Purpose & Hypothesis
用户设计：exp15 的「vit 天花板低」结论被熵调度混杂污染（e700 有 36 万局 0.03 中段探索，
exp15 在 240k 提前降熵）；历史上 cnn/vit 从未在同一配方下从零对照过（exp10 毕业赛两臂
同为 reuse44 但只到 240k）。本实验从零训练**除 --arch 外逐字节相同**的两臂，一次性回答：
①架构对渐近线（700k）的真实效应；②exp10 的 240k 优势在现代配方下是否复现；
③熵台阶（600k→0.01）在两种架构上的红利是否同量。

## Method
两臂唯一差异 `--arch`（默认 cnn_m 1.94M vs vit_small 0.82M）。同种子同调度同机型。
调度=现代最优共识：ppo_epochs=1 全程（避开 reuse44 的 240k 后倒退风险）+
熵台阶 0.03→600k 时 0.01（e700 相位结构的忠实复刻）。双臂全程 ladder。

## Config（两臂共享）
`--total_games 700000 --games_per_iter 2048 --dup_k 8 --workers 30 --lr 1e-4
--entropy_coef 0.03 --entropy_schedule 0:0.03,600000:0.01 --value_coef 0.5 --clip_eps 0.2
--ppo_epochs 1 --target_kl 0.03 --batch 8192 --drop_zero_return --train_device cuda
--ckpt_every 10 --milestones 20000,80000,240000,400000,600000,700000 --seed 42`
（cnn 臂不带 --arch；vit 臂 `--arch vit_small`；无 GAE，adv_clamp 默认 5）

## Success Criteria（发射前定死）
1. **主判定**：700k vs 700k 配对竞技场 200 副；显著方向 = 架构渐近线结论（任一方向都定论）。
2. **早期复检**：240k vs 240k 竞技场 200 副——exp10 的 vit 优势（当年 +1633）在
   ppo_epochs=1 配方下是否复现。
3. **台阶红利对比**：两臂 600k 前后的 ladder 增量（熵台阶是否架构无关）。
4. 健康：KL<0.03、熵地板 0.4、无 NaN、双臂均跑满 700k。

## Progress
- [2026-08-17 23:25] 预注册；双机启动发射中。

## Results
| Metric | cnn 臂 | vit 臂 | Success criterion |
|---|---|---|---|
| （待运行） | | | |

## Conclusion
（待运行）

## Artifacts
| Path | Description |
|---|---|
| gs://llm-mahjong-experiments/dnn_exp18_{cnn,vit}_20260817/ | 云端主目录 |
