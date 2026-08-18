# exp18：cnn vs vit 全同配置从零配对对照（架构效应的干净测量）

- **Date**: 预注册 2026-08-17 ~23:25；终判 2026-08-18  **Status**: done（cnn 渐近线胜出，exp10 结论修订）
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
- [2026-08-17 23:40] 双臂发射确认：cnn 首 iter win_rate 0.007、vit 0.020，熵调度 0.03
  生效，双 ladder 守护已挂；双跑满哨兵已挂（触发后自动跑 700k/240k 两组配对竞技场）。

## Results
| Metric | cnn 臂 | vit 臂 | Success criterion |
|---|---|---|---|
| **主判定：700k 对打（seed0=20260824）** | **胜（197:97）** | **−2000 ± 1151** | 显著 ⇒ **cnn 渐近线显著更高** |
| 早期复检：240k 对打（seed0=20260825） | — | +471 ± 907 **null** | exp10 的 +1633 **未复现** |
| 正式评分（全池 ×100 副） | **1013.4 ± 12.9** | 928.3 ± 12.7 | cnn 与 e700（1012.5）逐分重合 |
| 训练健康 | 双臂 700k 跑满、调度生效、无坍缩 | 同 | ✅ |

（判据 3 台阶红利对比受损：cnn 侧 ladder 守护在 235k 因 gsutil 超时未捕获而崩，cnn
曲线只到 194k——600k 前后增量仅 vit 侧可比，vit 台阶红利不明显（曲线在 600k 前已平台）。
守护 bug 已修。）

## Conclusion
1. **架构渐近线差异为真，cnn 高 ~85 Elo（受控条件）**：exp15 的结论洗清熵调度混杂嫌疑。
   vit 四条独立 run（exp15 951 / exp16 923 / 本臂 928）收敛于 ~930 平台；cnn 渐近线 ~1012。
2. **exp10 毕业赛结论被推翻（配方依赖幻象）**：当年 vit+1633 赢的是 reuse44 弱配方下的
   cnn-240k（Elo 884）；ppo_epochs=1 把 cnn 同点抬到 ~912 后优势归零（+471 null）。
   「BC 保真度优势 → RL 早期优势 → 受控归零」链条完整。vit 真正的遗产=早期爬升快
   （首 iter win_rate 3×、~80k 即达 880+），适合短预算场景。
3. **配方渐近线可复现**：exp18-cnn 从零单 run 1013.4 vs e700 血统的 1012.5——
   「cnn + reuse11 + 熵台阶 ≈ 1012」为稳定常数。主线 backbone 回归 cnn。
4. 下一爬升假说仅剩 exp17-C（cnn+GAE，A0=1022 线索）。

## Next Steps
- exp17-C：e700 配方 + GAE 单变量（待用户确认发射）。
- vit 谱系归档；exp10 EXPERIMENT.md 补充「结论被 exp18 修订」交叉引用。

## Artifacts
| Path | Description |
|---|---|
| gs://llm-mahjong-experiments/dnn_exp18_{cnn,vit}_20260817/ | 云端主目录 |
