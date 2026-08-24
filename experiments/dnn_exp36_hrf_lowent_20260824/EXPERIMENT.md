# dnn_exp36_hrf_lowent_20260824 — HRF × 恒定低熵、长视野、bf16+amp 提速（从零）

- **Date**: 2026-08-24  **Status**: launching（先验证吞吐，过关后 --resume 续到正式长度）
- 动机：b2_4/b2_4ext（handset 恒定 0.01）展现自组织分阶段探索 + 二次转型，T1 已破 1044；
  假设该性质随表达力增强（容量/混叠假说：expressivity 抑制棘轮全局化，[[handset-constant-entropy-recipe]]）；
  HRF（23.2M，8 层交叉注意力，比 handset 更强的局面个体化能力）是下一格验证。
  用户要求：发射前先优化 rollout+训练效率——本地已验证 bf16_infer（rollout 本地 1.12×，云端待测）
  + amp_update（更新步矩阵乘法本地 B=512 下 1.51×，云端待测，本地因显存不足无法测真实 batch=4096）。

## Method
`--arch hrf_xl_v4 --entropy_coef 0.01`（恒定，无 schedule）+ `--bf16_infer --amp_update`；
其余同冠军配方；对照 = exp30（同架构、冠军 schedule，同纪元，24.0 games/s 已知基线）。

**两段式发射**：
1. 验证段：短 total_games（如 20000），读 `rollout_s`/`update_s`/`games_per_sec` 打点，
   确认吞吐相对 exp30 fp32 基线（24.0/s）的实际提升、无 NaN/数值问题。
2. 正式段：验证通过后 `--resume` 同一 checkpoint，`--total_games` 改为 2,000,000（长视野，呼应
   b2_4ext"自组织平台需要预算才能变现"的教训），games 计数器不清零——同一条从零血统连续训练。

## Success Criteria（正式段）
1. 吞吐：games/s 相对 exp30 fp32 基线有实测提升（数字待验证段给出，不预设倍数）。
2. 复现自组织现象：熵是否出现 ≥0.4 的自持平台、是否出现类似 b2_4ext 的二次转型（熵回升 + 副露/立直再平衡）。
3. Elo：T1 vs exp30（冠军 schedule 对照）；长视野后再挑战 exp31-5（cnn_m 冠军配方 1031.9）。
4. 数值健康：amp_update 全程无 NaN/inf，approx_kl 与 fp32 时代量级一致。

## Progress
- 2026-08-24 验证段完成（20000 局，10 迭代）：**39.1 games/s，比 exp30 fp32 基线（24.0）快 1.63×**。
  拆分：rollout_s 稳定 ~12.1s/iter，update_s 稳定 ~27.0s/iter（收益主要来自 amp_update，更新步本就是大头）；
  approx_kl 全程 0.0015–0.0049（远低于 0.03 目标线），无 NaN，无漂移。另有 ~13.3s/iter"其他"开销
  （GAE/日志/ckpt 保存）未被打点覆盖，记为后续优化线索，不阻塞本次发射。
  验证机 mahjong-g4-e36 命中 total_games 后自动关机（预期行为）；ckpt 转存
  gs://llm-mahjong-experiments/resume/dnn_exp36_hrf_lowent_validation_20k.pt，删除旧机。
- 2026-08-24 正式段发射（mahjong-g4-e36full，同 SHA ec117a0，--resume，games=20480 起，
  total_games 2,000,000，milestones 100k/500k/1M/1.5M/2M）。
