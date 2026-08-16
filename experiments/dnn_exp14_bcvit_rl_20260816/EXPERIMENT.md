# exp14：BC warm-start + 自对弈 RL（教师先验的价值定量）

- **Date**: 预注册 2026-08-16 ~16:20 本地；发射待机器（zone 猎手拿到第二台 g2 即发）  **Status**: pre-registered
- **Git**: 394d592 + 未提交预注册
- **Env**: GCP g2-standard-32 on-demand（L4）
- **对照**: bc_vit = `arch_sweep/models/vit_small.pt`（起点，BC 保真度 89.3%）；
  vit240 / exp15 里程碑 = vit 从零谱系同局数点

## Purpose & Hypothesis
Elo 校准发现 bc_cnn 居全池第 2：纯自对弈烧 700k 局才超过简单教师模仿。假说：教师先验
（牌效率+立直价值）可免掉 ~240k 局「规则税」——BC warm-start 的 RL 在同局数处应显著
强于从零 RL，且终点更高。2026-08-16 用户决策：用 transformer（vit_small）谱系。
先例：8/14 本地小探针（cnn BC + entropy 0.02，19k 局）立直率保持 ~0.8/局——先验可在
温和熵下存活（该探针无档案，本记录即其教训的归档）。

## Method
- `--init arch_sweep/models/vit_small.pt --arch vit_small`（warm-start，games 从 0 计）。
- 配方 = 冠军定稿：ppo_epochs=1；**entropy_coef 0.01**（依据台阶律：BC 起点属「已具技能」
  状态，等价饱和后阶段；0.03 有洗掉先验坠入 88% 副露均衡的风险——此为预注册决策，
  若先验仍流失按判据 3 如实记录）。600k 局。
- 全程 ladder watch 出 Elo 曲线。

## Config
`--arch vit_small --init <bc_vit.pt> --total_games 600000 --games_per_iter 2048 --dup_k 8
--workers 30 --lr 1e-4 --entropy_coef 0.01 --value_coef 0.5 --clip_eps 0.2 --ppo_epochs 1
--target_kl 0.03 --batch 8192 --drop_zero_return --train_device cuda --ckpt_every 10
--milestones 20000,80000,240000,600000 --seed 42`

## Success Criteria（发射前定死）
1. **核心单变量判定**：240k 里程碑 vs vit240（同架构同局数，唯一差异=起点），200 副复式。
   显著正 = 教师先验加速确认；null/负 = 规则税假说削弱（同样有信息量）。
2. **增值判定**：600k 终点 vs bc_vit 起点，200 副，显著正（RL 在先验之上加了东西）。
3. **先验存活探针**：ladder 曲线不得出现深 V（先掉 >1000 分再爬回=先验被洗掉重学）；
   风格上 20k/240k/600k 处立直率保持 >0.3/局（8/14 探针水平的近似下界）。
4. 训练健康同 exp15。

## Progress
- [2026-08-16 ~16:20] 预注册。bc_vit checkpoint 需先上传 GCS 供 VM 拉取。

## Results
| Metric | This run | Baseline | Success criterion |
|---|---|---|---|
| （待运行） | | | |

## Conclusion
（待运行）

## Artifacts
| Path | Size | Description |
|---|---|---|
| gs://llm-mahjong-experiments/dnn_exp14_bcvit_rl_20260816/ | — | 云端主目录 |
