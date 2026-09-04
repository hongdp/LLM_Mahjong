# exp64 全量预训练 → 高分席位微调：BC 能否超过"示范者平均"

- **Date**: 2026-09-04  **Status**: running
- **Git**: 见 Progress；RunPod Secure L40S（$1.09/h），数据 = exp62 同一份 20,526 局快照（tenhou_raw.tar.zst），pod 零凭证
- **Env**: `train_human_bc.py --init --seat_min_rate --max_updates`（本实验新增），流式回放（无需物化）

## Purpose & Hypothesis
文献（docs/literature_review_2026-09.md §3）：BC 落在示范者分布的"平滑平均"之下；AlphaStar Unplugged 的同数据配方
"全量预训练 → 高 MMR 子集微调" +105 Elo（84%→89% 胜 very_hard），只过滤不预训练反而 84%→65%。
我们的凤凰卓数据里各席 R 值分布：中位 2160，**R≥2200 占 25.4% 席位**，R≥2150 占 56%。假设：以 bc49 为起点、只在 R≥2200 席位的
决策上微调，T=0 配对头对头对"同步数全席位微调"的控制臂 ≥ +1%，holdout 精度不降。反假设：凤凰卓内部技能差太小
（ILEED 同质 split 零增益），微调只带来 exp62 量级的噪声。

## Method（同批控制臂 + 双种子，按 exp62 教训）
四个 run 同一 pod 并行，全部 `--init bc49.pt`，lr 1e-4 const（原配方 1/3），batch 1024，bf16，early-stop 关（固定步数）：
- **T1/T2（实验）**：`--seat_min_rate 2200`（≈ 3.0M 决策/epoch），3 epoch ≈ 9k 更新，seed 1 / 2；
- **C1/C2（控制）**：全席位，`--max_updates` = T 臂实际更新数（步数匹配），seed 1 / 2。
每 epoch 记录全 holdout acc（1000 局全席位）。终评（工作站）：每个 T 对每个 C 的 n=4000 配对牌山 T=0（4 个交叉对），
汇总为 T 均值 vs C 均值；T/C 各对 bc49 原版 n=4000；A/A 对照。若 T−C 合并 ≥ +1.5σ 再补 n=16000。

## Config
- 数据快照 = exp62 同一 tarball；holdout 同 hash 规则（1000 局，全席位）；R 阈值 2200（top quartile）。
- 预算：物化省掉，流式 13 核 4 个 run 并行 ≈ 2 h ≈ **$2.5，上限 $5**。

## Success Criteria（预注册）
1. **主判据**：T（2 种子均值）vs C（2 种子均值），配对牌山 T=0，合并 n≥16000 对 share **≥ 0.51 且两个种子同号** → 高分席位微调成立，
   进入"数据 10× + 高分微调"组合（exp65 的默认收尾步骤）；0.50–0.51 → 噪声地板内，判平；<0.495 → 有害。
2. 精度：T 的全 holdout acc 不低于 C −0.3pp（微调到子集不能毁全局模仿）。
3. 诚实条款：T 与 C 步数匹配、同 init、同 lr；不做事后选 epoch（取最后一个 epoch 的权重）。

## Progress
- [09-04 01:07] git fb113b1。本机冒烟（300 局、25 更新）：热启动 0 键跳过、席位过滤 320/1200 单元、预算停止生效。
  pod `m9chauvfbxturo` L40S Secure US-TX-4（$1.09/h），bootstrap 2 分钟。**假发射**：repo.tar 不含 gitignore 的
  `experiments/_anchors_epoch6/bc49.pt` → 四臂 3 秒内 FileNotFoundError，心跳把 TRAIN_DONE 误判为完成。补传权重后
  01:10 重发：T 臂席位单元 18,792 / 73,832（25.5%），四臂并行各 3 worker（cgroup 13.6 核）。教训入 SKILLS：pod 归档要显式带锚点。

## Results

## Conclusion

## Next Steps

## Artifacts
