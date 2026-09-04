# exp62 花色置换增广 BC A/B：训练时对称增广能否兑现样本效率

- **Date**: 2026-09-03  **Status**: done
- **Git**: 见 Progress 首条（worktree `runpod-training-cost-optimization-626ab2`）
- **Env**: RunPod **Secure** L40S（$1.09/h，13 核；产物为冠军谱系权重 → 按安全分层走 Secure）；
  数据 = 凤凰卓 20.9k 局 `data/tenhou/raw`（scp 上 pod，pod 上零云凭证）；pod 上物化 v3r/mortal46 缓存后训练

## Purpose & Hypothesis
exp61 已证：bc49 对花色置换的破缺（5.7%）全在近平决策，测试时对称平均零效应。剩下的问题是**训练时增广**：
exp49 表明 BC 处于数据受限区（9× 数据 = +3.7–4.0pp 精度 = +27–57 Elo，18.4k 局未饱和），花色增广给每个样本
6 个等价视角（相关但不重复），相当于 2–3× 有效数据。假设：增广臂 holdout 精度 **+0.5pp 以上**，T=0 头对头对控制臂显著为正。
反假设：ConvFormer 的注意力桶偏置已经足够利用花色结构，增广只带来 <0.2pp 的噪声级差异。

## Method
同 [docs/champion_model.md §5](../../docs/champion_model.md) 的 bc49 配方（`convformer_m_v3r_m46`，全量非留出局，holdout 1000 局，
max_epochs 30 / patience 3 / min_delta 5e-4，batch 1024，lr 3e-4 const，seed 0），两臂在同一 pod、同一物化缓存上并行：
- **C（控制）**：原配方原样重跑（同时是 bc49 = 0.8059 的跨硬件复现检查，给出"硬件/非确定性噪声地板"）。
- **A（增广）**：`--suit_aug`：每个训练 batch 切 6 块，第 k 块施加第 k 个花色置换（planes 牌轴 / mask 槽位 / 标签一致置换，
  `src/agents/dnn/symmetry.py::make_batch_augmenter`）；手牌+副露绿牌 >7 张的样本保持恒等（绿一色守卫）。holdout 评测恒等。
增广是 exp61 基建的直接复用（6 项单元测试 + 一项增广一致性测试 + 本机真数据冒烟）。

## Config
- 缓存：pod 上 `materialize_bc.py --variant v3r --action_space mortal46 --workers 12`（本机 v3r2 缓存 31GB 无法上传）。
- 两臂并行各 `--workers 5`，输出 `/workspace/exp62_{ctrl,aug}`；每 epoch 存 best ckpt + metrics.json + TB。
- 终评（工作站，0 元）：A vs C 配对牌山单局 T=0 n=4000 对（+ A/A）；A vs bc49、C vs bc49 各 n=4000；破缺率诊断（预期 A → <1%）。
- 预算：约 2.5 小时 ≈ **$2.7，上限 $5**（含 pod 卡死重开）。

## Success Criteria（预注册）
1. **主判据（强度）**：A vs C，T=0 配对牌山 n=4000 对，share **≥ 0.52（+2.5σ）** → 增广兑现样本效率，进入下一步
   （数字翻转增广 + 用增广重训冠军配方作为加冕候选）。0.50–0.52 → 精度端看判据 2 再定；<0.49 → 增广有害（记录）。
2. **精度判据**：A holdout acc − C holdout acc ≥ **+0.5pp**（exp49 汇率 ≈ +5 Elo/… 0.5pp 是 seed 噪声地板的 ~3 倍）。
   |C − bc49 0.8059| > 0.3pp 时复现检查亮黄灯，A/B 仍以 A−C 为准。
3. **对称性判据**：A 的破缺率 < 1%（不满足说明增广未生效，A/B 结论作废）。
4. 诚实条款：早停规则两臂同；比较用各臂 best ckpt；不做事后选 epoch。

## Progress
- [09-03 20:30] git be89308（`--suit_aug` 入训练器；symmetry 7/7 测试；本机 v3r2 缓存 3 批真数据冒烟：34.8% 标签被置换、
  置换后标签全部合法）。**pod 建立**：`s8uslxwh2rasoq` L40S Secure US-TX-4，16 vCPU，$1.09/h，直连 195.26.232.163:38455，
  建后 36 秒端口就绪。3090 secure 仅 EU-CZ-1 且 LOW，未选。预计 ≈2.5h ≈ $2.7（上限 $5）。
- [09-03 20:33] scp repo.tar（2.6MB）+ tenhou_raw.tar.zst（133MB，家用上行）+ 脚本 → pod → bootstrap（apt zstd、pip、解包）
  → `exp62_pod_train.sh`（物化 v3r/mortal46 缓存 → 两臂并行）。
- [09-03 20:45] 物化 11.7 分钟（12,031,681 训练行 / 1,341,797 留出行 / bad 0，11 worker）；两臂并行发射，GPU 99%，
  控制臂 25.7k 样本/s、增广臂 22.1k 样本/s（置换开销 ~15%）。cgroup 实际 13.6 核（宣称 16 vCPU）。
- [09-03 22:33] **控制臂早停**（13 epoch，107.7 分钟）：best 0.8075@ep11（ckpt 0.8073@ep9，min_delta 门槛）；bc49 原版 0.8059 → 复现 +0.16pp 绿灯。
- [09-03 22:55] **增广臂早停**（14 epoch，129.4 分钟）：best 0.8080@ep10。两臂逐 epoch 曲线几乎重合（差 ≤0.1pp）；增广臂训练损失恒高 0.004。
- [09-03 23:00] 最后一拉（ckpt/metrics/TB/日志 → 主检出 `exp62_{ctrl,aug}` + GCS），pod **terminate（204）**，实际 ≈2.5h ≈ **$2.7**。
- [09-03 23:10] 工作站终评：n=4000（seed 67M）+ 补充 n=16000（seed 68M）；破缺率诊断两臂各 200 局。

## Results

| 判据 | 目标 | 实测 | 判定 |
|---|---|---|---|
| 1 主判据：A（增广）vs C（控制），配对牌山 T=0 | ≥0.52 | n=4000 **0.5119±0.0079**（+374 分/对）；n=16000 **0.5002±0.0040**（+24）；**合并 20k 0.5025±0.0035**；A/A 0.500 | ❌ 落在 0.50–0.52 档 → 看判据 2 |
| 2 精度：A − C holdout acc | ≥ +0.5pp | 0.8080 − 0.8075 = **+0.05pp**（ckpt 口径 0.8080 − 0.8073 = +0.07pp）；切牌 0.7735 vs 0.7734；CE 0.4971 vs 0.4991 | ❌ |
| 3 对称性：A 破缺率 | <1% | C **6.01%** → A **2.78%**（视角不一致 2.87%→1.32%；均值改选 1.93%→0.91%） | ⚠️ 减半但未达 <1%：增广只部分生效（每批 1/6 恒等 + 13 epoch） |
| 复现检查：C vs bc49 | \|Δacc\| ≤0.3pp | acc +0.16pp；头对头 n=4000 0.5081、n=16000 0.5053、**合并 0.5059±0.0035（+1.7σ）** | ✅ 精度；强度上同配方重跑比 bc49 高 ~0.6%（噪声地板见结论） |
| A vs bc49 | — | n=4000 0.5111、n=16000 0.5085、**合并 0.5090±0.0035（+2.6σ）** | 显著但 A−C 只 +0.25%：主要是"重跑效应"不是增广 |

两臂精度分桶（best epoch）：切牌 0.7734/0.7735、立直 0.760/0.794、鸣牌 0.770/0.754、和牌 1.000/0.999、pass 0.967/0.971、防守 0.8171/0.8173。

## Conclusion

**训练时花色增广在 18.4k 局配方上无可测增益**：精度 +0.05pp，头对头合并 20k 对 0.5025±0.0035（+0.7σ），与 exp61 的测试时零效应、
以及文献预期（发牌本身花色可交换 → 数据分布已花色不变；卡牌类系统一律用规范化而非增广，无人报告正增益）一致。
增广确实改变了模型（破缺率 6.0%→2.8%、训练损失恒高、立直精度 +3.4pp / 鸣牌 −1.6pp 的桶间搬动），但这些变化不在"胜份"上兑现。
**花色对称这条线到此关闭**（exp61+exp62 两个数字，都是麻将文献里的第一个）。

意外收获——**同配方重跑的噪声地板**：控制臂 = bc49 配方原样重跑（同 seed、同数据、L40S bf16 + 物化缓存），头对头对 bc49
0.5059±0.0035（+1.7σ），增广臂对 bc49 0.5090（+2.6σ）而对控制臂只 0.5025。即两次重跑相对原版都偏高 0.5–0.9%，
而 A−C 之差只有 0.25%。**训练级随机性（数据顺序/硬件）在胜份上是 ±0.5–1% 量级，与我们要找的效应同阶**。
结论：今后任何 BC/RL 的 A/B 必须（a）控制臂与实验臂同批重训而不是拿历史 ckpt 当控制；（b）效应 <1% 时要多种子，
配对牌山只能消评测噪声、消不掉训练噪声。这也解释了 exp46/59/60 里若干 ±0.5% 的"信号"。

## Next Steps
- 关闭花色增广/TTA 线；`--suit_aug` 保留为可选正则项不默认开。
- 按文献对照（docs/literature_review_2026-09.md）转向：①数据 10×（20 万局凤凰卓）；②顺位 pt 刻度重评 Mortal vs bc49（0 元）；
  ③全量预训练→高分席位微调；④搜索→蒸馏试点。RL 在 critic 看隐藏信息 + 评测到 10⁵ 级之前不再烧。
- A/B 方法学入 SKILLS：同批控制臂 + 多种子。

## Artifacts
| Path | Size | Description |
|---|---|---|
| experiments/exp62_ctrl/ + gs://llm-mahjong-experiments/exp62_ctrl/ | 8MB | 控制臂 best ckpt（0.8073@ep9）、metrics.json（13 epoch）、TB |
| experiments/exp62_aug/ + gs://…/exp62_aug/ | 8MB | 增广臂 best ckpt（0.8080@ep10）、metrics.json（14 epoch）、TB |
| experiments/exp62_pull/*.log + gs://…/exp62_pull/ | — | pod 端 train_ctrl/train_aug/materialize/bootstrap/exp62 日志 |
| experiments/probes/exp62_arms_n4000.json / exp62_arms_n16000.json | 1KB | 终评（含 A/A） |
| experiments/probes/exp62_symmetry_{ctrl,aug}.json | ~400KB ×2 | 两臂破缺率诊断（逐决策记录） |
| scripts/train_human_bc.py `--suit_aug/--green_max`, src/agents/dnn/symmetry.py `make_batch_augmenter` | — | 增广实现 |
