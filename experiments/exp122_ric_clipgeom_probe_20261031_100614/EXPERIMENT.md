# exp122 — RIPO/RIC 裁剪几何探针：PPO 固定 ε 裁剪在纯血线训练里到底触发多少、对哪些 π_old 触发

- **Date**: 2026-10-31 10:06  **Status**: done 10:17（本地 GPU 探针，$0，4 迭代，不保存 ckpt）→ **RIC 对纯血线是空操作，方向关闭**
- **Git**: 本 worktree 8b9b129；训练器与模型代码取 `claude/continue-task-3da2e7` 分支 **02d0cbe**（exp113 时代的 `scripts/train_dnn_ppo.py`，含 `--fast_trunk/--bf16_rollout`；runpod worktree 1736dfe 的训练器没有这些旗标）；Rust 引擎 `riichi_rs` 本机安装即由该分支 10-21 构建
- **Env**: 本地 RTX 4080 16GB + 12 CPU worker（雀魂服务在跑，留核）；conda `rlhf_mahjong`，torch 2.12 cu130
- **提议人**: 用户（2026-10-31）："RIPO 对我们的工作会有帮助吗" → "量一次"

## Purpose & Hypothesis
RIPO（Cai et al., ICML 2026, arXiv 2607.10169）的论点：PPO-Clip 的固定 |r−1| ≤ ε 对低概率动作太紧（π_old=0.01 只能涨到 0.012）、对高概率动作太松，
导致罕见但有价值的动作长期被压 → 探索塌缩；修正 RIC：ε(s,a) = √(δ/π_old)，每个 (s,a) 消耗相同的 KL 几何预算。
对纯血线的先验判断（见 FINDINGS 对照）：exp113 C 每步 approx_kl ≈ 0.0017、`ppo_epochs=1`、lr 按 KL 目标 0.001 自适应、熵稳定 0.27 nat 不塌缩
⇒ 固定裁剪几乎不触发，RIC 的重分配是空操作。本探针把"几乎"量成数字。

**H0（预期）**：在 exp113 C 384M 续训的真实更新批次里，固定 ε=0.2 裁剪的**有效触发率**（PPO min 真正切换到裁剪分支的样本比例）在所有 π_old 分层
都 < 1%；RIC（δ=0.02/0.05/0.08）边界的触发率同量级或更低；低概率层（π_old<0.01）的 |r−1| 分布与高概率层无质的差别。
**H1（RIPO 有戏的最低条件）**：π_old<0.01 层的固定 ε 有效触发率 ≥ 5% 且显著高于 π_old>0.1 层，即低概率动作确实被裁剪系统性压制。

## Method
- 训练器副本（scratchpad `train_dnn_ppo_clipprobe.py` = 02d0cbe `scripts/train_dnn_ppo.py` + 探针补丁，补丁存本目录 `clipprobe.patch`），
  `LLM_MAHJONG_ROOT` 指向 02d0cbe 的临时 detached worktree（scratchpad，用完 `git worktree remove`） 以加载 `cnn_l_v3r` 与 Rust 引擎。
- 从 `experiments/exp113_C/games_384000000.pt` 续训（恢复 Adam 矩，模拟 8M 段中段），**exp113 C 原配方逐字**：
  `--engine rust --arch cnn_l_v3r --games_per_iter 2048 --dup_k 8 --games_per_worker 1024 --batch 4096 --ppo_epochs 1 --lr 3e-5
  --entropy_schedule 0:0.003 --gamma 0.995 --gae_lambda 0.95 --target_kl 0.03 --clip_eps 0.2 --amp_update --shaping --shaping_schedule 0:1.0,400000:0.0
  --fast_trunk --bf16_rollout`（本地不加 `--compile`，省编译时间；workers 12）。
- 探针挂点：minibatch 循环里 `ratio = exp(chosen − old_lp[sel])` 之后、`loss.backward()` 之前，`no_grad` 记录每个样本的
  π_old = exp(old_lp)、dev = |r−1|、优势符号；按 π_old 分层 {<0.01, 0.01–0.05, 0.05–0.1, 0.1–0.5, ≥0.5} × minibatch 序号桶 {0, 1–4, ≥5} 累计：
  n、Σdev、Σdev²、dev>{0.05,0.1,0.2,0.5} 计数、**固定 ε 有效触发**（a>0 ∧ r>1+ε 或 a<0 ∧ r<1−ε）、RIC 有效触发（ε_ric=√(δ/π_old)，δ∈{0.02,0.05,0.08}）、
  RIC 边界比固定边界更松的比例。minibatch 0 的 dev 纯粹是 rollout（bf16）与更新前向（bf16 autocast→fp32 logits）之间的数值误差——
  单独报，作为"非策略变化"的地板。
- 跑 4 次迭代（≈4×44 个 minibatch ≈ 70 万样本）后 dump JSON 并退出；不写 ckpt。
- 参数全部沿用，模型权重会被更新 4 步（这是测"真实更新几何"的必要条件），产物不保留。

## Success Criteria（发射前定）
- 判据按 H0/H1 写在 Purpose：π_old<0.01 层固定 ε 有效触发率 ≥5% 且 ≥ 2× 高概率层 → "RIC 对我们有可测空间"，再写 RIPO 式训练预注册；
  否则 RIPO 方向关闭，记入 FINDINGS 一行。
- 附报：各层 dev 均值/RMS、RIC 各 δ 的触发率、RIC 边界松于固定的比例（说明 RIC 在我们的 π_old 分布下主要是放松还是收紧）。

## Progress
- [10:06] 建目录、复制训练器到 scratchpad、写本文件。
- [10:08] runpod worktree 1736dfe 的训练器没有 `--fast_trunk/--bf16_rollout`（exp113 用的是更新的代码）→ 改用 `claude/continue-task-3da2e7` 02d0cbe 的训练器
  （detached worktree 到 scratchpad）；本机 `riichi_rs` 即该分支 10-21 构建。补丁 5 块中 2 块手工重打。
- [10:11] 冒烟（256 局 ×8，1 迭代）通过，统计结构正确。
- [10:13] 正式跑第一次**崩溃**：探针块里的局部变量 `dev` 遮蔽了 main() 的 torch.device，迭代 2 的 `torch.arange(..., device=dev)` 报错（冒烟只跑 1 迭代所以没暴露）。
  改名 `rdev`，日志存 `run_attempt1_crash.log`。
- [10:14–10:17] 正式跑：4 迭代、每迭代 2048 局 ×8 = 16384 局、n_eff 12.1–12.5 万、30–31 个 minibatch；KL/迭代 0.00128/0.00128/0.00143/0.00111（与 exp113 C 日志 0.0017 同量级）。$0。

## Results（490,713 个样本，122 个 minibatch，ε=0.2，单位 = 样本比例）
**π_old 分布**（所选动作在采样策略下的概率）：≥0.5 **86.2%**、0.1–0.5 11.5%、0.05–0.1 1.3%、0.01–0.05 0.8%、<0.01 0.2%。策略很尖（熵 0.27 nat）。

**固定 ε=0.2 的有效触发率**（PPO 的 min 真正切到裁剪分支，样本梯度归零）
| π_old 层 | 样本份额 | mean \|r−1\| mb0 → mb5+ | 有效触发（全部 mb） | 其中 mb0（更新前，纯数值噪声） | up / dn | RIC δ=0.02 | RIC δ=0.05 | RIC δ=0.08 |
|---|---|---|---|---|---|---|---|---|
| <0.01 | 0.2% | 0.138 → 0.149 | **13.6%** | **16.0%**（n=25） | 6.3% / 7.3% | 0 | 0 | 0 |
| 0.01–0.05 | 0.8% | 0.120 → 0.127 | 9.7% | 8.9% | 4.0% / 5.7% | 0.19% | 0.07% | 0.07% |
| 0.05–0.1 | 1.3% | 0.086 → 0.092 | 6.5% | 6.25% | 3.1% / 3.4% | 0.40% | 0.10% | 0.03% |
| 0.1–0.5 | 11.5% | 0.063 → 0.066 | 3.3% | 2.3% | 1.7% / 1.6% | 1.67% | 0.30% | 0.14% |
| ≥0.5 | 86.2% | 0.0071 → 0.0074 | 0.11% | 0.10% | 0.04% / 0.07% | 0.15% | 0.03% | 0.01% |
| **全部** | 100% | — | **0.66%** | — | — | 0.33% | **0.06%** | ≈0.03% |
- RIC 边界 √(δ/π_old) 在 π_old<0.5 的所有样本上都比 ε=0.2 **更松**（δ=0.02 时 100%）；只在 π_old≥0.5 上更紧（δ=0.02：0.149）。
- 优势符号比例各层 0.33–0.47，无异常。

**判读要点**
1. 字面 H1 成立：π_old<0.01 层有效触发 13.6% ≥ 5%，且 ≫ 高概率层 0.11%。
2. **但 mb0 对照把它否了**：mb0 是本迭代第一个 minibatch，网络还没更新一步，ratio 本应 ≡ 1；实测 <0.01 层 mean|r−1| 已 0.138、触发 16.0%，0.01–0.05 层 8.9%，与 mb5+（0.149 / 13.6%，10.0%）几乎相同。
   偏差来源是**数值**：rollout 用 bf16 推理写下 old_logprobs，更新端 bf16 autocast 前向 + fp32 log_softmax，对小概率动作的 log-prob 差 ~0.1–0.2。
   从 mb0 到 mb5+ 的增量（真正由策略更新造成的 ratio 漂移）只有 ~0.01 的 mean|r−1|、≤1 pp 的触发率，发生在合计 2.3% 的样本上。
3. 触发在 up/dn 两侧**对称**（<0.01 层 6.3% vs 7.3%）：不是"罕见动作想涨被压住"，而是对称的随机梯度丢弃。RIPO 描述的机制（罕见动作被系统性地少更新）在纯血线训练里不存在。
4. RIC 的实际效果 = 把 π_old<0.5 的裁剪放松到几乎不触发（δ=0.05 总触发 0.06%，是固定 ε 的 1/11），把 π_old≥0.5 的裁剪略收紧（0.11%→0.15%）。
   两边都在 0.1% 量级上挪动——对 1.3e-3/步的 KL 预算而言是空操作。

## Conclusion
- **RIPO/RIC 方向关闭**：纯血线没有探索塌缩（熵稳定、两头拧熵都是 ±1 pp），固定裁剪的有效触发 0.66%，其中绝大部分是 bf16 数值噪声而不是策略更新；
  RIC 只会把 0.66% 变成 0.06%。预期收益 ≈ 0，不写训练预注册。
- **顺带的工程发现**：`--bf16_rollout` 写下的 old_logprobs 与更新端前向在小概率动作上差 ~0.1–0.2 log 单位，使 ε=0.2 裁剪在更新前就对
  π_old<0.05 的样本随机丢弃 ~10–16% 的梯度（这些样本合计占 1%）。exp113 就是这样训出来的，没有证据说明它有害；如果以后再训，
  最便宜的修法是打包阶段用更新端同一精度重算一次 old_logp（成本 ≈ 现有的价值前向），让 mb0 的 ratio 回到 1。记入 SKILLS。
- 关于把 ppo_epochs 从 1 放开（exp68 第 3 轮：4 epoch 让 100 万局模型停在"见牌就碰"）：本探针显示单 epoch 下 KL 预算远未用满（0.0013 vs target 0.03），
  多 epoch 的问题当年也不是裁剪几何而是重用偏差；RIC 不改变这个判断。

## Next Steps
- 无训练后续。FINDINGS 追加一行；SKILLS 追加 bf16 old_logp 噪声的工程注记。
- 清理：scratchpad 的 detached worktree `ct_code` 在 exp123 跑完后 `git worktree remove`。

## Artifacts
| Path | Size | Description |
|---|---|---|
| `experiments/probes/exp122_clipgeom.json`（主 repo） | 20 KB | 分层 × minibatch 桶统计 dump（n、Σdev、Σdev²、阈值计数、固定/RIC 触发计数、_iter_kl） |
| `experiments/exp122_ric_clipgeom_probe_20261031_100614/clipprobe.patch` | 102 行 | 对 02d0cbe `scripts/train_dnn_ppo.py` 的探针补丁（`--clip_probe_out/--clip_probe_iters`） |
| `experiments/exp122_ric_clipgeom_probe_20261031_100614/run.log` | 1 KB | 正式运行日志 |
| `experiments/exp122_ric_clipgeom_probe_20261031_100614/run_attempt1_crash.log` | — | 第一次运行（变量遮蔽崩溃）日志 |
