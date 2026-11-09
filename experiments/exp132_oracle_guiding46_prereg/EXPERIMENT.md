# exp132：ConvFormer46 上的 oracle 策略引导（臂 O），从零到 32M（2026-11-08）

- **Date**: 2026-11-08 14:40 PST  **Status**: running（2026-11-08 14:23 PST 起训）
- **Git**: perf 分支 `claude/perf-rollout-overlap` bf43537（v3ro 编码器 / `convformer_m_v3ro_m46` / VecEnv `oracle_hide_p`）；运维 `experiments/exp132_ops/`（`repo.tgz` = perf bf43537 的 git archive，`riichi_rs` cp312 轮子由同一提交本机构建）
- **Env**: RunPod Secure RTX 3090（或 A40，≤ $0.59/h），仅 22/tcp，pod 上无凭据；完成后自毁
- **Approval**: 本会话用户 11-08"都开"（exp132 + exp133，合计约 $30）；论文会话转达其用户"跑"
- **Budget**: ≈ 32M ÷ 300 局/s ≈ 30 h × $0.50 ≈ **$15**，上限 **$20**；超过 1.3× 预估时长报用户

## Purpose & Hypothesis
论文 RQ3：纯自对弈能否学得更快？Suphx 式 oracle 引导：训练前期策略看到隐藏信息（对手暗手、待张、牌山、里宝），用 dropout 逐步遮掉，最终只用公开信息。exp68（CNN、1M 局、Python 引擎）显示"加速器"：退火刚结束时 G vs P 0.541，但退火后 40 万局全盲训练把优势追平到 +0.7%（z 2.0），防守探针没有迁移。本臂在 ConvFormer46（exp124 主干）上从零复测。
- **纯度**：无人类数据；特权信息是引擎状态，只在训练 rollout 里可见，评测 / 部署永远看不到（见 Method 校验）。

## Method
- 唯一改动：`--arch convformer_m_v3ro_m46 --oracle_hide_schedule 0:0.0,8000000:1.0`。v3ro = v3r 的 56 平面 + 8 个特权平面（3 家暗手计数 / 4、3 家待张、牌山剩余 / 4、里宝指示牌），与 exp68 的 v1ro 同一组内容；每局开局按隐藏概率 p 抽一次是否可见（p 随已训局数线性 0 → 1，**8M 起全盲**，此后 24M 局全盲训练）。
- **退火长度的选择**（发射前写定）：exp68 是 1M 预算的前 60%（60 万局）退火；本臂取"最晚 8M 结束"，留 24M 全盲尾段——exp68 的教训是短的全盲尾段就能把引导优势抹平，长尾段才是"知识是否迁移到公开线索"的检验；32M 判决时完全是盲的。
- 其余 = exp124 A 第一段配方（perf 代码，同步调度，与 exp133 / exp127 A 臂相同）：lr 1e-4、seed 1241、warmup 3000、batch 4096、1 epoch、dup_k 8、γ 0.995、λ 0.95、熵 0.03、target_kl 0.03、adv_clamp 5、shaping 0→400k、`--rollout_values --fast_trunk --compile --bf16_rollout`，里程碑 8 / 16 / 24 / 32M。
- **评测永远是盲的**：判决用 `pairdiff_eval`（Python 引擎 `collect_parallel`），只有训练 rollout 传入 `oracle_hide_p`；评测路径不传 ⇒ 默认 1.0 ⇒ 特权平面全零，与部署时一致。
- **对照**：exp124 A 32M（0.4810）。主判据 = 32M 对 bc49 份额，标准 10,500 去重判决牌山，与 exp124 A 32M 逐牌山配对；次要 = 8 / 16 / 24 / 32M 前 2000 牌山曲线（8M 正是全盲开始点）、到达 0.48 所需局数。

## Success Criteria（发射前写定）
Δ = 臂 − 对照（同牌山配对）：**Δ ≥ +1.5 pp ⇒ 加速器**；**Δ ≤ −1.5 pp ⇒ 有害**；其余 ⇒ **H0**。1.5 pp ≈ 配对牌山差 3 SE；**不含种子方差**（单种子）。

## Progress
- [2026-11-08 14:40 PST] 预注册。移植验证（本机）：
  - Rust / Python v3ro 编码逐位一致（120 局随机对局全部状态，隐藏与可见两种情况；`tests/test_v3ro_parity.py`），原 v3r / oracle 平价测试仍通过；
  - 训练冒烟（1,536 局，hide 计划 0 → 1 @ 1,024）：迭代记录的 hide_p 0.0 → 0.5 → 1.0；hide_p = 0 时 rollout 特权平面非零占比 17.5%，hide_p = 1 时 0；
  - 冒烟检查点经 `pairdiff_eval`（判决路径）20 牌山正常运行；pod 启动检查（导入 / 架构 / VecEnv 接口）本机通过。

- [2026-11-08 14:35 PST] **发射**：pod `1h3k7x0pz8e7r7`（exp132-O），Secure RTX 3090 ×1、32 vCPU（cgroup 31 核）、EU-CZ-1、**$0.50/h**，仅 22/tcp，SSH 213.192.2.110:40070，pod 上无凭据。编译门 / 臂门（`convformer_m_v3ro_m46`）通过；22:23 UTC 从 0 局起训，标志 `--rollout_values --ckpt_every 400 --oracle_hide_schedule 0:0.0,8000000:1.0`。8.8 min 时 **291 局/s** ⇒ 32M ≈ 30.5 h ≈ **$15.3**（上限 $20）。train_log 的 hide_p 随局数线性上升（14.95 万局时 0.0184，计划值 0.0187）。本地 pull / watchdog / eval 循环已起；TB `_tb_qlearning/exp132_O_oracle46_0-32M_LIVE`；配置快照 `experiments/exp132_O/config.json`。
- [2026-11-08 22:05 PST] 8M 里程碑（md5 与 pod 一致）：oracle_hide_p 按计划线性升至 1.0（1.07M 处 0.134、7.50M 处 0.937、8.57M 处 1.0），无偏离。2,000 牌山（101112000 起）份额 0.4310 ± 0.0105，同牌山对 exp124 A 8M（0.4413）配对 **−1.03 ± 1.42 pp**，在轨读数、不作判决（判决在 32M）。吞吐 ~310 局/s。

## Results

## Conclusion

## Artifacts
