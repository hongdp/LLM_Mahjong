# exp124 A 192M 对 Mortal 298k 的直接复式单局

- **Date**: 2026-11-15 01:39 PST（预注册写定；192M 对 Mortal 的对局在本文件写定之后才发射）  **Status**: done 2026-11-15 02:58 PST
- **Git**: d62dbc0（分支 `claude/rl-model-defense-learning-8fb5c0`；未提交改动：无，驱动 `dup_driver.py` 是本目录新文件，gitignored）
- **Env**: 本机 RTX 4080，`rlhf_mahjong` 环境；Mortal 上游代码 `/home/hongdp/Workspace/mortal_upstream/mortal`（libriichi mjai Bot）；$0，不用云。
- **检查点**：
  - A 方（被测）= exp124 A 192M `experiments/exp124_A_p2/games_192000000.pt`，md5 `b71ba61ae23d3f6880c4d52dd80d83ca`。
    这就是论文 §strength 用的 192M 文件（对 bc49 单局判决 0.5113 的 `pairdiff_exp124_A_n10000_games_192000000.pt_*` 所用文件）。
    注：exp124 预注册 Stage 2 段把 md5 记成 19ee60c5…，那是 `games_final.pt` 的 md5（同局数、同迭代、文件差约 1 KB，见 `experiments/majsoul_exp124A_192Mfinal_20261107_100245/`）；本 run 用里程碑文件 b71ba61a…。
  - B 方（对手）= Mortal 298k `data/mortal_ext/mortal_298k.pth`，md5 `3be35703708096a97fcb6c1f272d2792`，原样走 mjai 桥（上游原生贪心）。权重不得再分发，本 run 不复制权重。

## Purpose & Hypothesis
论文里 192M 与 Mortal 的比较是间接的：两者各自对 bc49（192M 0.5113 ± 0.0033；Mortal 0.5054，20,500 副），差 +0.7 ± 0.4 pp，不显著。
盲审把"没有直接对战"列为主要弱点。间接差不保证能传递（SKILLS 2026-11-13：288M / 320M 对 bc49 的半庄间接差没能传递成对 256M 的直接差），
所以直接把 192M 和 Mortal 放上同一桌。
用户批准（11-15，经论文写作会话转达，原话"跑直接对战"）：仅这一项，仅本机，不花云上的钱。

不预设方向。间接读数暗示 192M 略占优或者两者等价。

## Method
- 2v2 复式：Mortal 坐 A 座位（0,2 / 1,3 两种安排），192M 坐 B 座位；每副牌山两种座位安排各打一局，点差相加后按符号记胜 / 平 / 负。
  bc49 不上桌。座位协议、`random.seed(wall)` 发牌、mjai 掩码视图、立直两步合并与 `scripts/arena_mortal_mjai.py` 及既有 `mortal_shard*.json` 完全一致。
- 192M 用 T=0 贪心（`net.act(temperature=0)`），与论文单局协议相同。
- 牌山复用 Mortal-vs-bc49 的两组，分片也照原样：
  - 组 1：seed0 101112000，walls 0–10,499，12 片 × 875（同 `probes/paper_ref/mortal_shard*.json`）；
  - 组 2：seed0 101152000，walls 0–9,999，8 片 × 1,250（同 `probes/r6_mortal/mortal_shard*.json`）。
  合计 20,500 副。
- 驱动 `dup_driver.py`（本目录）= 生成既有 Mortal shard 的原驱动 `experiments/paper_ref_rows_20261106_062737/dup_driver.py` 的副本，
  唯一改动：可选第 6 个参数替换 B 方的 bc49（仍 T=0），输出多记 `opp` / `opp_temp`（diff 见 Progress）。
  从 worktree 根目录运行（与原 run 相同）；原 run 在 4ae29e1，到 d62dbc0 之间 `src/tasks`、mjai 桥 / 导出、编码器、动作空间、
  `arena_mortal_mjai.py`、`run_arena_dnn.py` 均无改动。
  发射前对拍：B 方不传（= bc49 T=0），重放组 1 前 40 副、组 2 前 20 副，逐副比对 `paper_ref/mortal_shard0.json` / `r6_mortal/mortal_shard0.json` 的 diff，
  必须全部一致才发射。
- 输出 `experiments/probes/mortal_vs_a192/s{1,2}_shard{k}.json`。格式与 `mortal_shard*.json` 相同（rows 的 share / diff 是 **Mortal 一方**的），
  另加 `opp`、`opp_temp` 字段；192M 份额 = 1 − share。日志 `logs/`（本目录）。
- 192M 非法动作：策略只从引擎给出的合法动作表里选（`net_policy` 解析失败时退回 `actions[0]`，仍是合法动作），所以非法动作在结构上为 0；
  该回退次数本驱动不计数（原驱动也不计）。

## Config
`python dup_driver.py mortal:data/mortal_ext/mortal_298k.pth W0 W1 SEED0 OUT.json experiments/exp124_A_p2/games_192000000.pt`；
组 1 先跑（12 进程并行），跑完接组 2（8 进程并行）。驱动 `run.sh`，心跳 `heartbeat.sh`（本目录）。

## Success Criteria（发射前写定）
- **主指标**：192M 的单局份额（每副牌山胜 1 / 平 0.5 / 负 0，与论文同定义）；SE 按牌山算（每副牌山一个观测）。组 1、组 2 分别报，再合并报（20,500 副）。
- **判读**（合并 20,500 副）：
  - |z| ≥ 1.96（z = (份额 − 0.5) / SE）⇒ 有显著差异，写明方向；
  - 90% CI 完全落在 0.5 ± 1 pp 内 ⇒ 等价（论文的等价规则）；
  - 两者都不满足 ⇒ "差异不显著"，报区间。
  （两条可以同时满足，例如显著但幅度 < 1 pp：两条都报。）
- **不设延长规则**：20,500 副就是终点，无论结果如何都不加样本。
- 附报：点差 / 牌山（192M 一方）、Mortal ok / fallback 计数、192M 非法动作计数（预期 0）、与间接比较对照：
  Mortal 对 bc49 在同牌山上的份额（`paper_ref` / `r6_mortal` 已有）和 192M 对 bc49（组 1 前 10,000 副有 `pairdiff_exp124_A_n10000_games_192000000.pt_101112000.json`）。
- **范围戳**：ConvFormer-M46（`convformer_m_v3r_m46`）/ 纯自对弈 PPO / 192M 局 / T=0 单局复式 2v2，单检查点单训练种子，本项目引擎；
  Mortal 为公开 298k 权重、mjai 桥、上游原生贪心。失效条件：换检查点 / 换 Mortal 版本 / 换桌型（混桌）/ 半庄刻度不复现。

## Progress
- [2026-11-15 01:39 PST] 预注册写定。
- [2026-11-15 01:44 PST] 驱动修正：先写的 `mortal_vs_dnn.py` 对拍 2/16 不一致（漏了 `PyMahjongTable(randomize_round=True)`，场况与原 run 不同），已删除；
  改用原驱动副本 `dup_driver.py`（见 Method）。**对拍通过**：Mortal 对 bc49（T=0）组 1 前 40 副 40/40、组 2 前 20 副 20/20 逐副 diff 与存档完全一致，0 回退。
  192M 冒烟 8 副（组 1 walls 0–7）正常、434 决策 0 回退；这 8 副正式 run 里会重打，冒烟结果不计入。01:44:09 发射（组 1 12 片 → 组 2 8 片）：`run.sh` PID 980584，心跳 `heartbeat.sh` PID 989297（每 10 分钟，`logs/heartbeat.log`；
  第一版心跳的进程计数会把自己的启动 shell 算进去，01:45 按 PID 结束后改为只数 run.sh 的子进程重启）。按原 run 墙钟估计组 1 ≈ 40 分钟、组 2 ≈ 60 分钟。
- [2026-11-15 01:50 PST] 论文写作会话独立对拍 192M 一侧（转述）：用本目录 `dup_driver.py` 以 192M 为 A 方、bc49 为默认 B 方重放组 1 walls 0–29，
  逐副 diff 与 `probes/pairdiff_exp124_A_n10000_games_192000000.pt_101112000.json` 前 30 行 30/30 一致（输出在该会话 scratchpad）；
  它在 01:46 记录 `probes/mortal_vs_a192/` 为空，可作"判据早于任何结果"的旁证。

- [2026-11-15 03:02 PST] 全部完成：组 1 02:20:49（12 片各 36 分钟），组 2 02:58:34（8 片各 37 分钟），ALL_DONE 02:58:35；`analyze.py` 输出 `result.txt` 与 `probes/mortal_vs_a192/summary.json`。
  Mortal 1,229,762 个决策 0 回退。

## Results
192M（B 方，T=0）对 Mortal 298k（A 方，mjai 桥原生贪心），2v2 复式单局，SE 按牌山：

| 牌山组 | n | 192M 份额 ± SE | z | 192M 点差 / 牌山 | Mortal ok / 回退 | 同牌山间接对照：Mortal 对 bc49 |
|---|---|---|---|---|---|---|
| 组 1（101112000，0–10,499） | 10,500 | 0.5050 ± 0.0045 | +1.11 | +207 ± 131 | 629,609 / 0 | 0.5039 |
| 组 2（101152000，0–9,999） | 10,000 | 0.5016 ± 0.0047 | +0.33 | −91 ± 135 | 600,153 / 0 | 0.5071 |
| **合并** | **20,500** | **0.5033 ± 0.0033** | **+1.03** | +62 ± 94 | 1,229,762 / 0 | — |

- 合并 90% CI = [0.4980, 0.5087]。
- 按预注册判读：|z| = 1.03 < 1.96 ⇒ 无显著差异；90% CI 完全落在 [0.49, 0.51] 内 ⇒ **等价（±1 pp）**。
- 对照间接比较：各自对 bc49 的间接差是 +0.7 ± 0.4 pp（192M 0.5113，Mortal 0.5054）。直接读数 +0.33 ± 0.33 pp，方向相同、幅度约一半，两者差在 1 SE 内。
  组 1 上 192M 对 bc49（前 10,000 副）是 0.5136，Mortal 对 bc49（10,500 副）是 0.5039。
- 192M 非法动作：结构上为 0（见 Method）。

## Conclusion
**范围戳**：ConvFormer-M46（`convformer_m_v3r_m46`）/ 纯自对弈 PPO / 192M 局（`exp124_A_p2/games_192000000.pt`，md5 b71ba61a…）/ T=0 单局复式 2v2，
单检查点单训练种子，本项目引擎；Mortal 为公开 298k 权重、mjai 桥、上游原生贪心；20,500 副牌山（两组，复用 Mortal-vs-bc49 牌山）。
失效条件：换检查点 / 换 Mortal 版本 / 换桌型（混桌）/ 半庄刻度不复现。

1. **exp124 A 192M 与 Mortal 298k 在单局复式刻度上等价**：192M 份额 0.5033 ± 0.0033（z 1.0），90% CI [0.498, 0.509] 落在 ±1 pp 等价带内。
   把论文里的"间接打平"换成了直接等价结论。
2. 直接差（+0.33 pp）小于间接差（+0.7 pp），但在误差内一致；这一次间接比较没有误导方向。与 11-13 的教训（对 bc49 的半庄间接差没能传递）相比，单局刻度上的传递性这次成立。
3. 本结果只覆盖 192M。现役冠军 256M 对 192M 同牌山 +1.09 ± 0.43 pp（对 bc49），但 256M 对 Mortal 没有直接对战，不能据此推出 256M 优于 Mortal。

## Next Steps
- 论文写作会话从原始 rows 独立复算后写进论文（该会话自己负责 paper/）。
- 如需"冠军 vs Mortal"的直接读数：同一驱动、同牌山跑 256M（本机约 75 分钟、$0），需用户另行批准；半庄刻度需要 Mortal 走半庄 mjai 桥（`scripts/run_hanchan_arena.py`），另议。

## Artifacts
| Path | Size | Description |
|---|---|---|
| `experiments/probes/mortal_vs_a192/s1_shard{0..11}.json`（主 checkout） | 12 × ≈55 KB | 组 1 逐牌山 rows（Mortal 一方的 share / diff）+ stats |
| `experiments/probes/mortal_vs_a192/s2_shard{0..7}.json` | 8 × ≈78 KB | 组 2 同上 |
| `experiments/probes/mortal_vs_a192/summary.json`、本目录 `result.txt` | <1 KB | 合并数字 |
| 本目录 `logs/parity_s{1,2}_bc49.json`、`logs/smoke_a192.json` | — | 发射前对拍（Mortal 对 bc49）与 192M 冒烟，不计入结果 |
| `dup_driver.py`、`run.sh`、`heartbeat.sh`、`analyze.py`、`logs/`（本目录，gitignored） | — | 驱动、发射、心跳、汇总 |
