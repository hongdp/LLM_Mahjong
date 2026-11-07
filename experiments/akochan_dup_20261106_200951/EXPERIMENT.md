# akochan 单局复式参照行 vs bc49（论文 L5 主行）（2026-11-06）

- **Date**: 2026-11-06 20:35 PST  **Status**: queued（L6 完成后自动开始）
- **akochan**: github.com/critter-mj/akochan @ 53188a0，**未改动 ai_src**；本地构建见 `../akochan_feasibility_20261106_174541/`；**致谢 critter-mj（akochan 作者）**，按其 LICENSE 仅作非营利研究使用
- **Git**: 本仓库 b8d3021 之后；桥 `akochan_seat.py`、驱动 `ako_dup_driver.py`（由 L1 驱动派生），本目录
- **Env**: 本机 24 核 CPU（akochan）+ 4080（bc49 batch-1）；$0
- **Approval**: 本会话用户批准克隆 / 编译（11-06）；论文会话转达其用户对 L5 主行的批准（"Approve both"）

## Purpose & Hypothesis
论文需要一个"外部强规则 / 搜索型 AI"参照行：akochan 在与 exp124 判决相同牌山上的单局复式份额（对 bc49）。描述性。

## Method
- 牌山：标准 10,500 判决牌山的**前 2000 个**（101112000–101113999），T=0 复式 2v2，A = akochan（两个座位各一个 `system.exe pipe` 子进程，每局新开），B = bc49 T=0。驱动与 L1 相同的牌山 / 场况生成（已对存档 300/300 校验）。
- **tactics 改动（只改配置）**：`tactics_points.json` = 官方 `setup_mjai.json` 只把 `jun_pt` 从 `[90, 30, -30, -90]` 改成 `[0, 0, 0, 0]`（单局点差目标，不计顺位点）；其余字段逐字相同（`tactics_default.json` 为原样副本）。
- `start_game` 附加 `kyoku_first: 4, aka_flag: true`（akochan 必需字段）。
- 线程：每进程 `OMP_NUM_THREADS=4`，6 进程并行（避免 24 核超订）。
- **桥校验**：前 300 牌山（600 局）单独统计回退率（akochan 动作不在引擎合法集）与非法率；预注册门槛：回退率 < 0.5%。
- 扩展规则（论文会话指定）：2000 牌山份额在 0.50 ± 3 pp 内才扩到全部 10,500。
- 可选：500 场复式半庄（L3 协议、种子 201000000、akochan 默认顺位 tactics）——仅当主行完成后算力允许。

## Success Criteria
桥校验回退率 < 0.5%；描述性报告份额 ± SE、每牌山点差。

## Progress
- [2026-11-06 20:35 PST] 冒烟：2 牌山（4 局）135 个 akochan 决策 0 回退；4 线程、与 L6 争 CPU 时 75 s/局（≈2.4 核·秒/应对）⇒ 2000 牌山独占 24 核 ≈ 12 h。排队等 L6 完成。

## Results

## Conclusion

## Artifacts
