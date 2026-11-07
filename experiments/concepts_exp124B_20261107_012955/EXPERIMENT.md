# R4：exp124 B（lr 6e-5 兄弟臂）的概念涌现轨迹（2026-11-07）

- **Date**: 2026-11-07 01:35 PST  **Status**: running
- **Code**: `probe_concepts.py` 原样（论文会话分支 claude/paper-writing-suggestions-3ebde2 @ 843a2a3，文件 md5 4b85fcc0…），`PROBE_CODE_ROOT=/home/hongdp/Workspace/LLM_Mahjong_exp124`（f11278a），与 exp129 完全同协议
- **Env**: 本机 CPU（8 进程，与 akochan 并跑）+ 4080；$0
- **Approval**: 论文会话转达其用户 11-07 的批准

## Purpose & Hypothesis
论文需要第二条独立轨迹来检验 exp129 在 A 臂看到的概念涌现**顺序与时间**是否可复现。B 臂 = exp124 第一段同配方、lr 6e-5、seed 1242，Stage 0 后继续到 33.6M 停止（latest）。描述性。

## Method
- 检查点：`exp124_B/games_{0,4,8,16,24,32}M.pt` + 终点 `probes/snap_exp124_B_games_33587200.pt`（latest.pt 的冻结副本，md5 3f772b66…）。
- exp129 协议：牌山 50,000,000–50,003,999（4000 局），四家同模型，T=0，含安慰剂；全部指标与按局聚类 SE。
- 涌现规则同 exp129 预注册（对均匀期望、|z| ≥ 3 且此后保持；T7/T8b 用"威胁 − 安慰剂"），逐行报 B 臂的涌现点，与 A 臂并列比较。

## Success Criteria
描述性。

## Progress
- [2026-11-07 01:35 PST] 预注册；发射（R4 在 R2 之前跑）。

## Results

## Conclusion

## Artifacts
