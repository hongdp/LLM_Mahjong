# R7：exp127 交叉格（A1 cnn_l+46、A2 ConvFormer+374）32M 的概念探针（2026-11-07）

- **Date**: 2026-11-07 04:45 PST  **Status**: queued（akochan 主行与 L6b 之后）
- **Code**: exp129 `probe_concepts.py` 原样（843a2a3，md5 4b85fcc0…）；`PROBE_CODE_ROOT=/home/hongdp/Workspace/LLM_Mahjong_perf`（e5fbbfc = exp124 训练分支 f11278a + perf 提交 + `cnn_l_v3r_m46` zoo 条目；引擎与编码器同 exp124）——A1 的架构只在这份代码里有
- **Env**: 本机 CPU；$0
- **Approval**: 论文会话转达其用户的批准（同 R 系列）

## Purpose & Hypothesis
论文称 CNN 从不按条件立直，猜测是因为 374 槽动作空间里立直是逐牌动作。用 2×2 的两个交叉格在**行为**上拆开主干与动作头：若 A1（CNN+46）出现条件立直而 A2（ConvFormer+374）没有 ⇒ 动作头决定；反之 ⇒ 主干决定。描述性。

## Method
- exp129 协议：牌山 50,000,000–50,003,999，4000 局，四家同模型，T=0，含安慰剂，按局聚类 SE。
- 模型：`exp127_A1/games_32000000.pt`、`exp127_A2/games_32000000.pt`，外加 **exp124 A 32M 作对照**（同一次运行、同代码根）——其读数必须与 exp129 的 `A_32M` 逐项一致，否则说明换代码根改变了结果、R7 作废。
- 报：T6 good−bad、T6 early−late、dama 率、T7 iq − 安慰剂（均带 SE），与 exp124 A 32M（exp129）和 exp101 32M（exp129 `C_32M`）并列。

## Success Criteria
对照格逐项复现 exp129 A_32M；其余描述性。

## Progress
- [2026-11-07 04:45 PST] 预注册；排队。

## Results

## Conclusion

## Artifacts
