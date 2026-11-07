# R2：概念探针 T=1 复测（2026-11-07）

- **Date**: 2026-11-07 01:35 PST  **Status**: queued（R4 之后）
- **Code**: `probe_concepts.py` 原样（843a2a3，md5 4b85fcc0…），`--temperature 1.0`，`PROBE_CODE_ROOT=/home/hongdp/Workspace/LLM_Mahjong_exp124`
- **Env**: 本机 CPU（8 进程，与 akochan 并跑）+ 4080；$0
- **Approval**: 论文会话转达其用户 11-07 的批准

## Purpose & Hypothesis
exp129 只在 T=0（贪心）上测；T=1 是训练时的采样策略。检验涌现结论对温度是否稳健。描述性。

## Method
- 与 exp129 完全同协议（牌山 50,000,000–50,003,999、4000 局、同指标、含安慰剂、按局聚类 SE），只把温度改为 T=1。
- 模型：exp124 A 全部 20 点（0、4M–144M，同 exp129 的路径）+ bc49；再加 exp129 的 11 个 CNN 点（exp101 cnn_l 8M–392M）。
- 注意：T=1 下每局轨迹带采样随机性（种子固定，但依赖批次顺序），因此不与 T=0 逐局配对，只比较各指标水平与涌现点。

## Success Criteria
描述性。

## Progress
- [2026-11-07 01:35 PST] 预注册；排在 R4 之后。

## Results

## Conclusion

## Artifacts
