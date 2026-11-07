# exp124 192M 打法风格 vs bc49 / Mortal（2026-11-07）

- **Date**: 2026-11-07 10:25 PST  **Status**: running
- **Code**: exp129 `probe_concepts.py` 原样（843a2a3），`PROBE_CODE_ROOT=/home/hongdp/Workspace/LLM_Mahjong_exp124`
- **Env**: 本机 CPU（8 进程，与 akochan / R2 / R3 并跑）；$0

## Purpose
用户问"124 终局 192M 和 bc49 或 Mortal 在打法风格上一致吗"。exp129 只测到 144M；补测 192M，与 bc49（同协议）并列；Mortal 走 mjai 桥、不在概念探针里，用四方桌（L2 C 桌）的和牌 / 自摸 / 荣和 / 放铳率作对照。描述性。

## Method
exp129 协议：牌山 50,000,000–50,003,999，4000 局，四家同模型，T=0；T10 风格（和了 / 放铳 / 立直 / 副露 / 听牌率、和了巡目与打点）、T6 立直条件、T7/T8 防守、T1 牌效、T5 鸣牌。

## Progress
- [2026-11-07 10:25 PST] 发射。

## Results

## Conclusion
