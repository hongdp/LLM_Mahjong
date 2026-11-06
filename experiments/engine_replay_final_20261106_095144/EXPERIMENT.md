# 最终引擎重放 exp45 的 1520 局凤凰卓牌谱（论文 L4）（2026-11-06）

- **Date**: 2026-11-06 09:52 PST  **Status**: running
- **Git**: 9ce250f（Python 引擎 table.py / claims.py 最后改动 13095b9 2026-08-27 = exp45 动作缺口修复；Rust 引擎与之逐位对拍）
- **Env**: 本机 CPU；$0

## Purpose & Hypothesis
论文会话 L4：在当前（最终）引擎上重放 exp45 中期保真门用的同一 1520 局（973,970 个人类决策，08-26 原读数非法 6 例 = 0.0006%，全部为当时的引擎动作空间缺口，08-27 修复），报告非法动作数 / 率；另报当前规则测试规模（pytest 用例数）。
预期：0 非法（修复后原 6 例已复放归零，见 docs/engine_known_issues.md）。

## Method
- 文件集：从 `data/tenhou/decisions_p1500.jsonl` 取 1520 个唯一 mjlog（全部在 raw 中找到），软链成临时 raw 目录，运行 `scripts/harvest_human_bc.py --raw <tmp> --out <scratchpad>`（逐座 MjaiDnnBot 回放、人类动作 ∈ 引擎合法集断言、预注册歧义弃样）。
- 规则测试：`pytest --collect-only -q` 计数，另列引擎规则相关测试文件的用例数。

## Success Criteria
描述性；决策数应与 08-26 一致（973,970 保留 + 6 非法）或说明差异。

## Progress
- [2026-11-06 09:52 PST] 文件集重建（1520/1520），发射重放。

## Results

## Conclusion

## Artifacts
