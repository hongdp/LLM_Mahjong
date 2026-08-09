# exp3_nothink（预注册总纲；各阶段 run dir 由 VM 生成后挂靠）

- **Date**: 2026-08-09 预注册  **Status**: complete（2026-08-09；首发 OOM 快速止损后 batch-4 重发，四阶段全 exit 0）
- **类型**: 基建消融实验（加速主线 #1）；用户 2026-08-09 批准立项（"同意开始执行你提议的最终方案"）。
- **动机**: think 审计（exp1）：2B 模型 think 内受入声明仅 6-11% 正确（教师语料 100%）、RL 全程无改善；think 占生成 token 70-80%。若 think 对动作质量无因果贡献，砍掉 = rollout 直接 2-3× 提速，且为 v3「真值表进 prompt + 短 think」路线提供基线。
- **设计（构造级单变量）**: 同 seed(42) 同引擎（post-audit RCR）生成配对语料各 11390 样本，教师动作 100% 一致（前 2000 样本验证），唯一差异：
  - `sft_mahjong_think_v2.jsonl`（sha256 97f0ccf5…）：响应 = `<think>忠实CoT</think><action/>`
  - `sft_mahjong_nothink.jsonl`（sha256 e5e8f78d…）：响应 = 裸 `<action/>`，system prompt 禁止思考
  - 注意：think 侧也重新生成（不复用 exp2 的 005918 锚点），排除引擎版本/语料规模混淆。
- **流程**（单台 flex VM，4 阶段，scripts 见 exp3_nothink_vm.sh）:
  1. SFT-T：think_v2 语料 3×2000（lr 1e-4, batch 8）→ 锚点 T
  2. SFT-N：nothink 语料同参 → 锚点 N
  3. probe-T：锚点 T 自对弈 48 局（parallel 24，lr=0 只测不学，settlement reward 仅计分）
  4. probe-N：锚点 N 同 seed 48 局
- **Success Criteria（预注册，启动前）**:
  1. **速度（主判据）**: probe-N 的 rollout 墙钟时间 ≤ probe-T 的 1/2（≥2× 提速）。
  2. **质量不劣化**: probe-N 和牌局占比、放铳率与 probe-T 差 < 10pp；格式合规 ≥99%。
  3. **SFT 收敛**: 两侧 SFT 最终 epoch loss 同量级（nothink 序列短，loss 绝对值不可直接比，比收敛形态）。
  4. **判定**: 1&2 同时满足 → no-think 成为 v3 默认模板，后续 RL（exp4）建立在锚点 N 上；速度达标但质量劣化 → 转「短 think + 真值表」折中方案；速度不达标 → 记录并保留 think。
- **Env**: flex-start a2-highgpu-1g（us-east1-b，与竞技场机不同区），代码 = 加速包里程碑 commit（ref 缓存 + no_think 贯通 + parallel 24）。
- **Results sink**: `gs://llm-mahjong-experiments/exp3_sft_think_*/`, `exp3_sft_nothink_*/`, `exp3_probe_*/` + `exp3_logs/`。
- **Cost plan**: SFT ~30min ×2 + probe ~40min ×2 + bootstrap ≈ 3h × $2.02 ≈ $6。

## Results（判据对照）
| 预注册判据 | 结果 | 判定 |
|---|---|---|
| 1. 速度 ≥2×（主） | probe-think 60.7min vs probe-nothink 63.2min（48 局 ×24 并发） | ❌ 不达标 |
| 2. 质量差 <10pp | 和牌局率 68.8% vs 68.8%，放铳局率 52.1% vs 52.1%（完全一致）；格式 99.9% vs 100% | ✅ 通过 |
| 3. SFT 收敛 | think 0.183→0.080，nothink 0.122→0.070，形态一致 | ✅ 通过 |
| 4. 总判定 | 速度不达标分支 → **保留 think** | 按预注册执行 |

## Conclusion
1. **前提失效而非方案失败**：exp1 的「think 占 70-80% token」量自旧长 think 模型；本轮配对语料的教师 think 是紧凑单行（生成响应中位 145 vs 77 字符，仅 ~4× token 差），紧凑 think 语料已经预支了绝大部分预期加速。probe 墙钟被 prompt prefill + lr=0 更新等与 think 无关的成本主导（估算解码只占 think 侧 ~18%）。
2. **质量零差异**是干净的正结论：think 内容对动作质量无因果贡献（与 think 审计 6-11% 正确率互证）——决策信息全在 prompt 的状态里。
3. **风格重磅副产品**：同一教师策略下，think 锚点副露 5.06/局 vs nothink 1.06/局（质量完全相同）——风格指标对模板/采样高度敏感且与强度解耦。①exp2 H3 的「S 臂风格迁移」解读需加保留（已补 addendum）；②风格探针今后不作为漂移/强度的代理指标。
4. **复活条件**：未来若 rollout 转入解码瓶颈配置（vLLM spike、更高并发、长上下文 v3），4× 解码 token 差重新变现，此消融可直接复用本轮配对锚点。
5. 成本：$8（含首发 OOM 止损 $0.3）。
