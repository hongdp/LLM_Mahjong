# exp3_nothink（预注册总纲；各阶段 run dir 由 VM 生成后挂靠）

- **Date**: 2026-08-09 预注册  **Status**: pre-registered, pending launch
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
