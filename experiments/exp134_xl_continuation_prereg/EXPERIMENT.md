# exp134：XL（11.4M）从 64M 续训到 192M，检验大模型能否在更多对局后反超 M（2026-11-08）

- **Date**: 2026-11-08 21:40 PST  **Status**: prereg（发射见 Progress）
- **Git**: perf 分支 `claude/perf-rollout-overlap` bf43537（流式 rollout；本臂不用 v3ro / loo）；运维 `experiments/exp134_ops/`（派生自 `exp124s2_ops`）
- **Env**: RunPod Secure RTX 3090 / A40（≤ $0.59/h），仅 22/tcp，pod 上无凭据；完成后自毁
- **Approval**: 本会话用户 11-08"流式续训 $120"
- **Budget**: ≈ 128M 局 ÷ ≈150 局/s ≈ 240 h ≈ **$120**，上限 **$150**；超过 1.3× 预估时长报用户

## Purpose & Hypothesis
scaling law 的"交叉"问题：同类模型放大后，小预算时不占优，但上限更高、在更多对局后反超。exp125 只把 XL 训到 64M（XL − M +0.41 ± 0.60 pp，H0），而 M 在 64M 之后还涨到 192M（0.489 → 0.511），所以交叉没被检验过。本实验把 XL 续训到 192M，在 M 已有判决点的 112 / 160 / 192M 处同牌山配对。
- **H1（交叉）**：192M 时 XL − M ≥ +1.0 pp。
- **H0（同类放大不是杠杆，延伸到 192M）**：|XL − M| < 1.0 pp。
- **反向**：XL − M ≤ −1.0 pp。

## Method
- 起点：`exp125_XL/games_64000000.pt`（md5 456ead57…，含优化器），`convformer_xl_v3r_m46` 11.37M。
- 配方对齐 M 的第二段（`exp124_A_p2`）：lr 3e-5、warmup 0、seed 1351（M 第二段 1341；XL 第一段 1251 → +100，与 M 同规则）、batch 4096、1 epoch、dup_k 8、γ 0.995、λ 0.95、熵 0.03、target_kl 0.03、adv_clamp 5、`--rollout_values --fast_trunk --compile --bf16_rollout`，64M → 192M。
- **唯一与 M 第二段不同之处：流式 rollout**（`--stream_rollout --m46_one_forward`，K=512、块 128、队列 1，同 exp124s2）。依据：exp127 B1 续训阶段流式 − 同步 = +0.17 ± 0.43 pp（等价）。判读时注明。
- 显存：XL 在 24 GB 卡上同步 batch 4096 已通过（exp125 pod 闸门）；流式多一份演员网络。若 pod 臂门 OOM，改 batch 2048（梯度累计不变的条件下需同时注明），预注册的判据不变。
- 里程碑：80 / 96 / 112 / 128 / 144 / 160 / 176 / 192M；在轨每 2M 一次 2000 牌山读数。
- **判决点**：112M / 160M / 192M，标准 10,500 去重判决牌山（101112000 段 1 万 + 101112500 段末 500），与 `exp124_A` 同点逐牌山配对（`probes/pairdiff_exp124_A_n10000_games_{112,160,192}000000.pt_{101112000,101112500}.json`）。

## Success Criteria（发射前写定）
主判据 = **192M 的 XL − M（10,500 牌山配对）**：≥ +1.0 pp ⇒ H1（交叉成立）；≤ −1.0 pp ⇒ 反向；其余 ⇒ H0。辅判：112 → 160 → 192M 的 XL − M 是否单调上升（交叉的形态）。单种子、不含种子方差。
**早停**：若 160M 时 XL − M ≤ −2.0 pp（配对），报用户提议停止（省余下 ≈ $30）。

## Progress
- [2026-11-08 21:40 PST] 预注册。

## Results

## Conclusion

## Artifacts
