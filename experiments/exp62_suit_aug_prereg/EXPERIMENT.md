# exp62 花色置换增广 BC A/B：训练时对称增广能否兑现样本效率

- **Date**: 2026-09-03  **Status**: running
- **Git**: 见 Progress 首条（worktree `runpod-training-cost-optimization-626ab2`）
- **Env**: RunPod **Secure** L40S（$1.09/h，13 核；产物为冠军谱系权重 → 按安全分层走 Secure）；
  数据 = 凤凰卓 20.9k 局 `data/tenhou/raw`（scp 上 pod，pod 上零云凭证）；pod 上物化 v3r/mortal46 缓存后训练

## Purpose & Hypothesis
exp61 已证：bc49 对花色置换的破缺（5.7%）全在近平决策，测试时对称平均零效应。剩下的问题是**训练时增广**：
exp49 表明 BC 处于数据受限区（9× 数据 = +3.7–4.0pp 精度 = +27–57 Elo，18.4k 局未饱和），花色增广给每个样本
6 个等价视角（相关但不重复），相当于 2–3× 有效数据。假设：增广臂 holdout 精度 **+0.5pp 以上**，T=0 头对头对控制臂显著为正。
反假设：ConvFormer 的注意力桶偏置已经足够利用花色结构，增广只带来 <0.2pp 的噪声级差异。

## Method
同 [docs/champion_model.md §5](../../docs/champion_model.md) 的 bc49 配方（`convformer_m_v3r_m46`，全量非留出局，holdout 1000 局，
max_epochs 30 / patience 3 / min_delta 5e-4，batch 1024，lr 3e-4 const，seed 0），两臂在同一 pod、同一物化缓存上并行：
- **C（控制）**：原配方原样重跑（同时是 bc49 = 0.8059 的跨硬件复现检查，给出"硬件/非确定性噪声地板"）。
- **A（增广）**：`--suit_aug`：每个训练 batch 切 6 块，第 k 块施加第 k 个花色置换（planes 牌轴 / mask 槽位 / 标签一致置换，
  `src/agents/dnn/symmetry.py::make_batch_augmenter`）；手牌+副露绿牌 >7 张的样本保持恒等（绿一色守卫）。holdout 评测恒等。
增广是 exp61 基建的直接复用（6 项单元测试 + 一项增广一致性测试 + 本机真数据冒烟）。

## Config
- 缓存：pod 上 `materialize_bc.py --variant v3r --action_space mortal46 --workers 12`（本机 v3r2 缓存 31GB 无法上传）。
- 两臂并行各 `--workers 5`，输出 `/workspace/exp62_{ctrl,aug}`；每 epoch 存 best ckpt + metrics.json + TB。
- 终评（工作站，0 元）：A vs C 配对牌山单局 T=0 n=4000 对（+ A/A）；A vs bc49、C vs bc49 各 n=4000；破缺率诊断（预期 A → <1%）。
- 预算：约 2.5 小时 ≈ **$2.7，上限 $5**（含 pod 卡死重开）。

## Success Criteria（预注册）
1. **主判据（强度）**：A vs C，T=0 配对牌山 n=4000 对，share **≥ 0.52（+2.5σ）** → 增广兑现样本效率，进入下一步
   （数字翻转增广 + 用增广重训冠军配方作为加冕候选）。0.50–0.52 → 精度端看判据 2 再定；<0.49 → 增广有害（记录）。
2. **精度判据**：A holdout acc − C holdout acc ≥ **+0.5pp**（exp49 汇率 ≈ +5 Elo/… 0.5pp 是 seed 噪声地板的 ~3 倍）。
   |C − bc49 0.8059| > 0.3pp 时复现检查亮黄灯，A/B 仍以 A−C 为准。
3. **对称性判据**：A 的破缺率 < 1%（不满足说明增广未生效，A/B 结论作废）。
4. 诚实条款：早停规则两臂同；比较用各臂 best ckpt；不做事后选 epoch。

## Progress

## Results

## Conclusion

## Next Steps

## Artifacts
