# exp71 从零 Q learning 判决实验（纯血线估计器对照，用户 2026-09-06 批准"同步进行"）

- **Date**: 2026-09-06  **Status**: running（实现 + 冒烟）
- **Git**: 见 Progress；RunPod Secure L40S；纯血：随机初始化、镜像自对弈、零人类数据、零外部模型
- **Env**: `train_dnn_dqn.py`（exp59 训练器）新增：`--init` 可省略（随机初始化）、`--league` 可省略（四席学习者镜像）、`--temp_schedule`

## Purpose & Hypothesis
用户问题：Q learning 是否让从零训练更可行？分解后（2026-09-06 讨论）：Q 方法命中"归因信噪比"（一步/n 步自举替代 σ≈5,000 的蒙特卡洛回报）
和"样本效率"（回放池复用），不命中"单局点差均衡不防守"。exp59 的反例：从 bc49 热启动的**去锚**纯 TD 掉到对 bc49 0.21（≈纯血水平），
说明 5 万–20 万局尺度的 TD 不含排序信息——但那是热启动+冻结联赛环境，从零+镜像自对弈从未测过。
**假设 H1（估计器有用）**：同预算（100 万局）从零 Double DQN 产物 T=0 对 exp68r3-P3 ≥ 0.50。
**反假设 H0**：≤ 0.45（ε/Boltzmann 探索拆手牌 + Q 尺度被运气方差淹没 → 弱于策略梯度）。中间带 0.45–0.50 = 不分胜负，记录后不再追加预算。

## Method
- 训练器 exp59 `train_dnn_dqn.py` 在线模式（`collect_parallel`，无 store）；无 `--init`（随机）；无 `--league`（`league_plan` 空池 → 四席全学习者，
  `_package_game` 保留四席 episode，数据 4×）；`--self_frac 1.0`（完全在线，Q 的乐观错误自我证伪）。
- 目标：前 5 万局纯 MC（尺度校准），之后 n=10 步 Double DQN，γ=0.995，硬同步每 500 更新，reward_scale 0.05，Huber。
- 探索：Boltzmann(Q) 温度按局数分段线性 `0:1.0,50000:0.1,300000:0.03,1000000:0.01`（随机初始化 logit≈0 → T=1 即均匀随机；
  MC 校准后 Q 差 ~0.01–0.1 尺度单位，需 T≈0.02 才"按 Q 行动"）+ 单点偏离 `single_dev_p 0.04`（保持手牌连贯）。
- 回放：cap 2M 步（fp16 planes ≈ 8 GB RAM），replay_ratio 2、batch 512、lr 1e-4、bf16 更新。
- 预算 = P3 同预算 **100 万局**；架构 cnn_m_r（同 P3）；`games_per_iter 2048`，6 worker，gpu_infer。
- 无 DQfD 锚、无行为策略（纯度）；无 dueling（判决实验只测"换估计器"，不叠加架构变量）。
- 终评（工作站，T=0）：单局配对牌山 Q vs P3（67M n=4000 + 68M n=16000 双段合并）；Q vs exp27A；防守探针 + 暴露席放铳率；风格向量。

## Success Criteria（预注册）
1. **主判据**：Q vs P3 合并 share ≥ 0.50（n=20,000，SE≈0.0035）→ 估计器换法值得继续（下一步 dueling/n 步 critic 混合）；≤0.45 → 关闭"纯 DQN 从零"。
2. 中途健康：MC→TD 切换后 q_mean/target_mean 不发散（|q|<3 尺度单位）、和牌率单调上升到 ≥15%（PPO 从零同期 ≈20%）。
3. 防守探针只记录不判决。
- 预算：≈4.5–5h Secure L40S ≈ **$5，上限 $6**（与 exp70 并行，独立 pod，不影响 exp70 吞吐）。

## Progress
- [09-06 21:55 PDT] 用户批准同步开跑。训练器改动 + `tests/test_dqn_from_scratch.py`；本机冒烟后发射。
- [09-06 21:20 PDT] 本机冒烟 192 局：随机初始化 199 万参数、镜像四席、replay 21×34 fp16（≈89 步/局，2M cap ≈ 3.5 GB）、MC→TD 切换、温度表生效。
  提交 e6f35f2。发射：Secure L40S US-TX-4 pod `tvrd13n1982g0b`（$1.09/h，ssh 195.26.232.177:46107），GPU 数值校验通过，Q 臂 04:20 UTC 起跑。
  心跳/拉取（含 TB 镜像）挂好；TB 加 exp71_Q_LIVE。预计 ≈4.5–5h。

## Results

## Conclusion

## Next Steps

## Artifacts
