# GCP 算力选型：配额、计费模型与成本

> 2026-08-02 调研。目的：三臂实验按需（on-demand）跑一轮约 $286，是否有更划算且可靠的方案。
> 结论先行：**改用 DWS flex-start，成本降到 $158（−45%），且不会被抢占**；Spot 只再省 $7，
> 但要承担 26 小时长跑中途被抢的风险，不值得。

## 1. 结论与推荐

| 方案 | 整机单价 | 一次 run（26h） | 三臂舰队 | 中断风险 | 推荐 |
|---|---|---|---|---|---|
| 按需 STANDARD | $3.674/h | $95.5 | $286 | 无 | 仅调试/临时 |
| **DWS flex-start** | **$2.020/h（−45%）** | **$52.5** | **$158** | **7 天内不被抢占** | ✅ **默认** |
| Spot | $1.928/h（−47%） | $50.1 | $150 | 随时可能被抢 | 仅在 `--resume` 可靠后考虑 |

**为什么是 DWS 而不是 Spot**：两者差价仅 $0.09/小时 —— 跑完一整个 run 只差 **$2.41**，
三臂舰队差 $7.23。这点钱换来的是「一旦拿到资源，7 天内不会被回收」。我们的 run 要连续跑
26 小时，Spot 中途被抢一次损失的 GPU 小时数就远超省下的 $2.4，还要额外赌 `--resume`
断点续训的正确性。

## 2. 关键前提：一次 run 只要 ~26 小时

flex-start 的硬上限是 **7 天（168 小时）**，所以适用性取决于单次 run 的时长。
实测（rev3 三臂 run 的 TensorBoard wall time，parallel_games=12）：

| run | 已完成 epoch | 单 epoch 耗时 | 50 epochs 推算 |
|---|---|---|---|
| v2_engine_pbrs_run_20260802_054918 | 26 | 30.3 min | **25.2 h** |
| v2_engine_ppo_value_run_20260802_054921 | 25 | 31.2 min | **26.0 h** |

26 小时 vs 168 小时上限 —— **6 倍余量**，即使未来 epoch 数翻倍或吞吐退化一半也仍然安全。
（复算方法：`/api/metrics` 取 `rl/avg_episode_reward` 各点的 `wall`，相邻差值即 epoch 耗时。）

## 3. 计费明细（Billing Catalog API 实价，非估算）

机型 `a2-highgpu-1g` = 12 vCPU + 85 GB RAM + 1× A100 40GB，Americas / us-central1：

| 组件 | 按需 | Spot | DWS flex-start |
|---|---|---|---|
| A100 GPU ×1 | $2.93391 | $1.53990 | $1.61365 |
| vCPU ×12 | $0.37932 | $0.19908 | $0.20868 |
| RAM ×85 GB | $0.36040 | $0.18870 | $0.19805 |
| **合计/小时** | **$3.67363** | **$1.92768** | **$2.02038** |
| 相对按需 | — | −47.5% | −45.0% |

对应的 SKU 名称（查询时按此匹配）：

- 按需：`Nvidia Tesla A100 GPU running in Americas` / `A2 Instance Core|Ram running in Americas`
- Spot：`Nvidia Tesla A100 GPU attached to Spot Preemptible VMs` / `Spot Preemptible A2 Instance Core|Ram`
- DWS：`Nvidia Tesla A100 GPU attached to DWS Defined Duration VMs` / `DWS Defined Duration A2 Core|Ram`

> 注意 SKU 命名：A2 机型的 flex-start 计费走的是 **`DWS Defined Duration`** 系列
> （`DWS Flex Start` 前缀目前只覆盖 H4D）；`DWS Calendar Mode` 是另一种模式（预约日历，
> 对应 `--provisioning-model=RESERVATION_BOUND`），不是我们要用的。

复现价格查询：

```bash
gcloud auth print-access-token | xargs -I{} curl -s -H "Authorization: Bearer {}" \
  "https://cloudbilling.googleapis.com/v1/services/6F81-5844-456A/skus?pageSize=5000&currencyCode=USD"
```

（`6F81-5844-456A` = Compute Engine 服务 ID；结果分页，需带 `pageToken` 翻完约 32k 条 SKU。）

## 4. 配额现状（项目 `workstation-185016`）

GPU / TPU 区域配额在 43 个区域高度一致：

| 配额项 | 按需 | Spot / 抢占式 |
|---|---|---|
| **A100 40GB** | **1 / 区域** | **16 / 区域** |
| A100 80GB | 0（全部区域） | 0 |
| H100 / B200 | 无配额项（未开放） | — |
| L4 | 8 / 区域 | 8 / 区域 |
| T4 | 4 / 区域 | 4 / 区域 |
| V100 / P100 / P4 / K80 | 1 / 区域 | 1 / 区域 |
| TPU v5e podslice | 32 / 区域 | 1536 / 区域 |

**当前占用**：us-central1、us-east1、europe-west4 各 1/1 按需 A100 —— **按需配额已打满**，
再加一臂只能换没用过的区域。

**这正是 flex-start 的额外价值**：它消耗**抢占式配额**（A100 池 16/区域，目前一张没用），
不占那个 1/区域的按需名额，因此可以在同一区域并行开更多实验臂，不必再靠换区域扩容
（换区域还有停机丢容量的风险，见 SKILLS.md 的 Ops Lessons）。

### 配额认知修正

`PREEMPTIBLE_CPUS` 在全部 43 个区域都是 **0**，但**这不阻止 Spot**：没有专门的抢占式
CPU 配额时，Spot VM 直接消耗标准配额（`CPUS`=200、`A2_CPUS`=12）。早期笔记里
"no Spot until bumped" 的说法是错的。

真正的每区域瓶颈是 **`A2_CPUS` = 12 vCPU**，恰好一台 `a2-highgpu-1g`（按需 + Spot 合计）。
若要在同一区域同时跑两台 A2，需要申请把 `A2_CPUS` 提到 24，而不是提 `PREEMPTIBLE_CPUS`。

## 5. 怎么用

在 `scripts/phase1_ce/start_vm.sh` 的 create 分支追加以下参数即可：

```bash
--provisioning-model=FLEX_START \
--instance-termination-action=DELETE \
--max-run-duration=36h \
--request-valid-for-duration=2h
```

参数含义与取值理由：

- `--max-run-duration=36h`：安全阀。实测 26h，留 ~40% 余量；到期自动终止，避免脚本异常时空转烧钱。
  **计费按实际运行时长**，不是按申请时长，所以留余量不会多花钱。
- `--instance-termination-action=DELETE`：flex-start 必填。选 DELETE 与现有
  `run_training.sh` 的 EXIT trap（跑完先传 `gs://llm-mahjong-experiments/` 再自关机）天然契合。
- `--request-valid-for-duration=2h`：容量排队上限（全有或全无）。等不到就失败，不会半量启动。
- `--maintenance-policy=TERMINATE` 现有脚本已设置，符合 flex-start 要求。

### 需要相应调整的地方

1. **脚本的「存在就 start」分支不适用**。终止动作是 DELETE，VM 跑完即消失，每次都是全新创建；
   flex-start VM 若被停机后重启，配额不足会**立即失败**（不像创建会排队）。
2. **不能热迁移、不能用预留、不能用 spread 放置策略**。
3. 结果落盘依赖 GCS 上传（已有），不要把唯一副本留在本地盘。

## 6. 待验证 / 风险

- **`PREEMPTIBLE_CPUS = 0` 是否会拦住 flex-start**：官方文档称 flex-start 需要「抢占式配额」。
  Spot 在同样为 0 的情况下会回退到标准配额（已被实际使用验证），flex-start 大概率同理，
  但**尚未实证**。首次创建时留意是否报配额错；若报错，申请提升目标区域的 `PREEMPTIBLE_CPUS`。
- **容量排队**：flex-start 不保证立刻拿到卡。A100 在 us-central1-b/-f 有容量记录，
  但 stockout 随时可能发生（2026-08-01 us-central1-a、us-west1-b 都出现过）。
- **7 天上限**：若将来 epoch 数或模型规模大幅上调，需重新核算 run 时长是否仍在窗口内。

## 7. 顺带结论：G4（RTX PRO 6000）不适合当前负载

调研过 G4 系列（`nvidia-rtx-pro-6000`，us-central1-b 有货）：

- `g4-standard-48` = 1 整张 RTX PRO 6000 96GB，按需约 $4.50/h；`g4-standard-12` 是 **1/4 张卡**（24GB 切片）。
- 整卡带宽约 1790 GB/s vs A100 的 1555 GB/s（+15%），但价格 +23% —— 单位带宽成本反而更差。
- 我们的负载是 **rollout 解码带宽瓶颈**、模型仅 2B（bf16 权重 ~4GB），40GB 显存远未用满，
  吃不到 96GB 大显存和 Blackwell FP8/FP4 的红利。
- 迁移还有成本：`causal-conv1d` / `fla` 快速路径是按 sm_80 编译打包的，上 Blackwell（sm_120）要重编验证。

**何时再考虑 G4**：模型升到 7B+、或大幅加大 `parallel_games` / 上下文长度时；
以及作为独立容量池，在 A100 按需配额打满时应急扩容。

## 参考

- [About flex-start VMs](https://docs.cloud.google.com/compute/docs/instances/about-flex-start-vms)
- [Create flex-start VMs](https://docs.cloud.google.com/compute/docs/instances/create-flex-start-vm)
- [DWS 概念（GKE 文档，折扣说明最完整）](https://docs.cloud.google.com/kubernetes-engine/docs/concepts/dws)
- 价格数据：Cloud Billing Catalog API，service `6F81-5844-456A`，2026-08-02 拉取


## 配额提升（2026-08-09，Cloud Quotas API 即时批准）
| 配额 | us-central1 | us-east1 | europe-west4 |
|---|---|---|---|
| A2_CPUS | 12→**48** | 12→**48** | 12→**24** |
| NVIDIA_A100_GPUS（按需） | 1→**4** | 1→**4** | 1→**2** |

- 效果：每个美区可并发 **4 台** a2-highgpu-1g（原 1 台），三区合计上限 10 台；多臂实验不再被迫跨区分布。
- PREEMPTIBLE_NVIDIA_A100_GPUS 维持 16/区（flex-start 抽此额度，A2_CPUS 才是真闸门——现已放宽）。
- A100 80GB 配额仍为 0；若未来大 batch SFT 需要（40GB 上 batch 8 OOM 实录见 exp3），另行申请 `NVIDIA-A100-80GB-GPUS-per-project-region`。
- 申请方式：`gcloud alpha quotas preferences create --service=compute.googleapis.com --quota-id=... --preferred-value=N --dimensions=region=...`（不带位置参数名；本次全部秒批）。
