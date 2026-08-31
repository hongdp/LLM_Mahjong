# RunPod 算力选项：价格、与 GCE 的差异、踩过的坑

> 2026-08-30 由 web_canvas 项目实测后移交。该项目为验证云推理充值了 RunPod，
> 两次实验共花约 $5 后结论为不采用（量化模型质量不敌 grok），**账户余额留给本项目消化**。
> 所有数字均为当日实查（catalog API / 真实 pod），非官网宣传页。
> 余额只能在 [console](https://www.runpod.io/console) 看——API 不暴露；计费 API
> (`get-billing`) 有约小时级延迟，当日实花 ~$5 时它只报 $0.55，别拿它做实时核对。

## 1. 结论先行：按保证 vCPU 算，RunPod 打不过 g4-standard-48（2026-08-30 补查后反转）

> 初版此节以为同卡便宜 7–25% 即赢，补查定价页的 per-GPU vCPU 配额后反转：
> **RunPod 的 pod 是 GPU 富裕、CPU 贫穷型,而本负载（rollout CPU 瓶颈）恰恰相反。**
> g4-standard-48 的 48 保证核在市面上是罕见配置，正好长在负载的点上。

每 GPU 保证配额（定价页，2026-08-30）：A40 = **9 vCPU**/50GB；L40S、A100 SXM、
RTX PRO 6000 = **16 vCPU**；H100 = 20。按 g4 实测折算（279 局/s ÷ 48 核 ≈ 5.8 局/s/核）：

| | vCPU/$ | 估算局/s | 局/s 每 $ | 单 run 墙钟 |
|---|---|---|---|---|
| **g4-48 flex（现行默认）** | **21.3** | 279（实测） | **124** | 26h |
| A40 secure $0.44 | 20.5 | ~52 | 119 | ~140h |
| A40 community $0.35 | 25.7 | ~52 | 149 | ~140h |
| L40S secure $0.99 | 16.2 | ~93 | 94 | ~78h |
| RTX PRO 6000 secure $2.09 | 7.7 | ~93 | 44 | ~78h |

- **A40（"是不是更划算"）**：不。便宜的是 GPU，要买的是 CPU（9 核）。community 价每美元
  +20%，但单 run 5.4× 慢——本项目已有先例（flex vs spot，$7 换不抢占）说明迭代速度值钱。
- **同卡（RTX PRO 6000）**：保证核只有 g4 的 1/3，每美元 44 vs 124，判负。

### 1.2 真正的性价比赢家：RTX 3090 community（16 核/$0.22），叠 3 张凑 48 核

vCPU 配比在消费卡上极不均匀（定价页标称）：3090 = **16 vCPU + 125GB RAM**，而更贵的
4090 只配 6 核、5090 只配 9 核。于是：

| community | 保证 vCPU | $/h | vCPU/$ | 估算局/s | 局/s 每 $ | 单 run 墙钟 |
|---|---|---|---|---|---|---|
| g4-48 flex（基线） | 48 | $2.25 | 21.3 | 279 实测 | 124 | 26h |
| **RTX 3090 ×1** | 16 | $0.22 | **72.7** | ~93 | **~422** | ~78h |
| **RTX 3090 ×3（一个 pod）** | **48** | **$0.66** | 72.7 | **~279** | ~422 | **~26h，与 g4 平** |
| RTX A5000 ×1 | 9 | $0.16 | 56 | ~52 | ~326 | ~140h（LOW 库存） |

3×3090 = **g4 同等核数、29% 的价**：单臂 $52.5 → **~$17**，三臂舰队 $158 → **~$46**，
墙钟不牺牲。推理用一张卡就够（3090 带宽 936 GB/s ≥ 本机 4080 的 717，前向不会封顶；
handset_xl 在 L4 上封顶是因为 L4 只有 300 GB/s），另两张闲置不影响账。3090 是 sm_86，
cu129 wheel 原生支持，没有 Blackwell 编译事。

**四个前提，任一不成立就翻案，全部可用一个 ~$0.25 探针验证**（$0.22/h 开一台跑
rollout 基准对 279 局/s）：
1. 每核速率——5.8 局/s/核来自 g4 服务器核；community 宿主多为桌面 U，可能更快
   （本机桌面 24 核实测 8.5/核）也可能更慢。最大不确定项。
2. community 长跑可靠性——不抢占但散户机器会坏，`--resume` 必须真可靠。
3. vCPU 标称 vs 实配——community 每台宿主实际核数在 console 部署页可见，下单前核对。
4. 库存 MEDIUM（非 HIGH），多卡 pod 上限 6 张。

- **burst 变数**（对所有卡成立）：实测 pod 内 `nproc`=128、`-j128` 编译 llama.cpp 全套
  3.5 分钟——cgroup 放行宿主闲置核，实际吞吐可能远超保证值，但 burst 不是合同。
- 若 3090 探针不达标，余额退而求其次的花法是**短平快、按秒计费占优的活**：elo 梯子
  评测、一次性基准、临时大显存探针。

### 1.1 原始价格表（负载性质变化时参考）

基线：g4-standard-48 flex = RTX PRO 6000 96GB + **48 保证 vCPU**，$2.25/h，按需 $4.50
不可用（gcp 文档 §8）。RunPod 同卡 $2.09 secure / $1.69 community、HIGH 库存、无 168h
上限、stop/start 随意（磁盘费率见 §2）——单价确实低，但 vCPU 见上表。

| 卡 | 显存 | $/h secure | 库存 | 备注 |
|---|---|---|---|---|
| A100 SXM | 80GB | $1.59 | HIGH | 16 vCPU；显存型负载才有意义（§8 已判负 12 核的 GCE A100） |
| A40 | 48GB | $0.44 | HIGH | 9 vCPU——见 §1 的判决 |
| L40S | 48GB | $0.99 | HIGH | |
| H100 SXM | 80GB | $3.29 | MEDIUM | |
| H200 SXM | 141GB | $4.59 | HIGH | |

多卡：单 pod 最多 8–9 张（NVLink 拓扑没验证）；跨机走 Instant Clusters，没试过。

## 2. 与 GCE 语义上的关键差异（不是价格，是模型）

- **pod 是容器，不是 VM**。`runpod/pytorch` 镜像里 PID 1 是 docker-init，
  `kill 1` 即优雅退出并停止计费——自停脚本靠这个实现，不需要把 API key 放上机器。
  没有 systemd；`shutdown -h now` 之类的 VM 习惯无效。
- **stop ≠ 省钱**。GPU 计费停了，但 container disk 从 $0.10/GB/mo 变
  **$0.20/GB/mo（费率翻倍）**——400GB 盘停着放一个月是 $80。铁律：**用完删
  （terminate），别停（stop）**。checkpoint 必须在删之前推到 GCS（egress 免费，
  实测下行 ~250MB/s 单流、~1.2GB/s 三流并行；上行同量级）。
- **network volume**（$0.07/GB/mo，停机同价）只在「每次开 pod 都要同一批大文件、
  且重下载比 $/月更贵」时才值得。训练 checkpoint 走 GCS 即可，不需要它。
- **端口每次启动重新映射**。`portMappings` 是 per-run 状态，pod stop→start 后公网
  端口会变。任何写死地址的脚本在第一次重启后就是错的。
- **`desiredStatus=RUNNING` 是意愿不是事实**——start 被接受即翻 RUNNING，此时机器
  可能还在调度。就绪只认健康探测。
- **无配额也意味着无预约**：库存实时波动，发射前查一次
  `list-gpu-types`/`get-capacity`（LOW 的卡可能实际开不出来）。

## 3. 踩过的坑（每条都真实花了时间或钱）

1. **SSH 有两个入口，认证源不同**。`ssh <podid>-xxx@ssh.runpod.io`（proxy）只认
   **console 账户设置里注册的 key**；创建 pod 时传的 `sshPublicKey` 只进容器的
   `PUBLIC_KEY` 环境变量，只有**直连入口**（`ssh root@<ip> -p <port>`）认。
   直连地址在创建响应里是 null，要等 `runtime` 就绪后重新 GET pod 才有（~1 分钟）。
   用 proxy 入口 + 注入 key 会永远 `Permission denied`，且被 `2>/dev/null` 吞掉后
   与「还在启动」无法区分——白烧了 5 分钟计费。轮询类脚本**必须让第一次失败的
   stderr 可见**，并把 auth 类错误当致命而非可重试。
2. **`runpod/pytorch` 镜像有 nvcc 但不在 PATH**（`/usr/local/cuda-12.8/bin/nvcc`）。
   `which nvcc` 为空不代表要装 toolkit——差点为此多下 3GB。
3. Ubuntu 24.04 的 pip 要 `--break-system-packages`；`huggingface_hub[cli]` 的
   extra 在 1.x 已删除（警告但不装）——要 `hf` 命令直接装 `huggingface_hub`。
4. **余额为零时 catalog/billing 全部正常工作**，只有 create-pod 给 402。
   「key 能用」证明不了「能开机器」。
5. 下载大文件记得 `HF_HUB_ENABLE_HF_TRANSFER=1`（多连接下载器，装 `hf_transfer`）。

## 4. 怎么用

- **API key**：在 console → Settings → API Keys 管理。工作站上 Claude 的 runpod
  MCP 插件已配好一把（create-pod / list-pods / get-billing 等工具可直接用）；
  脚本化用 REST v2：`https://api.runpod.io/v2`，Bearer 认证，schema 在
  `/v2/openapi.json`。**key 不入库**。
- **最小 create-pod**（web_canvas 验证过的参数形态）：
  `gpuTypeIds=["NVIDIA A100-SXM4-80GB"]`, `cloudType="SECURE"`,
  `imageName="runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"`,
  `containerDiskInGb=<够放数据+ckpt>`, `ports=["22/tcp"]`,
  `sshPublicKey=<你的 id_rsa.pub 内容>`。
- **自停保险**（防训练挂死后空烧，pod 无 systemd，直接 nohup）：
  ```bash
  # 监控训练日志 mtime，静默 N 分钟即停止计费；kill 1 = 容器退出 = 停表
  nohup bash -c 'while sleep 60; do
    [ -f /workspace/train.log ] || continue
    idle=$(( $(date +%s) - $(stat -c %Y /workspace/train.log) ))
    [ $idle -ge 3600 ] && { sync; kill 1; }
  done' &
  ```
  阈值按训练相设：发射相另设短死线（同 SKILLS.md 2026-08-30 心跳分相的教训）。
- web_canvas 侧还有可抄的现成件：`scripts/cloud_endpoint/controllers.py` 的
  `RunPodController`（start/stop/状态/地址发现，纯 REST 无依赖，9 个测试）、
  `validate_pod.sh`（租前验证的结构：并行下载+编译、失败阶梯）。

## 5. 不适用 / 未验证

- 多机 NCCL（Instant Clusters）没碰过。
- community 云的长跑可靠性（26h 不断）没验证——首次长跑建议 secure，跑通后再拿
  一臂试 community 省 $5/run。
- Spot（RunPod 也有，更便宜可抢占）没查价——on-demand 已比 GCE DWS 便宜，
  抢占换小钱在本项目已有先例结论（见 gcp 文档 §1）。
