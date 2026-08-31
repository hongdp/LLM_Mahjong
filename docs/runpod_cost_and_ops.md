# RunPod 算力选项：价格、与 GCE 的差异、踩过的坑

> 2026-08-30 由 web_canvas 项目实测后移交。该项目为验证云推理充值了 RunPod，
> 两次实验共花约 $5 后结论为不采用（量化模型质量不敌 grok），**账户余额留给本项目消化**。
> 所有数字均为当日实查（catalog API / 真实 pod），非官网宣传页。
> 余额只能在 [console](https://www.runpod.io/console) 看——API 不暴露；计费 API
> (`get-billing`) 有约小时级延迟，当日实花 ~$5 时它只报 $0.55，别拿它做实时核对。

## 1. 结论先行：A100 一小时比 GCE DWS 便宜 21%，显存还翻倍

| | GCE DWS flex-start（现行默认） | RunPod secure | RunPod community |
|---|---|---|---|
| 卡 | A100 **40GB**（a2-highgpu-1g 整机） | A100 SXM **80GB** | 同 |
| 单价 | $2.020/h | **$1.59/h（−21%）** | $1.39/h（−31%） |
| 一次 26h run | $52.5 | **~$41** | ~$36 |
| 配额 | 按需 1/区域已打满；抢占式池 16/区域 | **无配额概念**，有货就开 | 同 |
| 时长上限 | 168h（flex 硬上限） | 无 | 无 |
| 中断 | 7 天内不被抢占 | on-demand 不被抢占 | 不被抢占，但主机是散户机器，硬件故障率未知 |
| 计费粒度 | 按秒（VM 运行中） | **按秒，无最低消费** | 同 |
| 库存（2026-08-30） | 排队看运气 | **HIGH** | HIGH |

其它可能相关的卡（secure 价 / 库存，均为实查）：

| 卡 | 显存 | $/h secure | 库存 | 备注 |
|---|---|---|---|---|
| A40 | 48GB | **$0.44** | HIGH | 若瓶颈在 CPU 模拟而非 DNN，这是 1/4.6 价的选项 |
| RTX A6000 | 48GB | $0.53 | LOW | 同上 |
| L40S | 48GB | $0.99 | HIGH | Ada，FP8 |
| RTX PRO 6000 | 96GB | $2.09 | HIGH | web_canvas 实测过的卡（Blackwell, sm_120） |
| H100 SXM | 80GB | $3.29 | MEDIUM | |
| H200 SXM | 141GB | $4.59 | HIGH | |
| A100 80GB PCIe | 80GB | $1.39 | LOW | 比 SXM 便宜但库存差 |

多卡：A100 SXM 单 pod 可到 8 卡（NVLink 拓扑没验证过）；跨机集群走
Instant Clusters，本项目没试过，多机 NCCL 需求出现时另行验证。

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
