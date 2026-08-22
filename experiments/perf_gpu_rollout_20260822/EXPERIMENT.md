# perf：GPU 批量推理服务（rollout 第二阶段，scale-up 前置）

- **Date**: 2026-08-22  **Status**: running（实现完成，吞吐基准进行中）
- **Git**: （本提交）  **Env**: 本地 24 vCPU + RTX 4080（与 llama-server 共享 GPU）；云端 g2 待验

## Purpose
20M 级模型在 CPU 逐次推理下 b1 17ms → 30 workers 仅 3.6 局/s（700k 局 ≈ $108）。GPU 批推理实测
天花板：192×40 在 4080 上 B=1..128 延迟几乎不变（~8ms，kernel 启动受限），B=512 达 30k 决策/s。
目标：同一台 g2 上把大模型 rollout 提 30-50×，让 scale-ladder 与 exp24（回放类方法）可负担。

## Design（docs/design_input_v3_scale_dqn.md §B 的落地）
跨进程 RPC，对局循环零改动：worker 的 `net.act()` 被 `RemotePolicy` 替换——写本 worker 的共享内存
槽位 → 入队 → 阻塞等 Event；一个 spawn 出的服务进程凑批（首请求后再等 ≤wait_ms 或 max_batch）
→ 一次前向 → GPU 采样 → 写回 → set Event。`cfg["gpu_infer"]=True` 即启用；服务在 fork worker 前
由父进程启动，句柄经 fork 继承（不经 Pool 参数 pickling）。
踩坑：①stdlib `multiprocessing` spawn 会把 torch 共享张量整体拷贝 → 改 `torch.multiprocessing`
（file_system 策略）；②spawn 子进程重 import `__main__`，stdin 脚本会卡死 → 诊断必须是文件；
③加了死锁护栏：服务异常时写 −1 并 set 全部 Event，客户端 5s 轮询检查服务存活。

## Baseline（本机）
- 单客户端热态往返 **2.37 ms/act**（含 1.0ms 凑批窗口）；cnn_m CPU 路径 16 workers 39.5 局/s。

## Success Criteria
1. cnn_m：GPU 路径 ≥ CPU 路径吞吐（不退化）。
2. 192×40：GPU 路径 ≥ 10× CPU 路径；worker 超额订阅（48）进一步提升。
3. 统计等价：同种子 CPU/GPU 两路的自对弈决出率/结算分布无显著差异（非逐字节）。

## Progress
- [2026-08-22 16:10] 实现 + 护栏 + 诊断通过；基准链（cnn_m w16/w32、192×40 CPU/GPU w16、GPU w48）后台运行。

## Results
（待运行）

## Conclusion
（待运行）
