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
- [2026-08-22 16:10] 实现 + 护栏 + 诊断通过；基准链后台运行。
- [2026-08-22 16:50] 首轮基准卡死根因：**spawn 服务进程重导入无 `__main__` 守卫的基准脚本**，在服务
  进程里递归启动服务（诊断脚本有守卫故正常）。已修；规则：任何启用 gpu_infer 的入口必须有守卫
  （trainer 已有）。附带教训：`pkill -f` 的模式若出现在当前 shell 命令行里会自杀，扫描要放脚本文件。
- [2026-08-22 16:55] **v1 基准（本机 24 vCPU + 4080，GPU 与 llama-server 共享）**：cnn_m CPU 40.2 局/s
  vs GPU w16 25.6 / w32 29.2（小模型 RPC 往返 > 0.65ms 前向，走 CPU 正确）；**192×40 CPU 3.5 →
  GPU w16 9.2（2.6×）→ w48 16.5 局/s（4.7×）**，1472 决策/s 远低于 GPU 理论 ~6000/s ⇒ 瓶颈在服务
  进程每批开销（Queue 逐条 get 锁争用、1ms 窗口切短 batch、逐个 Event.set）。v2：共享标志数组 +
  单信号量、窗口自适应（≥半数槽位或 4ms）。

- [2026-08-22 18:00] **v2 基准**（共享标志数组 + 单信号量 + 自适应窗口）：192×40 w48 14.7 / **w96 25.4 局/s
  （CPU 3.5 → 7.3×）**，w96+8ms 窗口 24.7（窗口无关）；cnn_m w32 32.4（CPU 40.2，小模型仍走 CPU）。
  服务端计时（192×40 w96，avg batch 61）：wait 1.46 / drain 0.01 / gather 0.27 / **fwd 11.1** /
  write 0.02 / **signal 3.37** ms/批，循环内 16.2ms；而每批周期 27ms ⇒ 服务 ~40% 时间在
  `sem.acquire` 空等 worker 回流（96 进程在 24 核上的唤醒风暴）。GPU 此时空闲（llama-server 0%）。
  v3 方向：①单次 `Condition.notify_all` 取代 61 次 Event.set；②cudnn.benchmark + CUDA graph
  分桶（40 块小卷积 = kernel 启动受限）；③双缓冲：收集下一批与 GPU 前向重叠。

- [2026-08-22 18:40] **v3**（单次 `Condition.notify_all` + done 代计数）：signal 3.4→0.27 ms、wait 1.5→0.3 ms
  ✅；但 cudnn.benchmark 在批尺寸漂移下反复自调 ⇒ fwd 11→14.5 ms，吞吐 25.4→22.3（已改为默认关）。
  结论：服务端非 GPU 开销已清零，**瓶颈 = 前向本身 11-14 ms/批，与批尺寸无关 ⇒ kernel 启动受限**。
- [2026-08-22 18:55] **v4 CUDA graph 分桶回放**已实现（eager 回退）；本机无法验证——4080 被 llama-server
  占 14.2GB，捕获 OOM。待云端 L4 验证（exp24/scale-ladder 的首个 run 顺带测）。

## Results
（待运行）

## Conclusion
（待运行）
