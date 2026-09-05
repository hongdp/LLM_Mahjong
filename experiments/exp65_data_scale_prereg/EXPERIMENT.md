# exp65 数据 5–10×：凤凰卓 2026 全年段 BC 重训（文献对照的第一杠杆）

- **Date**: 2026-09-04  **Status**: running（阶段 1 采集中）
- **Git**: 见 Progress；采集在工作站（`tools/tenhou/fetch_houou.py`，单会话、1 请求/秒、只取 .log.gz、不再分发）；训练上 RunPod Secure
- **Env**: 数据 `data/tenhou/raw/<date>/*.mjlog`（主检出）；训练 `train_human_bc.py`（bc49 配方，v3r + mortal46）

## Purpose & Hypothesis
exp49：9× 数据 = +3.7–4.0pp 精度 = +27–57 Elo，18.4k 局未饱和；文献（Tuyls 2023、Jones 2021、AlphaGo SL）一致：模仿学习的
数据缩放是幂律且顶部精度→强度是凸的。假设：把 2026-01-01 至 2026-09-03 的凤凰卓四人南喰赤全部取下（≈500 局/天 ≈ **12 万局**，
现有 2.09 万局的 ~6×），同配方重训 → holdout 精度 **≥ +2pp**，T=0 配对牌山对 bc49 **≥ 0.53**。
反假设：凤凰卓人类噪声地板已近（Suphx 15M 样本 76.7% 切牌精度），增益 <1pp。

## Method
阶段 1（采集，2026-09-04 起）：`fetch_houou.py --start 20260101 --end 20260903`，幂等续传；ToS：单会话、≥20 分钟轮询索引、
每日索引取一次、逐局 1 秒间隔（实测约 0.3–0.5 局/秒 → 12 万局约 3–4 天）；官方 2025 及更早的日归档已不再提供（探针 0 局），
不用第三方再分发包。
阶段 2（训练）：同 bc49 配方（`convformer_m_v3r_m46`，holdout 按 hash 10% 上限 1000 局同旧集，max_epochs 30 / patience 3 /
min_delta 5e-4，batch 1024，lr 3e-4）。数据量 ~78M 决策/epoch：pod 上物化 uint8 缓存（~150GB 盘）或流式；估 L40S 8–14 小时
≈ **$10–16**。按 exp62 教训：同批控制臂 = 旧 2.09 万局子集原配方重跑（同 pod、同代码），双方各 1 种子（预算），差异判据设在噪声地板之上。
阶段 3（可选，若 exp64 成立）：在新模型上做高分席位微调。

## Success Criteria（预注册）
1. 新模型 vs 同批控制臂：holdout acc ≥ +2pp（同一 holdout 集）；T=0 配对牌山 n=4000（再补 16000）share ≥ 0.53 → 数据缩放兑现，
   **冠军候选**：走加冕流程（docs/champion_model.md §7，含 Mortal 半庄头对头 n≥1200 与雀魂 maka）。
2. +1–2pp 且 share 0.51–0.53：兑现但斜率放缓，记录缩放曲线点（2k / 18k / 120k）。
3. <+1pp：饱和，数据线关闭，剩余杠杆只剩搜索蒸馏。

## Progress
- [09-04 00:45] 探针：2025-03-01、2025-10-01 日归档 0 局（不再提供）；2026-01-15 474 局 → 可取范围 2026-01-01 起。
  采集启动（pid 见 scratchpad `harvest65.log`），2026-01-01 570 局。
- [09-05] **采集完成**：2026-01-01 → 09-03 共下载 93,481 局，`data/tenhou/raw` 合计 **114,803 局**（≈ 旧集 20,526 的 5.6×），
  0 个 warn；单会话 throttle 0.4 s，约 26 小时。holdout 冻结为旧集的 1000 局（`experiments/configs/exp65_holdout_1000.txt`）。
  发射：L40S Secure US-TX-4，220GB 盘（全量 uint8 缓存 ≈ 145GB + 旧集 26GB），两臂并行：F = 全量 114.8k 局、C = 旧快照 20.5k 局（同批控制臂）。

## Results

## Conclusion

## Next Steps

## Artifacts
