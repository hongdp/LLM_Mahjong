# exp23：输入扩展 v3（完整公开对局记录，零派生特征）——单变量 vs GAE 冠军

- **Date**: 预注册 2026-08-22  **Status**: launching
- **Git**: bf12fd4  **Env**: mahjong-dnn-c3（us-east1-b）on-demand g2
- **对照**: exp17-C = `dnn_exp17c_gae_20260818`（同协议共享基线，1079.7；唯一差异 = 编码器 v1→v3）
- **设计**: docs/design_input_v3_scale_dqn.md §A；实现 encoder.py `_encode_v3`（50 平面 + 29 标量）

## Purpose & Hypothesis
用户指令「减少手工特征工程、不遗漏有效信息」。v1 编码为 LLM 信息对等而有意瘦身，遗漏了
牌河顺序、**立直宣言牌位置**、**摸切/手切**、被鸣走、副露来源、立直巡目、可见牌计数——
全部是桌面公开事实（零麻将知识）。假说：①补齐公开事实提升强度（Elo ≥ +40）；
②**读牌信息到位后防守变得可学**（defense_iq 显著转正）——exp21 诊断链的输入层假说。

## Method / Config
exp17-C 协议原样（cnn 64×3、GAE 0.95、熵台阶 0:0.03,600000:0.01、700k、seed 42）
+ **唯一差异 `--arch cnn_m_v3`**（同架构，in_planes 15→50、in_scalars 20→29；参数 1.95M≈1.94M）。
引擎侧公开事实记录为纯追加（贪心轨迹 hash 逐字节不变、134 测试）。

## Success Criteria（发射前定死）
1. **主判定**：700k vs exp17-C-700k 200 副复式 + 正式评分 ≥ 1079.7+40（≈z 2）。
2. **防守判定**：defense_iq（800 局探针）显著 >0（≥ +0.03，SE≈0.015）⇒ 输入层假说成立；
   仍≈0 ⇒ 输入不是防守的瓶颈（同样定论，把嫌疑收敛到生态/梯度竞争）。
3. 健康：吞吐 ≥20 局/s（编码器 v3 成本）、KL/熵正常、跑满。

## Progress
- [2026-08-22] 预注册；c3 发射（c3 自 8/19 关停，非关机窗口）。

## Results
| Metric | This run (v3) | Baseline (exp17-C, v1) | Success criterion |
|---|---|---|---|
| （待运行） | | | |

## Conclusion
（待运行）

## Artifacts
| Path | Description |
|---|---|
| gs://llm-mahjong-experiments/dnn_exp23_inputv3_20260822/ | 云端主目录 |
