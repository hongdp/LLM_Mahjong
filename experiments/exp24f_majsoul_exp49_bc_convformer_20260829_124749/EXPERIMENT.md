# exp24f_majsoul_live — 雀魂实战（exp49 B：人类 BC 旗舰 conv×v3r×46）

- **Date**: 2026-08-29 12:47  **Status**: done（约 9.9 小时，1659 决策，已切换到 exp24g）
- **Git**: beffea5
- **模型**: `experiments/exp49_20260827_205132/B/bc_convformer_m_v3r_m46_best.pt`
  = 人类牌谱行为克隆旗舰 **conv×v3r×46**（ConvFormer-m，编码器 v3r，**46 槽 Mortal 式动作空间**，2.0M 参数，
  bc_acc 0.806，epoch 6/18458 局训练数据）。**这不是纯自对弈冠军谱系**（exp27-A）——按项目北极星
  （AlphaZero 志向：零人类先验）规矩，人类 BC 模型只入 Elo 锚点池当基准/陪练，不进训练对手池，
  但**是当前 Elo 池史高**：全量数据版 1191.4，比纯自对弈冠军 exp27-A 高 +139（见 exp49_scale_hparam_prereg）。
- **动作空间桥接（本次新增基建）**：exp49 之前的所有雀魂部署都是「原生 374/272 槽」模型，
  `DnnPolicy` 直接单步查询。这个 checkpoint 是训练时用 `action_space.py` 的 `MortalActionSpace`
  （46 槽，立直/杠是「先宣告再选牌」两步查询），`mjai_bridge.py` 之前没有接这个抽象——已改造
  `DnnPolicy.__call__` 用 `action_space.get_space(net)` 透明分发（原生模型走一步，46 槽模型自动
  二次查询），复用训练侧已测试的 `action_space.py`/`mortal_action.py`（`tests/test_action_space.py`
  `tests/test_mortal_alignment.py`），未新写任何动作语义。
- **⚠️ 端口冲突**：发现另一进程已占用默认 8765（另一 worktree
  `mahjong-agent-integration-69f9e5`，跑 exp17-C，**当前正在真实对局中**，已有 7 次决策、日志刚更新），
  未打断它。本次改用 **端口 8766**。MahjongCopilot 若要接这个新模型，`llmmahjong_url` 需指向
  `http://127.0.0.1:8766`（或先确认那局结束后再切端口复用 8765）。
- **模式**: 贪心 T=0，device cpu（仅 2.0M 参数，4 ms/决策，不需要 GPU；GPU 当时已被占 14.9/16.4GB）
- **前置校验**: `verify_mjai_bridge.py --games 30 --device cuda` OK（416 决策张量/合法集逐位一致，
  含二次查询的立直/杠路径）；`test_mjai_bridge.py`/`test_action_space.py`/`test_mortal_alignment.py`
  共 38 项全绿。

## Purpose
迄今雀魂实战都用纯自对弈谱系（exp17-C/exp27-A/exp31-4ext）。这次换成 Elo 全池最强的人类 BC 旗舰，
看它在真人对局中的推图/防守风格与纯自对弈谱系的差异（自对弈谱系普遍立直率 0-36%、副露率 0-60% 大幅波动）。
**不代表北极星路线的强度判定**——它训练时见过人类先验，只是"当前最强可用模型"的实战抽样。

## Progress
- [12:47] 服务启动（8766），保真校验通过；MJAI 动作空间桥接改造完成并测试通过。
- [12:52] 用户确认切换：停掉 8765 上正在真实对局中的 exp17-C（另一 worktree
  `mahjong-agent-integration-69f9e5` 的 exp24_majsoul_live_20260829_124150 会话，停时 18 次决策），
  把 exp49-B 从 8766 迁到 **8765**（打牌机零配置改动，继续用原 URL）。原 8766 已释放。
  **风险已告知并确认**：若 8765 那局尚未结束，会中途切模型继续打完，风格突变、数据不纯净——
  用户明确同意接受。
- [22:46] 用户要求换用 exp46I_recipe 最新 checkpoint，会话在此结束（服务停，接 exp24g）。
  总计运行 ~9.9 小时，1659 次决策，全程 0 新增错误（45 条历史 error 全部来自切换那次的
  已知 3 分钟窗口，此后再无）。详见 analysis_final.txt。
