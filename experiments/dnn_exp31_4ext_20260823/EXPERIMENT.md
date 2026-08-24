# dnn_exp31_4ext_20260823 — exp31-4（handset lowent）续训 1.0M→2.0M

- **Date**: 2026-08-23  **Cost**: G4 flex ~$11（前身实测 57.3 games/s → +1.0M 局 ≈ 4.9h）  **Status**: launching
- 前身: `dnn_exp31_4_handset_lowent_20260823`（1.0M 收官 T1 1017.5；熵 0.579 仍在自组织高原；
  竞技场斜率末段仍 +28/270k → 未收敛）。

## Purpose & Hypothesis
1.0M 同预算对比对大模型（handset 6.6M+）天然不利——它的熵高原还没变现。续训回答两个问题：
(a) 自组织熵高原是否最终"收割"（熵下降 + Elo 上冲）；(b) handset 架构天花板是否高于 cnn_m_r 配方。
呼应规矩：不得在未确认训练效率信号前判 scale/架构无用。

## Method
`--resume`（games=1001472，optimizer/entropy_alpha 全量恢复），配方与前身完全一致
（handset_xl_cnn_m_r，entropy_coef 0.01 恒定），仅 `--total_games 2000000`，
milestones 1300000,1600000,2000000。ckpt 经 `gs://llm-mahjong-experiments/resume/` 上 VM。
对照 = 前身 1.0M 数（同模型同线，纯纵向比较）。

## Success Criteria
1. **架构有效线**：2.0M 时 T1 ≥ 1032（追平 exp31-5 冠军配方同纪元数）。
2. **高原变现观察**：若熵开始下行且 Elo 同步上冲 → 记为孵化理论正面证据；
   若 2.0M 时熵仍 >0.5 且 Elo 斜率仍正 → 判"仍未收敛"，另行决策是否再喂。
3. 风格三件套（houjuu/tsumo/riichi）随续训的漂移入记录（它的低铳高摸风格是否保持）。

## Progress
- 2026-08-23：发射于 mahjong-g4-ext（us-central1-b，SHA 0de2031，CKPT_OK 240MB）。
  吞吐期望 = 前身 57.3 games/s（±20% 内为正常）。
