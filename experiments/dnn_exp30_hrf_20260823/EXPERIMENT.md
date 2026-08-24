# （exp30 主臂：HandRiverFormer × 冠军时间表 × 纪元 4）

- 预注册: experiments/exp30_hrf_prereg/EXPERIMENT.md；对照 = exp31-5（1031.9±12.2 同配方 cnn_m_r）
- 配方来源: exp31 结论（目标熵失败、恒低熵不优 → 冠军时间表 0.03×600k→0.01）
- 2026-08-24 收官：**T1 = 852.1 ± 12.4**——与 cnn_xl 恒定低熵的 851.7 几乎同一数字。
  风格：riichi 2.1% / call 92.7% = **副露锁死**；agari 仅 11.3%、houjuu 8.1%（极端被动）。
  decomposition 也弱（greedy_agree 0.46）。**判定：HRF × 冠军 schedule 在 1.0M 预算内失败**，
  锁死病征与 cnn_xl/gen1c/exp31-6 同族。
  - defense_iq 0.064 为史上最高读数，但**判读需谨慎**：模型整体极端被动（agari 11.3%），
    高弃胡率可能是"弱而怂"的副产物而非防守能力——defense_iq 在低强度区间与被动性混淆，
    不作为防守涌现证据（对比 gen1b：defense_iq +0.025 同时保有 18.4% agari，那才是有效信号）。
  - **与 exp36 的对照意义重大**（同架构唯一变量=熵机制）：exp30（schedule）锁死在 852，
    exp36（恒定 0.01，仍在跑 @1.49M）已走完自组织逃逸全程（熵平台 0.70 → riichi 0.1%→20.8%、
    call 98.9%→21.7%、win_turn 15.9→13.5、熵变现段 0.70→0.51）。**在 HRF 上恒定低熵优于冠军
    schedule**——方向与 cnn_m 上完全相反（exp31-6 恒定锁死、schedule 成功）。架构×熵机制存在
    强交互：注意力架构自带的探索自组织会被过高的早期熵系数（0.03 段）破坏。待 exp36 终局 Elo 定量。
