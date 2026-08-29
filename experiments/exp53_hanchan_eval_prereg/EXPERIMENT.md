# exp53（预注册）— 半庄评测层：顺位口径的对战与 Elo（评估先行，训练后扩）

- **Date**: 2026-08-29  **Status**: building  **Cost**: 本地 CPU 为主
- **背景**: 用户指令。现行单局 sign-of-diff Elo 的两处失真（打点符号压缩、无顺位结构）中，
  本实验解决 ②：以真实半庄（E1..S4、连庄/本场/供托、uma 顺位分）作评测单元。
  引擎零改动（指纹不动）：场况经 HanchanTable 子类注入，本场支付/状态机在驱动器侧记账。

## Method

- `src/tasks/mahjong/hanchan.py`：HanchanTable（指定 dealer/场风/分数/供托开局）+
  HanchanDriver（连庄=亲和或亲听牌流局；本场支付 ron+300×n / tsumo+100×n×3 驱动器侧；
  供托跨局携带、和者取；击飞即终局；v1 无西入、无 dealer-top-stop，如实记）。
  每小局复用 `play_game_mjai(observer=None, null sink)`——与 exp50 桥同一条已验证循环，
  未来 Mortal 半庄直接同轨。
- `scripts/run_hanchan_arena.py`：2v2 座位轮转，每半庄记四家顺位与 uma（[+15,+5,-5,-15]×1000），
  按 side uma 差给 0/0.5/1 分 → 可比 Elo 拟合（**独立量纲**：hanchan_history.jsonl，
  绝不与单局池混写）。
- 已知观测缺口如实声明：本场数不在模型观测内（引擎已知缺口）；供托经 table.kyotaku 可见。

## 首批评测（判据）

1. 冒烟：任意两 ckpt × 20 半庄，状态机守恒断言（分数总和=100000+终局供托、连庄/轮转正确）；
2. 首轮排位：旗舰 bc49 vs 纯自对弈冠军 vs bcrl14 vs **Mortal（其主场！GRP 顺位头首次 in-distribution）**，
   各对 100+ 半庄。**预注册预期**：Mortal 在半庄刻度的领先应 ≥ 单局刻度的 27 分——若显著放大，
   证实"单局刻度低估顺位技能"，多局训练（引擎手术）立项依据到手；
3. 单局 Elo 序 vs 半庄 Elo 序的秩相关——两刻度分歧点即顺位技能所在。

## Progress
- [2026-08-29] 预注册。
