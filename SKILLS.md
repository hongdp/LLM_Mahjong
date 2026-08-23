# Project Skills & Knowledge Base: GCP RLHF

> **🤖 AI Agent Directive**: Whenever you start a new session or work on this repository, you MUST read this file first. As the project evolves, if you encounter new bugs, hardware limitations, or make architectural decisions, you are required to **continuously update and append** to this skill file so that the project's context is never lost.

## 1. Architectural Philosophy
*   **Three-Phase Deployment Strategy**:
    *   `Phase 0 (Local)`: Prototyping and logic debugging on local limited hardware.
    *   `Phase 1 (GCP Compute Engine)`: High-freedom agile development on an on-demand GPU VM (Start -> Sync -> Train -> Auto-Shutdown).
    *   `Phase 2 (GCP Vertex AI)`: Serverless, containerized Custom Training Jobs for large-scale unattended runs.
*   **Modular Reward System**: All environment reward models MUST be modularized. We use a registry pattern (`src/rewards/registry.py`) and a base class (`BaseRewardModel`). Never hardcode reward logic into the main training loop.

## 2. Hardware Constraints & Workarounds (Local RTX 4080 16GB)
*   **Constraint**: Standard FP16 RLHF training for modern models (e.g., 4B+ parameters like Gemma 4) will instantly cause Out-Of-Memory (OOM) errors on a 16GB VRAM GPU due to the need for multiple model copies (Active, Reference, Reward) and optimizer states.
*   **Solution implemented**:
    *   **QLoRA (4-bit)**: Must load models using `bitsandbytes` in 4-bit `nf4` precision.
    *   **TRL Native PEFT**: Rely on `trl`'s native handling of `peft_config`. By doing this, `trl` will use the base model weights with adapters turned off for the reference model pass, saving massive amounts of VRAM (eliminating the need to load a separate reference model).
    *   **Tiny Batch Sizes**: Keep `batch_size` and `mini_batch_size` down to 1 or 2 for Phase 0 testing.

## 3. Security & Git Hygiene
*   **NEVER** commit GCP service account keys (`*key.json`, `*credentials.json`), local environment variables (`.env`), or massive model checkpoints (`*.safetensors`, `*.pt`) to Git. The `.gitignore` has been specifically tailored to prevent this.
*   **AUTO-COMMIT RULE** (updated 2026-08-01, user-approved): The AI MAY run `git commit` autonomously at logical milestones — a completed feature/fix with relevant tests passing, or an experiment record update. Commit messages must be descriptive (conventional-commit style preferred). Keep commits scoped: don't bundle unrelated changes. **`git push` and any history rewrite (rebase/reset/amend of pushed commits) still require explicit user confirmation.**

## 4. Current Setup Status (May 2026)
- **Framework Replacement**: Abandoned `trl`'s `PPOTrainer`/`GRPOTrainer` due to their inability to handle interactive multi-turn POMDPs. Implemented a custom decoupled architecture (`ReplayBuffer` -> Custom Advantage Trainer).
- **Environment Decoupling**: All Mahjong physics and LangGraph orchestration logic moved to `src/tasks/mahjong`.
- **Secrets Management**: Implemented `python-dotenv` for securely loading `HF_TOKEN` from `.env`.
- **Local Testing Constraint**: The `peft` library upcasts embeddings to fp32. For Gemma models with 256k vocabs, this allocates >10GB VRAM instantly, causing OOM on 16GB GPUs even for 2B models. Use `Qwen2.5-0.5B` for local Phase 0 verifications.

- **RL Environment Stability**: For small LLMs, strict output formatting is achieved by combining Chain-of-Thought (`<think>`) prompts with Regex action parsing. Hallucinations (e.g. discarding unowned tiles) are managed via Action Masking (Forced legal rollout) to prevent infinite loops while preserving the negative gradient.

---
### Engine v2 & Training Lessons (Aug 2026)
- **Ukeire-greedy discard policy is a trap**: ranking discards by raw ukeire type-count REGRESSES hands — looser (higher-shanten) hands accept more tile types, so the metric favours breaking joints. Always rank by (post-discard shanten, then ukeire): `TileEfficiency.evaluate_discards_ranked`. This bug silently shaped both the original SFT teacher and the step reward model.
- **Greedy decoding kills RL**: `model.generate()` defaults to `do_sample=False`; policy-gradient RL then explores nothing and can only reinforce the SFT prior. Rollout generation must sample (we use temperature 0.9, top_p 0.95).
- **LangGraph recursion_limit defaults to 25** — a real full-length mahjong round needs hundreds of node transitions; pass `config={"recursion_limit": 1000}` to `graph.invoke`.
- **Qwen3.5 on 16GB (RTX 4080)**: Qwen3.5-2B trains fine with QLoRA but (a) peft has no default LoRA target-module mapping for `qwen3_5` — pass `target_modules=[q/k/v/o/gate/up/down_proj]` explicitly; (b) the 248k vocab doubles CE logits memory — max batch_size 2 at ~900-token sequences (batch 4 OOMs). Generation is fast (~1.5s/128tok on transformers 5.8).
- **SFT CoT must be faithful**: template "think" phrases get parroted verbatim by small models (observed May 2026). Derive the think text from the SAME computation that picks the action (shanten counts, ukeire comparisons, wait tiles, yaku from the scorer).

### RLHF Experiment Execution
- **Fresh Environments:** Every time a new training run is triggered, a new timestamped experiment directory must be created (e.g. `exp_name_YYYYMMDD_HHMMSS`) to avoid overwriting previous data. The only exception is when explicitly resuming a run (using the `--resume` flag).
- **Experiment Records (Aug 2026):** All experiments MUST follow the shared `ml-experiment-tracking` skill (`~/Workspace/SKILLS/ml-experiment-tracking/SKILL.md`, linked into `~/.claude/skills/`): write `EXPERIMENT.md` (purpose / method / success criteria) BEFORE launching, append progress during the run, record results / conclusion / artifact manifest after, and keep `experiments/INDEX.md` up to date. The May 2026 baseline run left no record of its intent or outcome — that must not happen again.

---
### Status Snapshot (2026-08-02 — supersedes the May/early-Aug snapshot)
- **Naming convention (adopted 2026-08-02)**: experiments get descriptive names (`exp1_shaping_arms`, `exp2_settlement_vs_pbrs`, …); `infra revN` refers ONLY to the training-stack configuration (rev1 baseline / rev2 fast-path+4-game batching / rev3 bf16_lora+12-game batching+update batch 4). Never number experiments by infra revision — that conflation caused real confusion in the 08-02 session.
- **exp1 (shaping arms) COMPLETE**: PBRS+REINFORCE vs +PPO vs PPO+value-bundle on infra rev3, stopped at ep24-26, arena-judged. Verdict: no significant strength change vs SFT anchors in any arm (+1038/+331/−1475, all CIs cross zero) despite large style migration. Full report: `docs/report_exp1_shaping_arms_20260802.md`.
- **May-era blocking issues all resolved**: format collapse (weak adapter) fixed by 3-epoch SFT adapters; action-type whitelist enforced in `table.py` ACTION_RE; reward exploits eliminated by the v2 engine + PBRS rewards (docs/reward_energy_pbrs.md).
- **Algorithm stack**: PBRS potential rewards (telescoping-consistent, unfarmable) + optional dora value term; PPO (no critic, KL early stop) or REINFORCE; optional initial-hand covariate baseline. 47 unit tests.
- **Throughput stack**: batched parallel rollout (near-linear to 24 games), bf16 LoRA on A100 (+55% over nf4), fast-path kernels; per-game cost 525s → 100s (5.2×).
- **Next majors queued** (TASKS.md): v3 threaded-context architecture (design doc ready), critic head vs duplicate-deal decision pending variance decomposition from the current fleet's data.
- **Repo layout**: legacy `checkpoints/`/`logs/` and `src/data_loader.py`/`src/models/` stubs unchanged (archive/delete manually when convenient).

### Ops Lessons (Aug 2026, GCP session)
> The generalizable GCP practices from this project (and the other workspace trainers) are consolidated in the shared **`gcp-trainer`** skill (`~/Workspace/SKILLS/gcp-trainer`, loaded via `~/.claude/skills`) — consult it first when launching cloud runs; new *generalizable* lessons go there, project-specific facts stay here.
- **Stopping a GPU VM forfeits its capacity** — us-west1-b A100 was gone on restart (`zonesAvailable: ''`); had to recreate in europe-west4. Weigh idle cost vs stockout risk before stopping.
- **Instance names are project-global with global DNS** — a TERMINATED VM blocks its name everywhere.
- **`pkill -f` self-match**: the pattern text appearing ANYWHERE in your own command line (even in a later pipeline segment) kills your own shell. Bracket trick `[r]un_training` only helps if the literal string appears nowhere else in the command.
- **nohup python stdout is block-buffered** — export PYTHONUNBUFFERED=1 (baked into run_training.sh) or logs look frozen mid-epoch.
- **Home-uplink rsync tax (~1Mbps)**: sync code via git push → VM `git pull` (repo is public); rsync only for data (with sha256 gate). A background task's `| tail` masks failures — check real exit signals, not pipe tails.
- **Don't sample generated files mid-write** — the value corpus sha changed between generation and shuffle finishing; verify AFTER the producing task completes.

---
### GCP Phase 1 Infra Facts & Lessons (Aug 2026)
- **Project**: `workstation-185016` (billing enabled). Results bucket: `gs://llm-mahjong-experiments` (us-central1). VM: `mahjong-a100` (a2-highgpu-1g, 1×A100 40GB) in **us-central1-b** — zone `-a` was A100-STOCKOUT on 2026-08-01; the error message lists which zones still have capacity.
- **Quotas (re-checked 2026-08-02)**: on-demand A100-40GB=**1 per region** (A2_CPUS=12, exactly one a2-highgpu-1g), preemptible A100=**16 per region**, L4=8, T4=4, A100-80GB=0, H100 metric absent (must request). `PREEMPTIBLE_CPUS=0` everywhere but **that does NOT block Spot** — with no dedicated preemptible CPU quota, Spot draws from the STANDARD pool (the earlier "Spot unusable" note was wrong). TPU v5e quota exists but bitsandbytes has no XLA backend and interactive rollouts thrash XLA recompilation — not worth the port for this workload.
- **Provisioning model: use DWS flex-start, not on-demand** — a2-highgpu-1g costs $3.674/h on-demand vs **$2.020/h flex-start (−45%)** vs $1.928/h Spot; flex-start is only ~$2.4 dearer per 26 h run than Spot but **cannot be preempted for up to 7 days**, and a 50-epoch run measures ~26 h (30 min/epoch at parallel_games=12), a 6× margin under the limit. It also draws preemptible quota, sidestepping the 1-per-region on-demand cap. Full cost/quota analysis, exact SKUs, gcloud flags and G4 (RTX PRO 6000) evaluation: **[docs/gcp_compute_cost_and_quota.md](docs/gcp_compute_cost_and_quota.md)**.
- **GPU selection for this pipeline**: wall-clock is dominated by RL rollout (batch-1 autoregressive decode), which is **memory-bandwidth-bound, not compute- or VRAM-bound**. Rank GPUs by bandwidth: L4 300GB/s < RTX 4080 717GB/s < A100 1.6TB/s. An L4 is *slower than the local 4080* and saves nothing ($43 vs $48 per full run); A100 40GB on-demand (~$3.67/h, ~13h) is the sweet spot.
- **Default VM service-account scope is `devstorage.read_only`** — GCS uploads fail silently late. Create VMs with `--scopes=storage-rw,...` (baked into `scripts/phase1_ce/start_vm.sh`).
- **DLVM `common-cu129-*` images ship NO conda** (despite older docs) — bootstrap a plain python3.10 venv; pinned deps in `scripts/phase1_ce/requirements_pinned.txt` (mirrors the local rlhf_mahjong env; torch cu130 wheels need the driver-580 image family).
- **`trainer.py --resume` only reuses the directory name** — no optimizer/epoch state restore, and no checkpoint exists until SFT fully completes. Until real mid-run resume lands, preemptible/Spot runs would corrupt the pre-registered epoch-count criteria; run on-demand.
- **Run hygiene**: `run_training.sh` traps EXIT → uploads the whole experiment dir + nohup log to GCS, then `shutdown -h now`, so the VM never idles on the meter even on crash.

---
### Rollout Performance Diagnosis (Aug 2026, controlled benchmarks)
- **Qwen3.5 without the fast-path kernels is host-launch-bound, not GPU-bound**: the hybrid linear-attention layers fall back to a torch implementation ("The fast path is not available…" warning at every startup) that fires dozens of tiny kernels per layer per token. Measured: 2B nf4+adapter decode = 18 tok/s on RTX 4080, **11 tok/s on A100** (the faster GPU LOSES because the a2 Xeon's single core is 1.85× slower than the local desktop CPU — GPU util ~18%). Decode speed is flat vs prompt length (25.4 tok/s @ 1 tok vs 24.4 @ 900 tok prompt), i.e. fixed per-token overhead, not O(T) recompute.
- **Quantization is a minor factor here**: bf16 (no nf4) only lifted A100 decode 11.1 → 14.6 tok/s (+32%).
- **Fast path needs BOTH `flash-linear-attention` AND `causal-conv1d`** (`is_fast_path_available = all(...)` in modeling_qwen3_5.py) — installing fla alone does nothing measurable. causal-conv1d must be COMPILED and torch's CUDA version must match system nvcc exactly at the major level (torch cu130 wheels + system CUDA 12.9 toolkit → build rejected; fix: `torch==2.12.1+cu129` on the DLVM cu129 image, then `CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install causal-conv1d --no-build-isolation` with CUDA_HOME=/usr/local/cuda-12.9; needs `wheel ninja packaging` preinstalled).
- **Old "generation is fast (~1.5s/128tok)" note above is superseded** — with sampling + adapter + fallback path the real number is ~7s/128tok locally.
- Headroom beyond kernels: parallel-game batched rollout (GPU util has 5-8× slack), bf16 rollout weights, torch.compile/static cache. Cost math that motivated all this: at 11 tok/s an A100 run = ~50h ≈ $180.

- **DNN rollout 时间构成与 GPU 批量推理决策（2026-08-15，已定，触发条件见末尾）**：DNN 自对弈 rollout 全在 CPU（fork worker、batch-1 推理），GPU 只做更新步。实测 batch-1 CPU 延迟：cnn_m 0.64ms、vit_small 2.31ms（3.6×），但整体吞吐只差 1.6×（25.4 vs 15.8 局/s，G2 30 workers）——反推每局 ~430 次网络调用，时间构成：**cnn 网络 23%/引擎 77%；vit 网络 52%/引擎 48%**。结论：①逐次调用发 GPU 是负收益（kernel 启动+PCIe > 0.64ms）；②中心化批量推理（并行对局状态凑 batch）受 Amdahl 封顶——cnn 上限 1.3× 不值得，vit_small 实际 ~1.6-1.8×，需重构对局循环为协程/状态机（鸣牌/打断窗口是易错区），当前不做。**触发条件（用户已拍板：未来探索更大 transformer 时做）**：batch-1 延迟 ≥10ms 级的架构（网络占比 80-90%，Amdahl 上限 5-10×）或大规模对局立项。排序提醒：引擎优化（向听缓存等）对所有架构生效且吃掉 cnn 的 77%，应排在 GPU rollout 之前。

---
### Rule Fidelity: EMA RCR 2016 Audit (Aug 2026)
- **Benchmark document**: the European Mahjong Association's *Riichi — Rules for Japanese Mahjong* (2016 revision). A Chinese translation lives at `breizhmahjong.fr/wp-content/uploads/2015/11/RCR_2016_trad-chinoise.pdf` (the host 403s bare `curl` — send a browser UA). No poppler on this box; extract text with `gs -sDEVICE=txtwrite -dTextFormat=3`.
- **The EMA rules have NO red fives.** `has_aka_dora=False` and a plain 1-9×4 wall are *conformant*, not a gap — the earlier "aka dora missing" entry in `engine_known_issues.md` was wrong. What RCR 3.12 does require is **ura dora**, which really was missing.
- **The subtlest engine bug found: furiten memory must survive the tile being called.** `_claim_discard` popped the called tile out of the discarder's river, and furiten read that river — so pon-ing a tile silently un-furitened the player who discarded it. Fix: keep a permanent `furiten_river` (append-only) separate from the display `discards`. Any engine that models "the river" as one list will have this bug.
- **Kan must consume a live-wall tile** (RCR 3.7.1). Drawing the replacement from the dead wall without moving the live wall's tail into it silently grows the round by one draw per kan. Combined with no round-wide 4-kan cap (3.7.2 — the old code only capped melds *per player*, so 16 kans were reachable) this was a live reward-hack path: each kan flips another dora indicator.
- **Riichi's 1000-point stick must be escrowed, not paid** — if the declaration tile is ronned the riichi never happened (3.12). Model it as `riichi_pending`, settled by `advance_turn`/`_claim_discard` and voided by the ron path.
- **Same-turn furiten needs a snapshot, not a live query.** Compute "who could have ronned this discard" at discard time (`_ron_chance`) and apply the flags only when the interrupt window closes — querying waits after the window would either see mutated hands or block the very ron being declared.
- **Chankan forces the added kan to be a two-phase action**: `step` opens a ron-only interrupt window and mutates nothing; `resolve_pending_kan()` completes it. Every driver of the engine (orchestrator, batch_rollout, SFT generator) must handle the new `info["chankan"]` branch — a driver that treats "not discarded" as "same player continues" will spin forever on the un-mutated kan.
- **`_waits` memoization is worth 4x**: furiten checks, the missed-ron snapshot and the riichi-ankan test all hit the same hand repeatedly; each miss is 34 shanten evaluations. Random-policy throughput went 1.2 → 5.1 games/s, i.e. the added rule fidelity ended up *cheaper* than the old engine.
- **Round randomization** (`randomize_round=True`, training default) samples round wind (东/南) and dealer seat. It changes only VALUES in the 场况/自风 lines, not the template shape — so an old SFT adapter still parses, but the teacher corpus should be regenerated (`RANDOMIZE_ROUND = True` in `generate_sft_data.py`) or the model only ever sees 东1局/庄家0.
- **Cross-run comparability**: a rule change invalidates comparison with in-flight runs. exp1's three arms (started 2026-08-01) are on the pre-audit engine and must not be compared against post-audit runs.

*(End of SKILLS.md. Append new learnings below this line in the future.)*

## 运维教训（Ops Lessons）
- **`pkill -f <pat>` 自杀陷阱（2026-08-14 第三次踩）**：pattern 出现在自身 `bash -c` 命令行里 → 杀掉自己的 shell，命令以 exit 144 中断，**后续同一调用里的语句不会执行**（这次导致文档编辑丢失）。可靠做法：先 `pgrep -af` 拿到 PID 再 `kill <pid>`，或把 kill 与后续动作拆成两次调用。

- **诊断性探针必须自带正对照**（2026-08-11 exp4）：单看「隐状态解释不了回报（EV 0.02）」无法区分「工具坏了 / 表征不行 / 目标本质不可预测」。加两条对照后结论才封闭——引擎真值特征同样只有 0.021（排除表征），隐状态解码 prompt 内明写的自家点数得 0.46（排除工具）。**留出集要按对局切分**（同局共享终局结算，步级切分会泄漏标签），**超参要在训练集内 CV 选**（在测试集上挑 alpha = 偷看，会系统性高估）。
- **子集覆盖率要核对**：引擎特征探针初版只解析出 21% 的状态（14 张手牌算不了受入被静默丢弃），与 LLM 探针的全量样本不可比。修正（按最佳切牌评估）后覆盖率 100%，结论才成立。任何「过滤后比较」都要先报覆盖率。

- **监控的「VM 不见了」必须二次确认**：`gcloud describe` 偶发瞬时失败会返回空串，单次空值当关机会误报完赛（2026-08-11 exp4 竞技场误报，实际 VM 正常运行）。判定关机需连续 2 次确认，且要把「查询失败」与「资源不存在」区分开。
- **在别的脚本还在 bootstrap 时挂第二个任务，venv 可能尚不存在**：`source ~/venvs/rlhf/bin/activate` 静默失败 → 退化到系统 python → ModuleNotFoundError。并行加挂任务时用**绝对路径的解释器**（`$HOME/venvs/rlhf/bin/python`），不要依赖 activate。

- **（2026-08-10 事故）flex 机 DELETE 终止动作 + 只在结束时上传 = 任何 GCE 发起的终止都全损**：宿主机事件在 36h 死线前 18h 终止了 exp4 训练机，DELETE 动作连盘蒸发 17h 状态。「不抢占」承诺不覆盖宿主故障。修复已固化：①终止动作改 STOP（盘活下来可走退役取证流程）；②run_training.sh 内置 10 分钟增量 GCS rsync（损失上界 = 10 分钟）；③max-run-duration 给 1.8× 余量（48h）。

- **老机退役标准流程（2026-08-09 三台 8-01 on-demand 机执行）**：停机不删 = 每台 200GB 盘 ~$20/月白烧 + 名字占坑（曾致创建静默失败）+ flex 机停机即废（不可重启）。退役流程：卸引导盘 → 同区临时 e2 只读挂载（`mount -o ro,noload`，选**最大**分区——云镜像有 BIOS/EFI 小分区，取「最后一个」会挂错）→ 盘上 experiments+日志 tar 流式上传 `_salvage/` → **回读 manifest 非空才放行删除** → 删临时机/实例/盘。三台共抢救 725MB（$0.015/月）换掉 $60/月。

- **临时机（DELETE-on-termination）把每个脚本错误都放大成整机自毁**：exp2 竞技场两次「启动即死」（① `gsutil rsync` 不自动创建本地目标子目录，与 `cp -r` 不同；② `python scripts/xxx.py` 的 sys.path[0] 是脚本目录，`import src` 必须 `PYTHONPATH=仓库根`——训练用 `python -m` 从未暴露）。对策已固化：EXIT trap 先抢救日志到 GCS orphan_logs 再关机；发射脚本内置 preflight import/CUDA 检查，败在第一场比赛之前；发射后必须亲眼确认第一步实质进展（preflight OK + 首场开打），不能只看进程存在。
- **多步脚本禁止无条件成功标记**：旧竞技场脚本三场全败仍打 ARENA_ALL_DONE。成功标记必须由每步退出码聚合决定（FAILED 标志位），否则监控把失败当完赛。

- **（2026-08-15 a0 事故）重复并发发射会让 bootstrap 自我竞态**：pgrep 误判时代对同一 VM 多次发射 runner，两份 bootstrap 并发——崩掉的那份在兄弟进程 pip 装完 mahjong 之前就 `import`，报 ModuleNotFoundError 后 EXIT trap 自关机；事后验尸 venv 却是完整的（另一份装完了）。这类「事发时缺、验尸时在」的幻影缺包，先怀疑并发发射而不是网络。修复已固化：runner 起跑前**硬校验依赖闭包**（在将要执行 run 的同一解释器里 import 全家，失败打印 `which python`/pip list 后大声退出）；发射永远单次、以 GCS artifact 验证而非进程列表。参见上文 venv bootstrap 竞态条目——同族，成因不同（他挂 vs 自挂）。

- **（2026-08-16）TensorBoard + rsync 镜像 = 静默失读**：TB 的 DirectoryWatcher 假定事件文件
  **只追加**；`gsutil rsync`/`rsync` 更新文件走"临时文件+rename"= 换 inode，TB 撞上后该 run
  静默停止索引（run 列表里在、scalar 全无——正是"cloud_vit_RL 又没了"的病因，概率性复发）。
  修复：两段式镜像——rsync 落 staging 目录，再 `cp -f` 原地覆写（同 inode，对 TB 等于纯追加）。

- **（2026-08-16 exp12 确认）PPO 后期平台 = 熵奖励均衡，不是噪声也不是步长**：恒定 entropy_coef
  （0.03）在策略梯度信号萎缩的末段变成主动约束，把策略钉在人工高熵位（H~1.18）。三臂单变量消融：
  熵 0.03→0.01 臂 +1073±594（发现+独立确认合并，p≈0.0004，新冠军 E-700k）；4× 批量臂 null
  （点估计 −841，反证"优势噪声地板"假说）；lr↓ 臂 null。**训练配方**：熵系数要退火——前期 0.03
  探索，决出率饱和/pg_loss 萎缩后降至 0.01；进一步退火待测。档案：experiments/exp12_plateau_prereg。

- **（2026-08-15 决策）自对弈 `win_rate` 更名 `decisive_rate`（仅 TB 展示层）**：该指标 = 有人和牌的对局比例（1−流局率，四座同网，上限 100%），是**风格**指标不是强度指标，曾两次误导分析（人类顶级对局参照：流局率 13-16%）。强度排名只认复式牌配对竞技场。实现：trainer 写 TB 时经 `TB_TAG` 映射；train_log.json 字段名**不动**（下游报告/探针/resume 兼容）；历史已完成 run 用 `scripts/rebuild_tb_from_log.py` 重建到 `tensorboard_r/`（原目录保留），正在跑的 run 于下次自然重启时切换。

- **（2026-08-16 决策）主线=纯自对弈（AlphaZero 志向），教师先验产物只做测量仪**：exp14
  证明 BC warm-start 在 240k 局甩开从零同行 +3482±1042（项目史上最大效应）——但用户明确：
  把教师知识装进起点违背「从对局中自行探索」的初衷（那是初版 AlphaGo 路线）。规则：
  ①冠军谱系只认零教师知识的 run；②教师系模型（bc_*、exp14 系）只入 Elo 锚点池当基准/
  陪练（只测不教）；③「exp14 参照曲线 − 纯自对弈曲线」的差距收敛到 0 = 自行探索的成功
  判据；④哲学一致的加速手段是往自对弈里加搜索（AlphaZero 的真配方是 self-play+MCTS
  放大），而不是加教师。档案：experiments/dnn_exp14_bcvit_rl_20260816。

- **（2026-08-17 配方结论）优势 winsorize 保留 ±5σ；vit 谱系天花板低于 cnn**：exp16 单变量证明
  放宽到 ±10σ 显著变弱（−1064±1015）且满贯+频率不变——大番 reward-cap 担忧结案，±5 是有益
  方差控制。exp15 证明 vit 在冠军配方下 ~350k 局到顶 ~950 Elo（此后 850k 局平台），败于
  cnn 冠军 e700（−1475±1103）：**结构先验改变爬升速度、不改渐近线**。主线下一杠杆：
  exp17 GAE 归因（A0=GAE+cnn 从零 600k 得 1022 分 + 自发立直 14.9%，两项都超 e700 谱系）。

- **（2026-08-19 配方结论）1012 平台被两条正交路线分别打破**：exp17-C 证明 GAE(λ=0.95) 是
  信用分配侧的真杠杆——同协议单变量 +66 Elo（1079.7，纯自对弈新纪录）、立直率 0.117→0.243、
  700k 终点仍在爬升；exp19 证明 ConvFormer（花色内卷积 stem + rank 相对偏置 + 容量对标 +
  warmup 优化包）让注意力从 928 抬到 1065.7（慢热高顶型）。**当前冠军配方候选 =
  GAE + 熵台阶（+ ConvFormer 待 exp20 合体验证）**；与教师参照线（1117.7）缺口收窄到 38 分。
  运维教训：warmup 步数必须按「占总更新数比例」（~5-10%）设定，绝对步数在小更新量训练里会
  吃掉半程（exp19-r1 因此报废重发）。

- **（2026-08-19 北极星，用户定调）项目最终目标 = (a)**：用**纯自我对弈、零人类知识注入**的学习
  系统，在麻将上自行发现完整技能栈（防守的自发涌现为里程碑判据），**爬到人类高手水平——
  到达本身即目的（科学验证）**。(b) 反向向模型学习麻将新知、(c) 方法论迁移到其他领域，
  为未来方向，现阶段不投入。每个实验设计须回答「它把我们推向 (a) 了吗」。
  推进方式：Claude 自主迭代（预注册→发射→判定→下一步），在里程碑汇报；仅在纯度相关分岔
  或成本跃迁时请用户裁决。前置基建：Elo 池须加入人类校准外部锚点（仅用于测量，不进训练），
  否则「人类高手水平」不可判定。

- **（2026-08-21 运维教训）复用云机前必须等 TERMINATED；死 run 是静默的，要有心跳告警**：exp20 首发射
  落在上一个 run 的 EXIT-trap 关机窗口里，2 分钟后被一并关掉，首 iter 哨兵给出假阳性，两天后才被
  发现重发（VM 关着未计费，但时间全损）。规则：`instances start` 对 RUNNING 机是 no-op 成功，
  不能当「机器可用」信号；发射后除首 iter 确认外，必须挂 train_log 新鲜度心跳（>30 min 未更新即报）。
- **（2026-08-21 用户规则）每个长跑任务必须有心跳监控**：统一用 `scripts/phase2_dnn/watch_run.sh <run>`
  （train_log 新鲜度 >40 min 报 STALE、games_final 报 DONE、30 min 无日志报 MISSING）；本地长评估
  用等价的 mtime/进程心跳。会话重启后第一件事是重建全部心跳。
- **（2026-08-21 决策）人类标尺 = Mortal**（用户批准）：仅作冻结外部锚点入 Elo 池用于测量，训练永远
  不接触。M3 前须先核实 Mortal 权重的可获得性（其官方不公开分发训练权重）；若不可得，备选
  Akochan（全开源）或人类牌谱 BC 标尺。
- **（2026-08-22 性能）rollout 引擎优化 3.17×（4.65→14.76 局/s 单线程，语义零变化）**：元凶是每次打牌后
  对三家各做 34 次 shanten 的 `_waits`（占 64%）；修法=13 张先判听牌再扫描 + 候选剪枝（同花色 ±2/
  同牌，国士形全扫）+ 进程级 shanten LRU + 立直候选 14 张门控 + 荣和用 memoized waits 预门控 +
  DNN 路径关掉 LLM 文本观测 + numpy 编码器 + tile_to_34 查表。护栏：固定种子贪心轨迹 sha256
  （`scratchpad/snapshot_traj.py` 思路）必须逐字节一致 + 全套测试。引擎份额 77%→~45%，
  下一步 GPU 批推理（scale-up 前置）。档案 experiments/perf_rollout_20260822。
- **（2026-08-22 结论）三条渐近线 + 梯度竞争假说证伪**：纯自对弈现配方族的平台——cnn 无 GAE ≈1012、
  ConvFormer（含 ×GAE 合体）≈1060、vit ≈930；**冠军仍 cnn+GAE 1079.7**，GAE 与 ConvFormer 不可加。
  exp20 在 1.2M 饱和后 defense_iq 仍 ≈0 ⇒「攻击饱和后防守自发浮现」不成立；剩余解释=生态均衡
  （种群推牌近似最优），对手联赛（纯）为主手术，输入 v3（exp23）裁决输入层假说。风格≠强度再证。
- **（2026-08-22 性能）GPU 批推理服务使 20M 模型 rollout 9.5×**（`src/agents/dnn/infer_server.py`，
  `cfg["gpu_infer"]`；跨进程 RPC：共享内存槽位 + 单信号量 + 广播 Condition + CUDA graph 分桶）。
  小模型（cnn_m）仍走 CPU。坑：①spawn 服务重导入 `__main__`——任何启用入口必须 `if __name__`
  守卫；②`torch.nonzero` 对并发写的共享张量会内部断言——用 numpy 快照；③`pkill -f` 的模式若出现在
  当前 shell 命令行会自杀——清理放脚本文件；④cudnn.benchmark 在批尺寸漂移下反复自调（默认关）；
  ⑤CUDA graph 每桶占显存，本机（llama-server 占 14GB）只能 ≤64 桶；⑥批推理后瓶颈转到 CPU worker
  周转，scale-up 时该买 vCPU 不是更贵的 GPU。档案 experiments/perf_gpu_rollout_20260822。
- **（2026-08-22 用户规则）后续所有训练 run 统一用 GPU 批推理 rollout**：trainer 加 `--gpu_infer`
  （`--infer_max_batch 128 --infer_wait_ms 4`），云端 L4 设 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。
  小模型（cnn_m）本机 GPU 路径略慢于 CPU（32 vs 40 局/s），仍按规则走 GPU，首个云端 run 实测校准；
  等价性为统计级（采样在 GPU RNG），贪心轨迹 hash 不再作为跨路径护栏。
- **（2026-08-22 结论）防守嫌疑全部排除后只剩生态均衡**：exp23 证明补齐完整公开记录（v3 编码）既不提升
  强度（1062.6 vs 1079.7）也不催生防守（defense_iq 0.008）；至此感知/信用/采样/输入/攻击饱和五项
  全部排除，唯一剩余解释=当前种群里推牌近似最优。**exp22 纯谱系对手联赛为唯一主手术**；
  联赛对手池只能含零教师知识模型（bcrl14/bc_* 不得进训练对手池，只留在评分池）。
- **（2026-08-22 用户裁定）rollout 基建统一，不按模型规模自动切换**：曾提议「<5M 参数走 CPU 推理、大模型
  走 GPU 服务」以省小模型 ~10% 开销，用户否决——统一用 `--gpu_infer`（学习者）+ `--gpu_infer_opponents`
  （联赛池）让各 run 的吞吐/对局构成可比。实测归因：exp22 比 exp17-C 快 1.55×（29→45 局/s）几乎全部
  来自引擎优化；GPU 服务对 2M 小模型零贡献、对 ≥10M 模型 ~10×。
- **（2026-08-23 基建）雀魂实战桥接 = MJAI 协议 bot + 引擎影子桌**（`src/agents/dnn/mjai_bridge.py`，手册
  `tools/majsoul_bridge/README.md`）：仿 MahjongCopilot，它负责抓包/翻译/点击，我们只实现 `react(mjai_event)`。
  关键决策：影子桌**继承真实引擎类**、由事件驱动而非另写观测翻译——编码器与合法动作生成零改动，训练/实战
  观测分布天然一致；保真由「引擎自对弈→MJAI 流→影子桌」逐决策张量对比守护（400 局 7004 决策逐位一致）。
  坑：MJAI 杠后单发岭上 `tsumo`（引擎隐式摸）→ 按 tsumo 事件计活牌；抢杠要在 kakan 写入前判定；MC 不转发
  局结果（安装器补丁转发 `end_kyoku`/`end_game` 才能计顺位）；引擎无红五 → 折叠观测、出牌优先留红五。
  评估须走 ml-experiment-tracking（预注册 `experiments/exp24_majsoul_live_prereg`），发射需用户账号与浏览器。
- **（2026-08-23 exp25）出牌温度是被忽略的强度杠杆**：同一冠军 checkpoint 贪心(T=0) vs T=1 采样，复式牌 1000 副
  **+484±418（贪心胜 56%）**，与一次代际提升同量级。历史 Elo 池/竞技场全是 T=1 口径，贪心口径绝对值整体上移
  （相对排序未验证）。实战（雀魂）与对外对比用 T=0；`run_arena_dnn.py --dnn_temperature_a/_b` 支持每边独立温度。
