# Project Skills & Knowledge Base

> **🤖 AI Agent Directive**: Whenever you start a new session or work on this repository, you MUST read this file first. As the project evolves, if you encounter new bugs, hardware limitations, or make architectural decisions, you are required to **continuously update and append** to this skill file so that the project's context is never lost.
>
> **当前状态快照不在本文件**：看 [README「当前状态」](README.md)（冠军/纪元/进行中实验）和
> [docs/roadmap_epoch3.md](docs/roadmap_epoch3.md)（队列与路线）。本文件是**追加式教训账本**（时间序）。
> 早期条目（Phase 1 LLM 时代：A100/QLoRA/文本 rollout）保留为历史，Phase 2 已转 DNN + G4 flex，机型/费用见
> docs/gcp_compute_cost_and_quota.md §8。

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
- **exp1 (shaping arms) COMPLETE**: PBRS+REINFORCE vs +PPO vs PPO+value-bundle on infra rev3, stopped at ep24-26, arena-judged. Verdict: no significant strength change vs SFT anchors in any arm (+1038/+331/−1475, all CIs cross zero) despite large style migration. Full report: `experiments/reports/report_exp1_shaping_arms_20260802.md`.
- **May-era blocking issues all resolved**: format collapse (weak adapter) fixed by 3-epoch SFT adapters; action-type whitelist enforced in `table.py` ACTION_RE; reward exploits eliminated by the v2 engine + PBRS rewards (experiments/designs/reward_energy_pbrs.md).
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
- **（2026-08-23 人类标尺）雀魂内置 AI 牌谱评分（"maka"）可作免费的人类校准读数**：C~C+≈初级–铜、B~A≈银–金、
  A+~S-≈金–玉–王座、S~S+≈魂天（来源 NGA tid=44460732 等）。纯自对弈冠军 exp17-C 贪心实战首读 **C+（仅 1 半庄，n=1）**
  ——方向性地把 Elo 池 1079.7 放在人类入门–铜级。maka **每日调用次数有限**，要按半庄攒样本、挑代表性对局评。这是北极星 (a) 第一次对人类刻度定位；以后每个候选冠军实战后都记
  maka 档位（局数 ≥30 半庄才写结论），与 Elo 池、defense_iq 并列成为三把尺。
- **（2026-08-23 测量 bug）竞技场配对分差曾用 A−50000**：引擎单局制下流局的立直棒不返还，四家点数和 <100000，
  整笔漏点被记成 A 的损失（≈ −170~−370/副，且量纲减半）。同一模型自打 1000 副给出 −167±40 才暴露——
  **任何 A/A 对照应恒为 0，是最便宜的指标体检，以后新指标先跑 A/A**。已改为 A−B（0f83e84）；旧数字
  ≈ (新−leak)/2，正结论偏保守、小幅负结论可能是假的；Elo 池纪元 2 重校一并修正。
- **（2026-08-23 exp26）杠不是当前瓶颈**：人类杠常识（听牌暗杠必杠、禁大明杠）套在冠军上无显著收益
  （−4±316）；模型对杠的取舍已是自对弈生态下的局部最优。用户实战印象"不爱杠"来自生态（杠的期望收益被压扁），
  不是漏学。规则覆盖器 `src/agents/dnn/overrides.py` 是只测不训的诊断工具，可复用于其他"有没有学到 X"问题。

- **规则近似会悄悄削减动作空间（2026-08-23 暗杠案例）**：`_can_ankan` 对 RCR 3.12(2)"四张只能读作刻子"用了"手里有相邻牌就拒绝"的保守近似，看板回放时发现立直后摸第四张 5s（234s+555s+单骑）没有暗杠选项。改为枚举所有听牌拆分精确判定（`_tile_only_as_triplet`）。600 局冠军自对弈实测：立直中摸第四张 13 次，旧规则放行 7、新规则 13 → 每 100 局多 1 个合法动作（0.014% 决策），训练分布影响可忽略，但按纪元规则仍视为引擎变更。**流程教训**：引擎改动必须 (1) `anchors.json` 记录引擎指纹（`run_elo_league.py stamp`），`rate` 遇到不同指纹拒绝运行（`--allow_engine_mismatch` 可覆盖并打标）；(2) 正在跑的实验（含本地收尾链）必须在旧引擎下收尾——改动放分支，等实验关账后再合并并重校锚点池。
- **（2026-08-23 费用）DNN 自对弈的每美元吞吐由 vCPU 决定，不是 GPU**：g2-standard-32（$2.6/h，32 vCPU）≈ 39 局/s
  ≈ 15 局/s/$；a2-highgpu-1g A100 flex（$2.0/h）只有 12 vCPU ≈ 15 局/s ≈ 7.5 局/s/$——Phase-1 的"A100 flex 最划算"
  只对 LLM 解码成立。省钱顺序：本机 24 核 + 4080（78 局/s，免费）> spot g2（~$0.9/h，需自动复活 + `--resume`）>
  换 vCPU/GPU 比更高的机型。VM 跑完直接删除（盘费 $10/台/月）。
- **（2026-08-23 实测）云端默认机型改为 g4-standard-48 flex-start（$2.25/h，48 vCPU + RTX PRO 6000）**：
  同一 K=32 向量化 rollout 下 cnn_m_r **279 局/s**（L4 g2-32：104）、handset_xl **159.5**（L4：36）——每美元 3–5× 于
  g2 按需，且 flex 7 天内不抢占。要点：**只用 flex**（按需 $4.5/h 性价比崩）；盘必须 hyperdisk-balanced（pd-balanced 被拒）；
  `--provisioning-model=FLEX_START --instance-termination-action=DELETE --max-run-duration=12h`；us-central1-b 秒拿到容量；
  torch cu129 wheel 原生支持 Blackwell（sm_120）。L4 上 handset_xl 是 GPU 封顶（单行前向 206 µs vs 4080 95 µs）。
- **（2026-08-23 再犯）`pkill -f` 模式出现在本 shell 命令行（含 heredoc 正文）就会自杀（exit 144）**：把 kill 脚本先单独写进文件，
  再用一条不含模式字符串的命令执行。
- **（2026-08-23 用户规则）任务 launch 之后必须确认其性能符合预期，否则立刻查问题**：exp27-B（handset_xl）发射后
  GPU 100%、第一轮 3.8 min、第二轮 14 min 不出——不是"大模型就慢"，而是 rank 偏置 `bias[:, bucket]` 的高级索引反向在
  GPU 上对 20 个槽位做 1.1 亿次原子累加（30 s/minibatch）。发射清单：①前 3 轮的 局/s 与基准对照（训练口径 ≈ rollout
  基准的 40–50%）；②GPU 利用率/显存与预期量级对照；③对不上就 py-spy dump 看卡在哪一行。
- **（2026-08-23）云机取代码改为 `git clone` 公开仓库 + 固定 SHA**（`scripts/phase2_dnn/launch_g4_git.sh`）：比 scp 整目录快、
  可追溯；脚本硬门：SHA 必须已在 origin/master，否则拒绝发射（实验改动先 push 再发）。
- **（2026-08-23 坑）在云机上 `pkill run_dnn_cloud.sh` 会触发 runner 的 EXIT trap → `shutdown -h` → flex VM 直接 TERMINATED**。
  要重启某个 run：只 kill 训练器进程并先摘掉 trap，或干脆删机重建（更干净）。
- **（2026-08-23 用户规则）探索更大模型或不同架构时，不得直接下"scale / 该架构没用"的结论**：先用多个指标确认训练效率正常——熵平台高度/出现时机
  （熵奖励是否压过了策略梯度）、decisive_rate/胜率的上升斜率、approx_kl、EV、梯度范数——再仔细调 hparams（熵系数、LR、
  batch、warmup），只有在调过之后仍不及基线才算架构/规模结论（exp18 vit 输 CNN 85 Elo、exp19/20 ConvFormer 的结论同样是在未调 hparams 的冠军配方下得出的，属"该配方下"的结论）。exp27-C（6.6M CNN）在冠军配方下熵平台 1.2 且回升、
  decisive 只有 A 的一半，正是"配方卡住"而非"架构不行"的典型；exp15 vit-scale 的平台结论需重新审视。
- **（2026-08-23 exp27/28 结论）**：等配方下实例级手牌集合注意力（20M）不优于 34 轴 CNN（拆分探针 65.6 vs 65.9%，Elo 略低）；
  降熵计划抬 T=1、收窄贪心−采样差距，但贪心上限 −26（探索损失）；混合温度无增益。**冠军 = 纪元 3 `cnn_m_r`（A）**：
  从零 1.0M 局 = 旧谱系 2.1M 强度（T=0 1121.8 史上最高），且识赤宝牌——实战部署一律用 A 系。冠军配方（熵 0.03→600k:0.01）保持。
- **（2026-08-23 教训）链式后台脚本的每一步都必须"失败即中止"**：纪元 4 首跑里 `git merge --ff-only` 失败（master 领先），
  无守卫的链继续把校准跑在旧引擎上、还会盖错指纹——20 分钟白跑。规范：合并/测试/改名每步 `|| exit`，
  且发射前打印 `git rev-parse HEAD` + 引擎指纹入日志。另外 `pkill -f` 自杀问题今天又踩了两次：**kill 脚本必须先落盘、
  再用一条不含模式串的命令执行**（heredoc 正文也算在命令行里）。
- **（2026-08-23 exp31 机制结论）副露流是早期局部最优，立直流靠"熵高原"孵化**：切换风格需要整局一致放过鸣牌窗口
  （相关偏移，单步随机采不出）。冠军时间表 0.03×600k 的作用 = 让鸣牌决策保持近掷硬币，使**成整局的门清轨迹**
  留在数据里直到价值网学会它们的优势；H 在 250k 掉到 0.4 的臂（目标熵 1/2）门清轨迹提前消失 → 锁死副露流
  90%+、Elo −55。熵奖励的本质是给替代风格续命，不是逐步探索——熵计划要"高原够长+晚降"，不是"平滑低目标"。
  这是"过早锐化卡死 local minimum"的第一个实锤，卡的层级在防守之前（门清/立直）。

## 2026-08-24 教训：验证段→正式段必须换 run 名，guard 删除必须门控在收尾成功上
`--resume` 续训到正式长度时，**绝不能沿用验证段的 run 名**——旧的 `games_final.pt` 还在同一 GCS 路径，
`watch_run.sh` 只判文件存在不判局数，resume 后几分钟内就误判 DONE，触发收尾链和删机。
已实测差点删掉正在训练的 VM（[[mahjong-current-champion]] 相关线，exp36）。
**修复两处**：① 正式段永远换新 run 名（如 `_full` 后缀），不复用；② `run_guard.sh` 的 VM 删除循环
改为门控在 `close_run.sh` 确认打印 `CLOSE_DONE` 之后，不再无条件执行——即使 close 步骤被杀/崩溃，
guard 也不会继续删 VM。

## 2026-08-24 测量学：镜像自对弈里 agari_rate 是 draw_rate 的恒等式，win_turn 右删失
两条会导致误读的性质，读任何风格画像前必须记住：
1. **`agari_rate ≡ (1 − draw_rate) / 4`**（四座共享同一策略，每局至多一个赢家）。实测六个模型
   全部吻合到 ±0.002（多家和的双响是那点微小超出）。**"和牌率低"和"流局率高"是同一个事实的
   两种说法，不是两条独立证据**——报告时不要当成两个指标并列。
2. **`win_turn` 在牌山耗尽处（~17–18 巡）被右删失**：一手"本可以在第 20 巡和"的牌不会表现为
   高 win_turn，它会变成一次流局。所以**策略变慢不会推高 win_turn，只会推高 draw_rate**；
   win_turn 会在 13–15 附近饱和，看起来"还行"。它同时还是条件统计量（只统计和了的牌），
   带幸存者偏差：只和最快最容易的牌、其余全流局的策略，条件均值反而好看。
**未删失的真话指标 = `tenpai_rate` / `tenpai_turn`**（2026-08-24 新加，正是为此）。
实例：exp30 win_turn 15.3 与 exp31-6 的 14.1 只差 1.2 巡，但 tenpai_rate 39.1% vs 59.6%
差了 20 个点——后者才暴露"根本听不上牌"这个真实病征。

## 2026-08-25 教训：重发实验必须同时停掉旧守护，否则留下误报告警
守护进程（`run_guard.sh` + `watch_run.sh`）和 VM 是**分开管理**的：删掉 VM、换新 run 名重发之后，
旧守护仍在盯着那个再也不会更新的旧 run 名，45 分钟后会发一条 **STALE 心跳告警**。
它本身无害（旧 run 永远不产生 `games_final.pt`，触发不了收尾链；删一个不存在的 VM 是空操作），
但**误报会稀释真告警的可信度**——而心跳告警正是我们抓到 exp41 arm B 被 OOM 杀死的唯一途径。
**规则**：重发/放弃一个 run 时，同一步里 `pkill` 掉它的守护；收尾后用
`pgrep -af run_guard.sh` 核对存活守护与在跑 VM 一一对应。
（注意 pgrep 结果过滤时 run 名含大写字母，`[a-z0-9_]` 类的正则会漏掉。）

## 2026-08-25 推理服务器：不要等待攒批，让 GPU 周转自己决定批大小（白拿 ~10%）
`--infer_wait_ms` 默认从 4.0 改为 **0**。原理：服务器是「取走当前 pending → 前向 → 再取走」，
上一批在 GPU 上算的几毫秒里请求自然堆积，**批大小自平衡在 到达率 × 前向耗时**，不需要人为等待常数。
实测（本地 4080，18 worker）：

| 路径 | wait=4 | wait=0 |
|---|---|---|
| mortal_full（慢模型） | 3599 决策/s | **3974（+10%）** |
| cnn_m_r（现役冠军） | 143.0 games/s | **157.2（+10%）** |
| 低并发 3 worker | 325 决策/s | 320（持平） |

**为什么等待是净亏**：大批次的前向单行效率确实更高（8.6 vs 7.0 行/ms），但为凑批而等的死时间
把这点好处吃光还倒贴。**不需要最小批/最大等待保护**：低并发下本来就没有更多请求可等，
批大小两种设置都是 2.2。
**相关背景**：CUDA graph 消掉 kernel 启动开销后，前向变成纯算力受限、耗时随批大小成正比
（实测翻倍 worker → 批大 2.2 倍、前向慢 2.6 倍、吞吐反降），所以「攒大批」这条常见直觉在这里不成立。

## 2026-08-25 云上 worker 数最优 = 92（不是核数 46），但收益只有 9%
在生产硬件（G4：48 vCPU + RTX PRO 6000 Blackwell 96GB）上实测 `mortal_full_xl_m46` rollout：

| workers | games/s | 决策/s | GPU% | load |
|---|---|---|---|---|
| 46（核数） | 125.8 | 11,176 | 37% | 7.2 |
| **92（2× 核数）** | **136.1** | **12,170** | 37% | 8.4 |
| 138 | 126.0 | 11,287 | 32% | 8.0 |
| 184 | 117.5 | 10,449 | 29% | 13.3 |

**结论：新 run 用 `--workers 92`**（超订阅到核数的 2 倍；worker 大部分时间阻塞在等 GPU，不占核）。
**但更重要的是那两列利用率**：所有档位下 GPU 只有 29–37%、CPU load 只有 7–13/48，
**两边都远未饱和，且加 worker 推不上去**。所以瓶颈不是任一侧的算力，而是**往返延迟结构**
（worker 发出请求即阻塞 → 服务器串行处理 → 唤醒），加 worker 超过 ~2× 核数后
信号量/共享内存争用反而吃掉收益。

**这否定了两个直觉**：① 「worker 83% 阻塞，超订阅应有大收益」——实际只有 9%；
② 「GPU 37% 说明 CPU 是瓶颈」——加 CPU 侧并发也推不动。
真正的下一步应该是**减少往返次数或让 worker 不阻塞**（流水线/双缓冲），而不是继续调并发参数。

## 2026-08-25 事故：/dev/shm 泄漏杀死 exp41 两臂（潜伏数月，被宽观测放大 45 倍才致命）
两臂在**完全相同的 337,920 局**（165 迭代）死于
`unable to allocate shared memory (shm): No space left on device`。

**根因**：`infer_server.py` 开头的 `mp.set_sharing_strategy("file_system")`。该策略把共享张量落成
/dev/shm 文件且**设计上不随引用消失而 unlink**（它的用途正是"进程死了数据还在"）。而
`collect_parallel` **每个迭代新建一个 InferenceServer**，于是每迭代泄漏一整份 planes 缓冲区。

**算术精确吻合**：g4-standard-48 内存 180 GB → /dev/shm 默认 90 GB；46 worker 配置
= 4416 槽 × 934 平面 × 34 × 4B = **561 MB/迭代**；90 GB ÷ 561 MB = **164 迭代 = 336,479 局**，
实际死于 165 迭代 = 337,920 局。

**为什么潜伏这么久**：cnn 的 21 平面只泄漏 12.6 MB/迭代，1M 局（488 迭代）累计 6 GB，
远低于 90 GB。Mortal 的 934 平面把泄漏放大 **45 倍**才第一次撞顶。

**修复**：改用 `file_descriptor` 策略（可用 `TORCH_SHARE_STRATEGY` 覆盖）。实测泄漏归零
（6 迭代 0–1 MB），24 worker fork+conda 压力测试未复现当初改成 file_system 想修的
`SocketClient FileNotFoundError`；`ulimit -n` = 1048576，不会撞 FD 上限。
**显式 `del` 共享张量无效**——试过，仍然 94 MB/迭代，因为策略本身就不 unlink。

**教训**：① 资源泄漏的可见性与负载规模成正比，小配置下的"没问题"不是证据；
② 每迭代重建重量级对象（推理服务器 + worker 池 + 6 个 CUDA graph）除了泄漏，
还有 4.2 s/迭代的固定开销——**复用一个常驻服务器能同时解决两者**，但需要权重热更新路径
（漏更新 = 用旧策略采样却按新策略算 ratio，静默致命），未做。

## 2026-08-28 exp46 基建教训：torch.where 不隔离未选分支的梯度

KL(π‖π_BC) 写成 `where(finite, p*(logp−alogp), 0)` 时前向有限但反向 NaN——masked 槽的
(−inf)−(−inf)=nan 仍在计算图里毒化 backward，任何 lr 都会在第一步把权重打成 nan
（症状：pg=nan 且降 lr/warmup 无效，这是与"步长过大"鉴别的关键指纹）。修复=先
`where` 净化差值再与 `exp(logp)` 相乘（exp(−inf)=0 的梯度是干净的）。熵项的 `safe`
防护是同一原理的既有先例——凡与 masked log-prob 做代数，必须逐因子净化。

## 2026-08-29 exp46 发射夜的两类反复咬人的坑

- **CUDA 分配器尺寸类碎片棘轮**：症状=主进程显存每迭代 +（恰好一个 rollout 张量的大小），
  台阶式增长至 OOM；隔离复现（固定网络+随机批量）反而无罪——因为棘轮由"每迭代长度微变的
  大 cat 张量"触发。**指纹鉴别**：allocated 平稳而 reserved 阶梯涨=分配器问题非 Python 泄漏。
  解法一行：`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（实测 1GB/迭代 → 10 分钟 delta 0）。
  注意力架构比 CNN 更易触发（激活尺寸类更多样）。
- **pkill/pgrep -f 自匹配（本夜第 4 次）**：模式哪怕加了 [x] 括号，同一命令行里其它参数
  （sed 的源路径、grep 的目标串）含有该字符串照样自杀，退出码 144。铁律：**kill 类命令必须
  独立成单条调用**，且调用里不得出现目标字符串的任何明文形态。


## 心跳监控的报警时延必须匹配故障时标（2026-08-30）
今天两次发射期静默死亡（SSH 段静默失败、日志永不出现）都是用户先于监控发现的——旧心跳只有"60 分钟无变化"一档报警，为训练中途卡死设计，对发射失败（~10 分钟内 GCS 必有日志对象，否则即死）完全失聪。规则：**心跳必须分相**——发射相带 15 分钟死线（无日志对象即刻 ALERT），训练相 30 分钟无变化报 STALL；终止标记（DRIVER DONE/failed/Traceback）即时报。另：触发标记必须取被监控日志实际会出现的行（此前"code at <sha>"指纹只存在于本地发射日志，导致心跳整场哑火）。

## 实验结果总结的存放地（2026-08-30 起）
结果类总结一律进 [experiments/FINDINGS.md](experiments/FINDINGS.md)（结果台账）与各 `experiments/*_prereg/EXPERIMENT.md`；本文件只保留教训 / 方法论 / 运维与硬件约束。

## RunPod 成为算力选项，账户已充值（2026-08-30）
web_canvas 项目充值并实测了 RunPod 后不再使用，余额留给本项目。**判决（补查 vCPU 后）：
按保证值打不过 g4-48**——RunPod 每 GPU 只配 9（A40）～16（PRO 6000/A100）个保证核，
而本负载瓶颈是 rollout CPU；同卡每美元 44 vs g4 的 124 局/s，A40 community 每美元 +20%
但单 run 5.4× 慢。**例外是 RTX 3090 community：16 核/$0.22，
叠 3 张 = 48 核 $0.66/h——g4 同核数 29% 价，估算局/s 每 $ 3.4×**（前提：每核速率、
community 长跑可靠性、标称核数落实，~$0.25 探针即可全部验证）。burst 是普适变数
（实测 nproc=128 但非合同）。3090 探针不达标则余额只适合短平快活（评测/探针），
按秒计费占优。运维三条铁律：**用完删 pod 别停**（container disk 停机费率翻倍，
400GB≈$80/月）；SSH 走**直连**入口（`sshPublicKey` 注入的 key 只有直连认，proxy 入口
永远 Permission denied）；pod 是容器，自停靠 `kill 1` 不靠 shutdown。价格表、
create-pod 参数、全部坑见 [docs/runpod_cost_and_ops.md](docs/runpod_cost_and_ops.md)。

## 2026-10-25 行为探针结论要带训练规模（exp120 教训）
- exp21（≤2M 局）"防守只能从示范来"、exp68"从零 RL 从未学会防守"被当成纯血线的定性事实沿用了两个月；
  exp120 用同一仪器（probe_defense）对 ≈100M/384M 局的纯血模型测出 defense_iq 0.10–0.12（1M 局模型 0.00）——结论随规模翻转。
- 规矩：① 任何"模型会/不会 X"的行为结论写进 FINDINGS 时必须带训练局数；② 新里程碑 ckpt 例行复跑历史行为探针，不沿用旧判决；
  ③ 行为代理（现物率）与 oracle 结算（反事实荣和）可能不一致——exp120 中 π 的非现物选择在真实牌桌上几乎不被荣和，代理会低估防守。
- 探针工程：模型代码在别的 worktree 时用 `LLM_MAHJONG_ROOT` 环境变量指路（`scripts/probe_value_1ply.py`），别把 arch 复制过来。
- **后见之明 max 不是收益上限（exp121 Stage 0）**：对同牌山反事实取"最优替代 − 所选"，等价替代（不可能更好）也给出 2.0 千点的"增益"——
  任何"max over alternatives"门槛必须配等价替代对照（混沌地板）和配对系统差；单牌山局末值做逐决策金标准需 K≈(σ/Δ)² 次确定化，先算 K 再承诺。
- **提议"没试过的杠杆"前先 grep 三处台账（2026-10-29）**：同一会话两次把已判负方向当新方向——因子化短视界价值（已被 exp99/114/115/116/119 边界化，
  在 continue-task 分支）、对手池/league/非传递性诊断/检查点集成（已被 exp82/85/87/89/92 与 09-26"博弈论防线总结"判决：非传递性未观察到、M 不可剥削、集成 +0.3 pp）。
  用户已按错误前提批准第二项，发现后中止（$0）。台账分三个 checkout，关键判决常在综述条目而不在单 run 行里。
  规矩：提议前 `grep -i` 主 repo + runpod worktree + continue-task 分支的 FINDINGS/INDEX 并读综述条目，命中的 exp 编号与判决写进提议；grep 为空才可说"未试过"。
  **11-01 修订**：grep 命中不等于方向关闭——先读命中判决的范围戳（架构 / 信号源 / 规模），戳与提议不匹配（如 ≤2M 局的架构判决 vs 100M 目标）时，"在目标范围复测"本身就是合法提议；exp124 证明被这条规矩压住的架构轴正是突破口。
- **bf16 rollout 的 old_logprobs 对小概率动作有 ~0.1–0.2 的 log 误差（exp122，2026-10-31）**：PPO 第一个 minibatch 的 ratio 本应 ≡1，实测 π_old<0.01 的样本
  mean|r−1| 0.14、16% 已越过 ε=0.2——裁剪在更新前就随机丢弃这些样本的梯度（对称、占全部样本 ~1%）。exp113 以此配方训完无害证据，但任何
  "裁剪触发率 / ratio 统计"类读数必须先看 mb0 作数值地板，否则会把噪声读成策略漂移（本探针若无 mb0 对照会判 H1）。修法（若再训）：打包阶段按更新端精度重算 old_logp。
- **训练器探针不要用 `dev` 之类的名字做局部变量**：main() 里 `dev` 是 torch.device，探针块 `dev = (ratio-1).abs()` 遮蔽后第 2 迭代 `torch.arange(device=dev)` 崩；
  只跑 1 迭代的冒烟测不出跨迭代的遮蔽——冒烟至少 2 迭代。
- **注意力类 zoo 架构从未接过 Rust rollout（bench_throughput，2026-10-31）**：`TilesTransformer.act` 缺 `check` 参数，`--engine rust` 直接 TypeError；
  `--fast_trunk/--compile` 原本只认 CNN。ConvFormer 快速主干（批量花色卷积 + SDPA 加性偏置）数学等价、1.5×。
  吞吐剖析先分段（rollout 内 Rust/GPU/传输、打包、更新），cnn_l 三段串行、CPU 与 GPU 互不重叠是下一个 ~2× 的来源。
- **torch.compile + bf16 autocast 可以静默算错（torch 2.12/cu130，2026-10-31 定位）**：ConvFormer 整个主干一张图编译，fp32 精确，bf16 autocast 下
  误差 0.02–0.84 随批量大小跳变、训练器内少数批次出 NaN；逐算子、单 block、stem+1 block 都对，stem（bf16 conv/linear 与 fp32 嵌入拼接）+ ≥2 block 同图就错。
  修法：stem eager、只编译 block 栈，精度与 eager bf16 相同、速度不损。**规矩**：任何新 compile 路径上训练前，必须在真实训练器里逐调用对照 eager
  （`FC_DEBUG` 式挂钩）跑 ≥2 迭代（含更新），并扫 ≥40 个批量大小；随机输入的微基准测不出来（本例前三轮探针全部漏掉）。
- **Rust rollout 只认原生 374 动作空间（2026-10-31，exp124）**：`collect_rust` 对 `net.action_space` 不做任何分支，46 槽（Mortal / bc49）网络在 Rust 引擎上
  根本跑不起来——此前所有 46 槽 RL（exp41/46/96）都走 Python 引擎。新增 `m46_native.py` 从原生掩码向量化导出 46 槽（吃只需被鸣牌 = v3r 第 13 平面）。
  对拍时抓到的坑：吃的三种形状按"被鸣牌 ± 偏移"取原生键，花色边缘（如鸣 2m 时 high 形的键 −1）裁剪到 0 会**串到另一形状的槽**，必须按同花色判有效。
  **规矩**：①换动作空间 / 引擎组合前先查 rollout 路径是否真支持，不信"架构在 zoo 里就能训"；②新映射必须对拍参考实现（这里是 `MortalActionSpace`），
  用"门清向听最优"一类策略专门刷出立直 / 和牌 / 选牌，随机策略几乎碰不到；③pod 镜像的 torch 与本机不同（2.8 vs 2.12），compile 路径在 pod 上先过
  FC_DEBUG 闸门再开训（`exp124_ops/pod_boot.sh`）。
- **2026-10-31 exp124 发射后 5 分钟两臂同崩：种子不是局 ID**。46 槽 rollout 的行键用 (seed, slot, seat)，但 dup_k=8 下同一 VecEnv 槽位一轮内可能两次拿到同一种子 ⇒ 键撞车，
  改写的计数校验 raise（好在是 raise 不是静默错位）。本机冒烟 4–7 迭代撞不到，pod 上 ~20 迭代必撞。修复为"该槽已完成局数"计数器（a8fef66）。
  **规矩**：①跨步对齐的键必须用引擎/宿主自己维护的唯一局序号，不用种子（dup / 重放让种子重复）；②稀有碰撞类 bug 不能靠"冒烟跑几轮没事"，
  要写**强制触发**的测试（这里：4 槽 × 同一种子 6 次）；③pull_loop 收到 TRAIN_FAILED 会发 PULLED_OK 让 pod 自毁——要原地修就先停 pull_loop 与 pod_run 再动手。
- **2026-11-01 工作站无人值守重启吞掉整套监控**：内核自动更新（6.8.0-146→147）01:17 重启，exp124 的 pull_loop / watchdog / eval_loop / TensorBoard（全是 setsid nohup）
  一起消失，pod 照常计费训练；预注册的"8M 选臂即终止败臂"因此没执行，败臂多烧 12 h ≈ $11，发现靴子落地是 13 h 后人工巡检。
  **规矩**：①本机侧长跑监控必须抗重启——用 `systemd --user` 单元（`Restart=always` + `loginctl enable-linger`）或 `@reboot` crontab 自愈，setsid 只配短任务；
  ②凡"到 X 里程碑自动做 Y"的关键决策（选臂 / 终止 pod），不能只靠本机循环，pod 侧也要有兜底（例如 pod_run 在里程碑后自查工作站心跳文件，超时自停）；
  ③回来第一件事是 `uptime` + `ps` 对账，再看 pod。

## 2026-11-01 方法论纠错：架构轴被 2M 局的判决锁死了四个月（exp124 教训，用户点名复盘）
exp124（bc49 同结构 ConvFormer46 纯血从零）32M 局读数 0.481，比 cnn_l 同里程碑高 +5 pp、高于 cnn_l 的 192M 终点和"纯血平台 ≈0.47"；雀魂 B+/A-（n=2）。
此前 ≥100M 的纯血 run 清一色 CNN（exp27A→exp90→exp101→exp113 384M），七个杠杆（规模/目标/姿态/先验/估计器/归因/搜索）全在 CNN 上换，
"平台"其实是一个架构的平台。错在流程，不在某次实验：
- **小规模判决被当成永久判决**：HandSet/HRF/Mortal 设计（exp27/30/36/37/41，1–2M 局）判"架构上限"后再没在规模上重开；
  我们自己 08-28 写的"小数据点的架构排序不可外推、设计比较必须在目标数据量上终审"只用在了 BC 数据缩放上。
- **单架构结论被当成问题性质**：exp68/69"感知不是瓶颈"、exp113"规模 H0"都只对 CNN 成立，却被读成"方法到顶"。
- **归因混淆**：exp45/49"示范信号 ≫ 自对弈信号"比的是 ConvFormer46+人类数据 vs CNN+自对弈，两个变量同时变，功劳全记给了数据。
- **regime 隔离只用了一个方向**：08-28 同时写了"结构才是兑现变量"（BC）和"从零自对弈的架构定论不可外推到示范 regime"，却没把 BC 的强结果当 RL 的待检假设。
- **基础设施替科学做决定**：Rust rollout 只认 374 空间，46 槽网络只能走慢 10× 的 Python 引擎，"跑得快的"决定了"上规模的"；接线只花一晚。
- **流程放大器**：10-29 的"提议前 grep 三处台账"防重复失败，副作用是旧判决无范围戳就被当永久；CNN 谱系 384M 的沉没成本让它成了"纯血线"本身。
**规矩（已写入 CLAUDE.md 关键规则 + ml-experiment-tracking skill）**：
① FINDINGS/INDEX 的每条判决必须带范围戳（架构 / 信号源 / 规模 / 评测口径）和失效条件，引用判决先看戳，戳不匹配即可提"在目标范围复测"；
② 宣布"方法平台"前必须在目标规模上换过模型类（≥1 个非当前主干跑到 ≥50M）；③ zoo 里上不了快引擎的架构在台账标"无法在规模上检验"而非判负；
④ 一个 regime 的强结果自动登记为另一 regime 的待检假设；⑤ 10-29 grep 规矩修订：grep 命中 ≠ 方向关闭，命中后必须核对范围戳。
仍待检：主干 vs 动作头的贡献（Stage 2 消融）、192M 终点判决；上述流程错误与 192M 结果无关，已成立。
- **2026-11-02 Community pod 可能没有公网 TCP（只能走 ssh.runpod.io 代理）**：exp124 迁移订到的 Community 4090 runtime 只映射了 http 端口，`ssh.direct` 永远为空；
  RunPod 代理只给交互式 PTY，`ssh host cmd`、scp、rsync 一律失败 ⇒ 依赖 rsync 的 deploy/migrate 流程整个不可用。**规矩**：开机后 2 分钟内看 `runtime.ports` 有没有 `22/tcp` 公网映射，
  没有就直接 delete 再等（$0.02），不要试图改走代理；create-pod 没有"要求公网 IP"的过滤项。
- **2026-11-03 迁移两坑**：①Community 宿主可能连不上 sh.rustup.rs（curl 443 超时）⇒ bootstrap 装不了 cargo 就编不出 riichi_rs wheel；ops 现随附预编译 cp312 wheel（`exp124_ops/riichi_rs-*.whl`，pod_boot 优先用它）。
  ②migrate.sh"等下一次 latest.pt 存盘再停源机"盯错了目录（第一段 `exp124_A/` 已不更新，活动目录是 `_p2`），40 min 超时后硬停，丢 0.7M 局；已改为自动选活动目录。**规矩**：任何"等文件更新"的守候都先确认盯的是当前写入的文件。
- **2026-11-03 Community ≠ 便宜：先量吞吐再定去留**。exp124 迁到 Community 4090（$0.34/h）后吞吐 89 局/s vs Secure 432 局/s（宿主 EPYC 7B13、cgroup 27 核、单核慢；load avg 从 75 降到 15 后速度不变 ⇒ 结构性）。
  Rust 自对弈 rollout 是 CPU 密集，Community 卡价格只省 GPU 不省 CPU：余 80M 局 Community ≈$85/10 天 vs Secure ≈$46/52 h。**规矩**：迁移后首 10 分钟必读 `局/s`，低于 Secure 的 40%（本项目 ≈170 局/s）即迁回；
  `migrate.sh` 之前先在目标机跑 1 迭代 bench（`pod_boot.sh` 的 gate 已有 4096 局计时：gate 5.6 min ≈ 12 局/s 的 gate 数字只测编译不测吞吐，不要拿它当吞吐）。create-pod 可能被 auto 模式分类器拦截，迁回要用户放行。
- **2026-11-03 迁机三条补充**：①挑机器看 vCPU 不看卡型——同一时刻 Secure 4090 只有 16 vCPU 宿主而 3090 有 32 vCPU，3090 $0.50/h 跑出 309 局/s（4090/55 核 432、Community/27 核 89）；`create-pod` 用 `gpu.minVcpuCountPerGpu` 过滤，缺货就换卡型再试。
  ②RunPod 有的宿主 /workspace 禁 chown：`rsync -a` 会以 rc=23 退出（文件其实传齐），脚本里一律 `rsync -rlt`。③迁移脚本的"等下一次存盘"要带新鲜度短路：慢机上两次 latest.pt 相隔数小时，否则白等 40 min 还丢进度。
- **2026-11-03 rollout 剖析纠错：先分段再下结论**。我曾对用户说"Rust 引擎吃 CPU"，分段后发现 exp124 rollout 54% 是 GPU 前向（46 槽立直/杠第二次查询对同一观测重算整网占 15%），Rust 只占 ≈25%。
  Community 宿主慢的真因是单核慢拖累前向的 kernel 派发 + Rust step。**规矩**：吞吐问题先用计时包装（monkeypatch VecEnv 方法 + cuda 同步计时前向）分段，再提方案。
  异步重叠的关键是给 rollout 的 CUDA stream 高优先级（priority=-1）：普通优先级时后台 rollout 被更新抢 GPU 拖慢 1.8× 成关键路径（4.30 s/迭代），高优先级后 3.55 s。
- **2026-11-03 梯度噪声探针**（`--grad_noise_probe`，分支 claude/perf-rollout-overlap）：回答"batch 该多大"先测 B_simple 再花钱。exp124 的策略梯度在每迭代全部数据量上都是纯噪声（|g|∝1/√b），价值梯度在共享主干上大 12–21×、clip 每步生效。
- **2026-11-03 流式 rollout 三个坑**：①torch.compile 不是线程安全的，dynamic=True 仍对批量 0/1 特化——后台线程里遇到 1 行的批次会现场编译，曾在块尾卡死一次；所有图必须在主线程预热（`StreamActor.warmup`）。
  ②训练器 `if n_eff < batch: continue` 会静默丢弃不足一个 minibatch 的数据，小块场景下丢一半；切分要按 round(n/batch) 均分。③每个小迭代都全量重写 train_log.json 是平方级，6400 块后每迭代多 40 ms；按 2048 局聚合。
  另：流式的数据过时度由演员并发 K 和队列深度决定（稳态一局寿命 ≈ K / 每步消费局数），不是由演员速度决定；估算时别假设演员不受限速。
- **2026-11-04 解析标记行别用 `cut -d= -fN`**：exp126 完成标记 "TRAIN_DONE arm=async md5=…" 含两个等号，`cut -d= -f2` 取到 "async md5"，md5 握手失败 20 分钟，pod 差点空等 3 小时。按键名解析（`sed 's/.*md5=//'`）；改运行中的 bash 脚本要写新文件再 mv（bash 边执行边读原文件）。
- **2026-11-05 镜像循环会被 pod 端改动覆盖**：pull_loop 用 `rsync -a --inplace` 每 10 分钟镜像整个 run 目录，若 pod/宿主被入侵、旧检查点在 pod 上被改，本地镜像会被静默覆盖。最低限度：巡检比对旧里程碑 mtime（已加入 cron）；更稳的做法是里程碑文件只拉一次（`--ignore-existing`），只让 latest.pt 和日志保持更新，拉到后记 md5。**已于 11-05 在 exp124/125 pull_loop 落地**：里程碑须在 pod 上静置 ≥2 min（`find -mmin +2`）才拉，防止拉到写了一半的文件后永不更新。
