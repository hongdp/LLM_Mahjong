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
