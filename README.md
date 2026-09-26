# LLM Mahjong — Riichi Mahjong RL (two lineages)

English | [中文](README.zh.md)

Riichi mahjong agents trained by self-play RL, in two parallel lineages (since 2026-08-27):

- **Human-prior lineage** (current architecture carrier). North star: iterate a *simple-input-plane* model from a
  human-game prior until it surpasses Mortal. Path: Tenhou Phoenix behaviour cloning (bc49) → RL on the prior
  (exp46 fixed four trainer pathologies) → hanchan placement training (exp55-D).
- **Pure lineage** (AlphaZero-style). Zero human data, zero external models as training signal; the model must
  discover the whole skill stack from random init. Human/teacher models are yardsticks only — never champions,
  never opponents in its pool; scenario curricula are permanently rejected; leagues of its own frozen snapshots
  and EMA self-anchors count as pure.

**Required reading**: [CLAUDE.md](CLAUDE.md) (rules + project map) · [SKILLS.md](SKILLS.md) (lessons / hardware /
pitfalls) · [experiments/INDEX.md](experiments/INDEX.md) (one line per run) ·
[experiments/FINDINGS.md](experiments/FINDINGS.md) (results ledger).
**To run the strongest model**: [docs/champion_model.md](docs/champion_model.md) (model card, four ways to run it,
resources, version history) · [experiments/LEADERBOARD.md](experiments/LEADERBOARD.md) (current board).

## Current status (2026-09-26)

**Deployment champion = bc49** (human-prior lineage; ConvFormer × v3r encoder × 46-slot action head, 2.00M params,
pure BC on 18.5k Phoenix games, holdout accuracy 0.806). Greedy (T=0) in play and in every final verdict.
On the hanchan all-greedy ladder (the cleanest scale, 13 anchors, 200 duplicate pairs each): Mortal 298k 1465 ± 12,
exp46-I 1457 ± 11, **bc49 1443 ± 11**. Mortal head-to-head over 1,200 hanchan: 0.477 ± 0.014 vs bc49 — tied, i.e.
"match Mortal" reached, "surpass" not. Majsoul maka S+ twice. Details and ckpt paths in
[docs/champion_model.md](docs/champion_model.md); numbers in [experiments/LEADERBOARD.md](experiments/LEADERBOARD.md).

**Pure lineage — the $200 program (2026-09-06 → 09-21, ≈$92 spent)**. Goal: a from-scratch model at bc49/Mortal
level (T=0 single-deal share ≥ 0.50 vs bc49 over ≥ 20k paired walls, and hanchan ≥ 0.50 over ≥ 1,200 matches).
Best so far: **exp88_Q1x, 0.4568 ± 0.0035 single-deal vs bc49**; hanchan 0.37–0.39. Every terminal number lives in
[experiments/INDEX.md](experiments/INDEX.md); the program design and verdicts in
[experiments/designs/design_pure_line_200usd_program.md](experiments/designs/design_pure_line_200usd_program.md).
What was learned (exp70–exp97):

- **Scaling is log-linear**, ≈ +1.5–2 pp per doubling, and continued training of the same recipe asymptotes at
  0.45–0.46 (lr/entropy anneals exhausted, exp78/80; +1 pp from high-entropy regularisation, exp88).
- **Closed levers** (each pre-registered, each judged on 20k-pair T=0 evals): hanchan/placement objectives
  (exp70/83/84/91), value-based methods ×4 (exp59/60/71/96), PBRS (exp73/74), counterfactual-rollout advantages
  (exp72), PSRO best response (exp87), QRE/KL magnets (exp88), style-conditioned populations (exp89), global and
  shanten-weighted deal-in penalties with anneal (exp85/90/95 — the induced defence snaps back when the penalty is
  removed), ensembles (exp92), run-time adaptation (exp93), oracle guiding and auxiliary wait heads (exp68/69),
  same-lineage opponent pools (exp82), inference-time search (exp86).
- **Diagnosis** (exp94 counterfactual takeover probes, true-table clones, all greedy, deal-clustered SEs): the gap to
  bc49 is one skill — *multi-step* play when a far hand faces a riichi. bc49 taking over from an exposed decision is
  worth **+258 ± 94 pts/decision**, all from deal-ins 16.5 % → 11.5 % with win rate unchanged; any *single-step*
  change is worth ≈ 0 for either learner at either table. Policy gradient sees only the single-step quantity, so the
  chain never gets climbed. First positive pure macro: "shanten ≥ 2 → discard only genbutsu, chosen by the model's own
  efficiency" = +108–153 pts/decision (z ≈ 3, 18k deals), ≈ +19 ± 9 pts/deal.
- **Closed since 09-21** (all H0, pre-registered): parameter-space noise (exp97, trainer failed without IS), greedy-continuation
  single-deviation PPO (exp98), learned danger-set macro (exp99), mode-sequence PIMC search (exp100), static safe-tile keeping.
  Game-theoretic defences (opponent pools, PSRO, entropy/QRE, style populations) were measured too: the population sits at an
  equilibrium that nobody can exploit — a low-quality one, since both sides lack the same multi-step skill; no cycling, no
  degenerate signal. A stage-by-turn probe shows the model already folds in the last few turns (live-tile rate 96 % below chance)
  and the gap opens toward the early game.
- **Running**: **exp101** — the last untested axis, capacity × scale: `cnn_l_v3r` (4.0M params, 2× cnn_m) from scratch toward
  400M deals on one Secure RTX 4000 Ada ($0.28/h, ≈408 deals/s, cap $100). Decision points 32M / 64M / 112M (≥0.462 vs bc49 =
  capacity pays; ≤0.452 = switch the budget to pure scale of Q1x); goal ≥0.50 at the end. Spend to date ≈$111 of $200.

**Infrastructure**: the deal engine is ported to Rust (`rust/riichi_rs`, bit-exact parity with the Python engine,
≈ 575 deals/s on one RTX 2000 Ada, cost per million deals $3.1 → $0.07). Training runs on RunPod (Secure Cloud
whenever champion weights are on the pod); the local RTX 4080 is for smoke tests, probes and evaluation only.

## Two phases

- **Phase 1 (2026-05 → 08-14, archived)**: LLM (Qwen + LoRA) + text rollouts + PBRS/PPO. All arena results null,
  returns not decodable from hidden states → retired. Legacy: the engine, the reward registry, the arena protocol,
  the GCP workflow. Archives: `experiments/reports/report_exp1..exp5`, `src/core/`, `scripts/phase1_ce/`.
- **Phase 2 (current)**: small dedicated networks (2–23M params) + tensor encodings + self-play PPO / BC / DQN.

## Architecture (Phase 2, active parts)

```
src/tasks/mahjong/
├── table.py            # 136-tile deal engine; epoch-4 rules = Majsoul single-deal (red fives, abortive draws,
│                       #   double yakuman, nagashi mangan, kan-dora timing, kokushi ankan-chankan, chankan furiten)
│                       #   + round-context randomisation; reward = start-point delta (+ placement bonus)
├── claims.py           # call-window arbitration (ron > pon/kan > chi, double/triple ron)
├── hanchan.py          # full hanchan: renchan / honba / nagashi / uma — the verdict scale
├── arena.py            # duplicate-wall arena (A−B symmetric pairing)
└── shanten.py          # shanten / acceptance / dora mapping
rust/riichi_rs/         # Rust port of the engine + encoders + VecEnv (vectorised self-play, hanchan, league seat
│                       #   routing, per-seat reward styles, genbutsu masks); bit-exact with the Python engine
src/agents/dnn/
├── encoder.py          # observation encodings v1/v1r/v3/v3r(+red)/v3s/v4; 374-slot and 46-slot action spaces
├── arch_zoo.py         # cnn_m_r / cnn_m_v3r (pure line) / convformer_m_v3r_m46 (bc49) / ensembles /
│                       #   AnchoredQPolicy (prior-anchored Q head, exp96) / handset / HandRiverFormer / vit
├── net.py              # base net + load_compatible (cross action-space / variant checkpoint loading)
├── selfplay.py         # Python-engine self-play (play_game / generator play_game_gen) — probes and bc49 tables
├── rust_rollout.py     # Rust-engine rollout (collect_rust): league seats, reward styles, uint8 planes
├── parallel_rollout.py # multi-process Python rollout + GPU batch inference (infer_server.py)
├── style_stats.py      # capability metrics (win / deal-in / riichi / call rates, turns) for TB and evals
└── mjai_bridge.py      # Majsoul live bridge (MJAI shadow table; encoder + legal actions reused unchanged)
scripts/
├── train_dnn_ppo.py    # PPO trainer: --engine rust, dup_k=8 duplicate-wall group baseline, GAE, entropy
│                       #   schedules / target-entropy dual control, KL anchors, league pools, PBRS shaping,
│                       #   deal-in penalties (global / shanten-weighted / annealed), counterfactual advantages
├── train_dnn_bc.py     # behaviour cloning on Tenhou logs (bc49 recipe)
├── train_dnn_dqn.py / train_dnn_qstitch.py   # value-method line (Double DQN on a prior; prior-anchored Q-stitch)
├── run_elo_league.py   # anchor pool Elo (single-deal and --hanchan, vectorised GPU, engine-fingerprint guard)
├── rating.py           # rating system v2 (append-only ledger, four-entity tables, pt board) — epoch 7 pending
├── probe_*.py / eval_style_profile.py         # probe family: defence IQ, decomposition, conditional entropy, style
├── serve_mjai_bot.py   # HTTP agent service for the Majsoul bridge
└── phase2_dnn/         # cloud launch scripts (GCP G4 flex; RunPod runbooks live in SKILLS.md)
tools/webui/            # inspector: training curves, self-play viewer (per-step probabilities / V), replay
tools/majsoul_bridge/   # MahjongCopilot plugin (live play = champion greedy; one of the three human yardsticks)
```

## Evaluation

Three scales, never mixed in one table (see the notes at the top of
[experiments/LEADERBOARD.md](experiments/LEADERBOARD.md)):

1. **Anchor-pool Elo** (`experiments/elo_league/`): 13 anchors, sign-MLE, `bc_cnn` pinned at 1000. Engine changes
   invalidate history (engine-fingerprint guard; a new epoch = full recalibration). Anchors play under the
   temperature they were calibrated at (stored in the pool file).
2. **Hanchan scale** (`hanchan.py`, own anchor pool under `experiments/elo_league/hanchan/`): the verdict scale.
   Vectorised GPU evaluation ≈ 967 hanchan/min; Mortal is rated with `rate_mortal_hanchan.py`.
3. **Probes and human yardsticks**: defence IQ, style profile (human reference: agari .212 / houjuu .125 /
   riichi .182 / call .338), counterfactual takeover probes (exp94), Majsoul maka grades.

**Verdict protocol** (CLAUDE.md): candidates at T=0, out-of-family ladders, hanchan n ≥ 300 for finals; single-deal
finals are 20k paired walls in two seed segments (SE ≈ 0.0035); in-family T=1 curves never decide anything on their
own. Probes that take several decisions per deal report **deal-clustered** standard errors (per-decision SEs
understate by ≈ 2×). A champion changes only on a significantly positive T=0 head-to-head; ladder scores rank, they
do not crown.

## Quick start

```bash
conda activate rlhf_mahjong
python -m pytest tests -q                              # engine parity, encoders, trainers, Rust ↔ Python
# build the Rust engine (after any change under rust/riichi_rs)
cd rust/riichi_rs && maturin build --release -i "$(which python)" && pip install --force-reinstall target/wheels/riichi_rs-*.whl && cd -
# serve the champion (bc49, greedy) for the Majsoul bridge — full manual in docs/champion_model.md
PYTHONPATH=. python scripts/serve_mjai_bot.py --ckpt experiments/_anchors_epoch6/bc49.pt --temperature 0
# pure-line PPO smoke on the Rust engine (long runs go to the cloud — see SKILLS.md for the RunPod runbook)
python scripts/train_dnn_ppo.py --engine rust --arch cnn_m_v3r --total_games 200000 --games_per_iter 2048 \
  --dup_k 8 --games_per_worker 1024 --workers 1 --exp_dir experiments/my_run_$(date +%Y%m%d_%H%M%S)
# T=0 paired evaluation of a checkpoint against bc49 (single deal / hanchan)
python -c "from scripts.run_elo_league import play_pair_vector as p; sc,_,_=p('CAND.pt','experiments/_anchors_epoch6/bc49.pt',4000,67000000,20,'cuda',temp_a=0.0,temp_b=0.0,hanchan=False); print(sum(sc)/len(sc))"
conda run -n rlhf_mahjong python tools/webui/server.py --port 8642   # inspector
```

Every run needs an `EXPERIMENT.md` (purpose / method / success criteria) *before* launch, a heartbeat monitor and
TensorBoard mirror once launched, and a row in `experiments/INDEX.md` when it ends — see CLAUDE.md.

## Live play on Majsoul (Windows client machine)

The human yardsticks (maka grade, placements, deal-in rate) can only be read from real games. Standard topology is
two machines — the model machine runs this repo and the checkpoint, the client machine (Windows) runs
[MahjongCopilot](https://github.com/latorc/MahjongCopilot) (MC) + Chrome — joined by an SSH tunnel. A single-machine
setup works too.

```
client Windows 11: MC + plugin ── mitmproxy:10999 ──► Chrome (Majsoul)
                      └─ bot_llmmahjong ──► 127.0.0.1:8765 ──ssh -L tunnel──► model machine: serve_mjai_bot.py
```

1. **Model machine** (Linux, repo root): start the agent service with the champion, greedy:
   ```bash
   PYTHONPATH=. python scripts/serve_mjai_bot.py --ckpt experiments/_anchors_epoch6/bc49.pt \
     --temperature 0 --log experiments/exp24_majsoul_live_$(date +%Y%m%d_%H%M%S)/mjai_session.jsonl
   ```
   `curl localhost:8765/health` returns ok when ready. The service has **no auth and listens on 127.0.0.1 only**;
   never expose it.
2. **Install MC on the client** (PowerShell; do not use an old conda on that machine):
   ```powershell
   winget install Python.Python.3.12 --scope user
   git clone https://github.com/latorc/MahjongCopilot $env:USERPROFILE\MahjongCopilot
   cd $env:USERPROFILE\MahjongCopilot; python -m venv venv; .\venv\Scripts\pip install -r requirements.txt; .\venv\Scripts\playwright install chromium
   ```
3. **Apply the three Windows patches and install our bot plugin** (patches verified against MC `31be3de`):
   ```powershell
   git apply <this repo>\tools\majsoul_bridge\mahjongcopilot_windows.patch
   python <this repo>\tools\majsoul_bridge\install.py $env:USERPROFILE\MahjongCopilot
   ```
   The patches fix the three Windows blockers: Playwright's bundled Chromium SxS error (use system Chrome), the
   46 MB Majsoul wasm buffered by mitmproxy (stream large responses), and the mitm root certificate needing admin
   (install into the current-user store).
4. **Tunnel** (client, keep open): `ssh -N -L 8765:127.0.0.1:8765 <model machine>`; MC's URL stays
   `http://127.0.0.1:8765`.
5. **Configure and start MC**: in `settings.json` set `"model_type": "LLM_Mahjong"`,
   `"llmmahjong_url": "http://127.0.0.1:8765"`, `"ai_randomize_choice": 0`; `enable_automation` = `false` for
   assist mode (you click; the panel shows probabilities / V) or `true` for auto-play (official scoring).
   Start: `cd $env:USERPROFILE\MahjongCopilot; .\venv\Scripts\python.exe main.py` → launch browser → log in.
6. **Verify and score**: play one friendly-room game first; every `Bot in: tsumo` in the MC log must have a
   `Bot out: dahai` and `no op list` must be 0. Afterwards, on the model machine:
   `python scripts/analyze_majsoul_session.py <session>.jsonl` (placements / wins / deal-ins / riichi / calls).

**Detailed runbooks**: [tools/majsoul_bridge/README.md](tools/majsoul_bridge/README.md) (general flow, both modes,
session log format, protocol pitfalls) · [tools/majsoul_bridge/WINDOWS.md](tools/majsoul_bridge/WINDOWS.md)
(Windows field notes: symptoms and root causes of the three patches, new Unity client compatibility,
`spawn UNKNOWN` / black screen / certificate troubleshooting) · [docs/champion_model.md](docs/champion_model.md)
(which checkpoint, how much hardware).

> **Risk**: third-party automation violates Majsoul's terms of service and **can get the account banned**. Use only
> accounts you can afford to lose.

**Discipline** (enforced by CLAUDE.md): pre-register every run; check throughput after launch; heartbeat on every
long job; terminate cloud machines when done (never merely stop them); reward logic goes through the registry; new
lessons are appended to SKILLS.md with a date.
