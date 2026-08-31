# LLM Mahjong — Riichi RL (two lineages) — Pure Self-Play Riichi Mahjong RL

English | [中文](README.md)

> **EN mirror status (2026-08-30)**: the Chinese README is authoritative and freshly updated;
> the sections below still carry 2026-08-23 wording. Headlines: two parallel lineages
> (human-prior = current main carrier, north star = a simple-input-plane model surpassing Mortal;
> pure self-play unchanged). Deployment champion = **bc49** (human BC flagship, 2.00M params /
> 56 planes / 46 actions): **1189.0 ± 7.9 on the epoch-6 deployment scale**, Majsoul maka S+
> twice; real Mortal 298k reference 1199.6 ± 8.0 (same protocol) — **gap ≈ 10 ± 11**.
> **How to configure, run and resource that model: [docs/champion_model.md](docs/champion_model.md)**
> (model card + deployment runbook + champion version history; §9 is an English quick reference).
> exp46 C~J fixed four trainer pathologies (value-gradient trunk corruption -> --value_detach,
> entropy diffusion -> KL anchor, advantage tail censoring -> clamp removed, T=1 in-family
> metric distortion -> protocol switch); best RL artifact exp46-I ties bc49 (0.5005 head-to-head
> at T=0 both sides). Epoch 6 is open: engine action-gap fixes merged, 13-anchor recalibration
> done — current board in [experiments/LEADERBOARD.md](experiments/LEADERBOARD.md). exp55-D
> hanchan-placement training pipeline is built and ready to launch. See
> experiments/INDEX.md and experiments/FINDINGS.md.

**North star (goal a)**: inspired by **AlphaZero** — pure self-play with zero human/teacher knowledge. Starting from random initialization, the model must discover the full skill stack
on its own (tile efficiency → riichi/closed-hand play → hand value → defense) and climb toward human expert
level; the emergence of defense is a milestone, and reaching the goal is the goal itself. Goals (b) — learning
new mahjong knowledge from the model — and (c) — transferring the methodology — come later. Purity rules:
teacher-prior models serve only as Elo yardsticks, never as champions or training opponents; scenario
curricula (starting deals from opponent-riichi states) are permanently rejected; league play (opponents =
frozen snapshots of the model itself) counts as pure.

**Required reading**: [SKILLS.md](SKILLS.md) (lessons / hardware / pitfalls — read before developing) ·
[docs/roadmap_epoch3.md](docs/roadmap_epoch3.md) (current roadmap: queue, entropy & stochasticity,
capability-ordering) · [experiments/INDEX.md](experiments/INDEX.md) (experiment ledger)

## Two phases

- **Phase 1 (2026-05 to 08-14, archived)**: LLM (Qwen + LoRA) + text rollouts + PBRS/PPO. Verdict: all
  arena results null, returns undecodable from hidden states → line retired. Legacy = the engine, the
  reward registry, the arena protocol, and the GCP workflow. Archives: experiments/reports/report_exp1..exp5,
  `src/core/`, `scripts/phase1_ce/`.
- **Phase 2 (current)**: small dedicated networks (2–23M params) + tensor encodings + pure self-play PPO.
  One 1.0M-game training run ≈ 85 minutes / ~$3 on a g4-standard-48 flex-start VM.

## Architecture (active Phase-2 parts)

```
src/tasks/mahjong/
├── table.py            # 136-tile table engine; epoch-4 rules = Majsoul single-hand alignment
│                       #   (red fives / abortive draws / double yakuman / nagashi mangan / open-kan
│                       #   dora timing / kokushi robbing an ankan / chankan furiten) + match-context
│                       #   randomization (East-1 starts exactly equal, score spread sigma=4500*sqrt(k)
│                       #   by round progression, carried sticks / West round; reward = delta from the
│                       #   starting scores + placement bonus)
├── claims.py           # claim-window resolution (ron > pon/kan > chi, double ron, triple ron = draw)
├── arena.py            # duplicate-deal arena (symmetric A−B paired diff; same seed = same context)
└── shanten.py          # shanten / ukeire / dora mapping
src/agents/dnn/
├── encoder.py          # observation encodings v1 / v1r (+red & yakuhai planes) / v3 (complete public
│                       #   record) / v4 (event buffer); 374-action space (11 types x 34 key tiles,
│                       #   legacy checkpoints widened automatically)
├── arch_zoo.py         # cnn_m_r (champion, 2M) / cnn_xl_r / handset_* (tile-instance set attention) /
│                       #   HandRiverFormer (exp30: hand tokens cross-attending the river event
│                       #   sequence) / ConvFormer / vit
├── net.py              # base nets + load_compatible (checkpoint loading across action spaces/variants)
├── selfplay.py         # self-play (play_game / generator play_game_gen), per-game style facts
├── parallel_rollout.py # multi-process rollouts; vectorized worker (K games per process, one batched
│                       #   RPC per round — cnn 204 games/s locally)
├── infer_server.py     # batched GPU inference server (shared-memory slots / CUDA-graph buckets /
│                       #   multi-model hosting)
├── style_stats.py      # capability metrics (win & deal-in rates and turns, riichi/call rates) —
│                       #   shared by training TB and evaluation
└── mjai_bridge.py      # live Majsoul bridge (MJAI shadow table reusing the encoder & legal-action
                        #   generators unchanged)
scripts/
├── train_dnn_ppo.py    # PPO trainer (GAE lambda=0.95, dup_k=8 duplicate deals with leave-one-out
│                       #   baseline, entropy schedules / target-entropy dual control, mixed-temperature
│                       #   behaviour-policy logprobs, --gpu_infer, style/* TB metrics)
├── run_elo_league.py   # Elo anchor pool (epoch 4 = 9 anchors, engine-fingerprint guard,
│                       #   --temperature for greedy rating)
├── elo_ladder_watcher.py / watch_run.sh   # in-training ladder rating + heartbeats (every long-running
│                       #   task gets one)
├── probe_defense.py / probe_decomposition.py / probe_conditional_entropy.py / eval_style_profile.py
│                       # probe suite: defense IQ / decomposition agreement / conditional-entropy curve /
│                       #   style profile (--vs_anchors for the ecology-free reading)
├── run_arena_dnn.py    # duplicate-deal arena (--override_* diagnostic wrappers, per-side temperatures)
└── phase2_dnn/         # cloud workflow: launch_g4_git.sh (G4 flex + pinned-SHA gate), run_dnn_cloud.sh
tools/webui/            # inspection console: training curves + self-play dashboard (per-step
                        #   probabilities / V) + Majsoul-style replay
tools/majsoul_bridge/   # MahjongCopilot plugin (live play = champion greedy; one of the three yardsticks)
```

## Evaluation (three yardsticks)

1. **Elo anchor pool** (`experiments/elo_league/`): 9 anchors, sign-based MLE, bc_cnn pinned at 1000.
   **Epoch rule**: any engine change voids all historical matches and forces a full recalibration
   (anchors.json carries an engine content fingerprint; rating refuses on mismatch). Epoch 4 in force:
   teacher reference line bcrl14 at 1107.7; champion exp27A at 1059 in-pool / 1121.8 as a greedy candidate.
2. **Probe suite**: defense_iq (is folding conditioned on one's own hand), decomposition probe
   (agreement with the tile-efficiency oracle), conditional-entropy curve (does randomness follow the
   value gap — verified monotonically decreasing for all three recent champions), style profile
   (win/deal-in rates and turns against fixed anchors).
3. **Human scale**: live Majsoul play (MahjongCopilot bridge, greedy) — the in-game AI review grade
   ("maka", first reading C+ at n=1) plus placement / deal-in statistics.

## Current status (2026-08-23)

- **Champion = exp27-A** (cnn_m_r, epoch-3 native: knows red fives / yakuhai planes): from scratch,
  1.0M games matches the old lineage's 2.1M-game strength; greedy rating 1121.8 is the all-time high.
  Live deployment always uses the A lineage, greedy.
- **Epoch 4 in force**: rules-audit fixes + match-context randomization (gives placement pressure a
  within-hand learning signal).
- **Running**: exp31, four arms (target-entropy recipes x scale recheck, G4 flex); exp30 HandRiverFormer
  preregistered and awaiting the winning recipe.
- **Refuted hypotheses** (see INDEX): teacher-prior路线, scenario curricula, v3 inputs, GAE x ConvFormer
  additivity, defense-after-attack-saturation, league-induced defense, hand-instance sets over CNN,
  mixed-temperature gains.
- **Defense**: defense_iq ≈ 0 and independent of entropy level — the bottleneck is credit, not sampling;
  the current lever is context randomization + placement pressure; the next surgery (multi-hand match
  structure) is deliberately postponed (needs a per-match reward redesign).

## Quick start

```bash
conda activate rlhf_mahjong
python -m pytest tests -q                        # ~196 tests
# run the current champion bc49 (Majsoul live-play server, greedy) — full runbook in docs/champion_model.md
PYTHONPATH=. python scripts/serve_mjai_bot.py --ckpt experiments/_anchors_epoch6/bc49.pt --temperature 0
# local training (RTX 4080: cnn_m_r ~100 games/s at the trainer level)
python scripts/train_dnn_ppo.py --arch cnn_m_r --total_games 1000000 --gpu_infer \
  --games_per_worker 32 --infer_max_batch 512 --exp_dir experiments/my_run_$(date +%Y%m%d_%H%M%S)
# cloud (G4 flex; push first — the launcher verifies the SHA is on origin/master)
bash scripts/phase2_dnn/launch_g4_git.sh my-vm us-central1-b my_run $(git rev-parse HEAD) -- \
  scripts/train_dnn_ppo.py --arch cnn_m_r ... --exp_dir experiments/my_run
conda run -n rlhf_mahjong python tools/webui/server.py --port 8642   # inspection console
```

## Live Majsoul testing (Windows play machine)

The human-scale yardstick (in-game "maka" grade, placement, deal-in rate) can only be read from real
games. Standard topology is **two machines**: the model machine runs this repo plus the checkpoint, the
play machine (Windows) runs [MahjongCopilot](https://github.com/latorc/MahjongCopilot) (MC) + Chrome, and
an SSH tunnel connects them. A single-machine setup (both on the same box) works just as well.

```
Windows play machine: MC + plugin ── mitmproxy:10999 ──> Chrome (Majsoul)
                          └─ bot_llmmahjong ──> 127.0.0.1:8765 ──ssh -L tunnel──> model machine: serve_mjai_bot.py
```

1. **Model machine** (Linux, repo root) — champion, greedy:
   ```bash
   PYTHONPATH=. python scripts/serve_mjai_bot.py --ckpt experiments/_anchors_epoch6/bc49.pt \
     --temperature 0 --log experiments/exp24_majsoul_live_$(date +%Y%m%d_%H%M%S)/mjai_session.jsonl
   ```
   `curl localhost:8765/health` returning ok means it is ready. The server has **no auth and binds
   127.0.0.1 only** — never expose it to the internet.
2. **Install MC on the play machine** (PowerShell; do not reuse an old conda):
   ```powershell
   winget install Python.Python.3.12 --scope user
   git clone https://github.com/latorc/MahjongCopilot $env:USERPROFILE\MahjongCopilot
   cd $env:USERPROFILE\MahjongCopilot; python -m venv venv; .\venv\Scripts\pip install -r requirements.txt; .\venv\Scripts\playwright install chromium
   ```
3. **Apply the three Windows patches + install our bot plugin** (patch verified against MC `31be3de`):
   ```powershell
   git apply <repo>\tools\majsoul_bridge\mahjongcopilot_windows.patch
   python <repo>\tools\majsoul_bridge\install.py $env:USERPROFILE\MahjongCopilot
   ```
   The three patches fix the three Windows blockers you *will* hit: Playwright's bundled Chromium failing
   with a side-by-side error (use the system Chrome instead), Majsoul's 46 MB wasm being buffered by
   mitmproxy into a black screen (stream large bodies), and the mitm root cert needing admin rights
   (install into the current-user store).
4. **Open the tunnel** (play machine, keep it running): `ssh -N -L 8765:127.0.0.1:8765 <model-machine>`;
   MC's URL stays `http://127.0.0.1:8765`.
5. **Configure and start MC**: in `settings.json` set `"model_type": "LLM_Mahjong"`,
   `"llmmahjong_url": "http://127.0.0.1:8765"`, `"ai_randomize_choice": 0`; `enable_automation` = `false`
   for assist mode (you click, the panel shows the policy distribution and V) or `true` for auto-play
   (used for scored runs). Start with
   `cd $env:USERPROFILE\MahjongCopilot; .\venv\Scripts\python.exe main.py`, then launch the browser and log in.
6. **Acceptance + scoring**: play one friend-room game first and check that every `Bot in: tsumo` in MC's
   log has a matching `Bot out: dahai` and that `no op list` count is 0; afterwards run
   `python scripts/analyze_majsoul_session.py <session>.jsonl` on the model machine for placement /
   win / deal-in / riichi / call rates.

**Detailed runbooks**: [tools/majsoul_bridge/README.md](tools/majsoul_bridge/README.md) (general flow, the
two modes, per-decision record format, protocol pitfalls) ·
[tools/majsoul_bridge/WINDOWS.md](tools/majsoul_bridge/WINDOWS.md) (Windows field notes: symptoms and root
cause of each patch, new Unity client compatibility, troubleshooting table for `spawn UNKNOWN` / black
screen / certificates) · [docs/champion_model.md](docs/champion_model.md) (which checkpoint, what resources).

> **Risk**: using third-party automation violates Majsoul's terms of service and **can get the account
> banned** — only use an account you can afford to lose.

**Discipline** (enforced by CLAUDE.md): write `EXPERIMENT.md` (purpose / method / success criteria)
BEFORE launching any run; verify throughput matches expectations right after launch; every long-running
task gets a heartbeat; delete VMs when done; reward logic goes through the registry; append new lessons
to SKILLS.md.
