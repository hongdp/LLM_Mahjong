#!/bin/bash
set -uo pipefail
cd /home/hongdp/Workspace/LLM_Mahjong
echo "=== v3r2 BC full start $(date) ==="
conda run --no-capture-output -n rlhf_mahjong python scripts/train_human_bc.py   --arch convformer_m_v3r2_m46 --limit_games 0 --holdout_games 1000   --max_epochs 30 --patience 3 --min_delta 0.0005   --batch 1024 --lr 3e-4 --workers 12 --seed 0 --out experiments/exp51_20260828_203354   || echo "!!! TRAIN FAILED"
echo "=== ELO r300 ==="
conda run --no-capture-output -n rlhf_mahjong python scripts/run_elo_league.py rate   --ckpt experiments/exp51_20260828_203354/bc_convformer_m_v3r2_m46_best.pt --label bc51_v3r2_full_T1   --deals 300 --seed0 46000001 --allow_engine_mismatch || echo "!!! ELO FAILED"
echo "=== DEFENSE ==="
conda run --no-capture-output -n rlhf_mahjong python scripts/probe_defense.py   --ckpt v3r2=experiments/exp51_20260828_203354/bc_convformer_m_v3r2_m46_best.pt   --games 800 --seed0 46100001 --out experiments/exp51_20260828_203354/defense_iq.json || echo "!!! DEFENSE FAILED"
echo "=== EXP51 ALL DONE $(date) ==="
