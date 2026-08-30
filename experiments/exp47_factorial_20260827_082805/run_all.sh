#!/bin/bash
set -uo pipefail
cd /home/hongdp/Workspace/LLM_Mahjong
for ARCH in mortal_bb_xl_r convformer_m_v3r convformer_m_r_m46 convformer_m_v3r_m46 mortal_bb_xl_v3r_m46; do
  echo "=== ARM $ARCH start $(date) ==="
  conda run --no-capture-output -n rlhf_mahjong python scripts/train_human_bc.py     --arch $ARCH --limit_games 2000 --holdout_games 400     --max_epochs 30 --patience 3 --min_delta 0.0005     --batch 1024 --lr 3e-4 --workers 10 --seed 0 --out experiments/exp47_factorial_20260827_082805     || { echo "!!! ARM $ARCH FAILED"; continue; }
  echo "=== ELO $ARCH ==="
  conda run --no-capture-output -n rlhf_mahjong python scripts/run_elo_league.py rate     --ckpt experiments/exp47_factorial_20260827_082805/bc_${ARCH}_best.pt --label bc47_${ARCH}_T1 --deals 100     --seed0 45100001 --allow_engine_mismatch || echo "!!! ELO $ARCH FAILED"
done
echo "=== DEFENSE PROBE ==="
conda run --no-capture-output -n rlhf_mahjong python scripts/probe_defense.py   --ckpt bb_xl=experiments/exp47_factorial_20260827_082805/bc_mortal_bb_xl_r_best.pt conv_v3r=experiments/exp47_factorial_20260827_082805/bc_convformer_m_v3r_best.pt conv_m46=experiments/exp47_factorial_20260827_082805/bc_convformer_m_r_m46_best.pt conv_v3r_m46=experiments/exp47_factorial_20260827_082805/bc_convformer_m_v3r_m46_best.pt bb_v3r_m46=experiments/exp47_factorial_20260827_082805/bc_mortal_bb_xl_v3r_m46_best.pt   --games 800 --seed0 45200001 --out experiments/exp47_factorial_20260827_082805/defense_iq.json || echo "!!! DEFENSE FAILED"
echo "=== EXP47 ALL DONE $(date) ==="
