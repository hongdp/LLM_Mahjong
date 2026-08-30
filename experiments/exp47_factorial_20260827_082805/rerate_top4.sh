#!/bin/bash
set -uo pipefail
cd /home/hongdp/Workspace/LLM_Mahjong
rate() { conda run --no-capture-output -n rlhf_mahjong python scripts/run_elo_league.py rate   --ckpt $1 --label $2 --deals 300 --seed0 45300001 --allow_engine_mismatch   || echo "!!! RERATE $2 FAILED"; }
rate experiments/exp47_factorial_20260827_082805/bc_mortal_bb_xl_v3r_m46_best.pt bc47_bb_v3r_m46_r300
rate experiments/exp47_factorial_20260827_082805/bc_convformer_m_v3r_m46_best.pt bc47_conv_v3r_m46_r300
rate experiments/exp45_bc10_20260827_002245/bc_mortal_full_xl_pure_m46_best.pt bc45_mortal46_r300
rate experiments/exp45_bc10_20260827_002245/bc_convformer_m_r_best.pt bc45_conv_r300
echo "=== RERATE DONE $(date) ==="
