#!/bin/bash
# chained: waits for the five-arm batch, then trains the mortal46 arm
cd /home/hongdp/Workspace/LLM_Mahjong
until grep -q "ALL ARMS DONE" experiments/exp45_bc10_20260827_002245/run.log 2>/dev/null; do sleep 60; done
echo "=== ARM mortal_full_xl_pure_m46 start $(date) ==="
conda run --no-capture-output -n rlhf_mahjong python scripts/train_human_bc.py   --arch mortal_full_xl_pure_m46 --limit_games 2000 --holdout_games 400   --epochs 4 --batch 1024 --lr 3e-4 --workers 10 --seed 0 --out experiments/exp45_bc10_20260827_002245   || echo "!!! ARM mortal_full_xl_pure_m46 FAILED"
echo "=== MORTAL ARM DONE $(date) ==="
