#!/bin/bash
# exp45 100% point: all non-holdout games (~18k), convergence protocol,
# same seeds/lr as the 10% batch. Launched after 10% conclusions land.
set -uo pipefail
cd /home/hongdp/Workspace/LLM_Mahjong
for ARCH in cnn_m_r cnn_xl_r convformer_m_r handset_xl_cnn_m_r hrf_xl_v4 mortal_full_xl_pure_m46; do
  echo "=== ARM $ARCH start $(date) ==="
  conda run --no-capture-output -n rlhf_mahjong python scripts/train_human_bc.py     --arch $ARCH --limit_games 0 --holdout_games 1000     --max_epochs 30 --patience 3 --min_delta 0.0005     --batch 1024 --lr 3e-4 --workers 12 --seed 0 --out experiments/exp45_bc100_20260827     || echo "!!! ARM $ARCH FAILED"
done
echo "=== ALL ARMS DONE $(date) ==="
