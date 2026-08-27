#!/bin/bash
set -uo pipefail
cd /home/hongdp/Workspace/LLM_Mahjong
for ARCH in cnn_m_r cnn_xl_r convformer_m_r handset_xl_cnn_m_r hrf_xl_v4; do
  echo "=== ARM $ARCH start $(date) ==="
  conda run --no-capture-output -n rlhf_mahjong python scripts/train_human_bc.py     --arch $ARCH --limit_games 2000 --holdout_games 400 --epochs 4     --batch 1024 --lr 3e-4 --workers 10 --seed 0 --out experiments/exp45_bc10_20260827_002245     || echo "!!! ARM $ARCH FAILED"
done
echo "=== ALL ARMS DONE $(date) ==="
