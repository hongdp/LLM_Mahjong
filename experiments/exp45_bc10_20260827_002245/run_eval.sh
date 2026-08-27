#!/bin/bash
# exp45 H2/H3 stage: Elo + defense_iq for all six BC checkpoints.
# Waits for the training batch, then evaluates on the fixed-engine branch
# (ratings tagged --allow_engine_mismatch; delta vs epoch-5 anchors is the
# two action-gap fixes, ~1e-5 of decisions).
set -uo pipefail
cd /home/hongdp/Workspace/LLM_Mahjong
DIR=experiments/exp45_bc10_20260827_002245
until grep -q "ALL ARMS DONE" $DIR/run.log 2>/dev/null; do sleep 120; done
CKPTS=""
for ARCH in cnn_m_r cnn_xl_r convformer_m_r handset_xl_cnn_m_r hrf_xl_v4 mortal_full_xl_pure_m46; do
  CK=$DIR/bc_${ARCH}_best.pt
  [ -f "$CK" ] || { echo "!!! missing $CK"; continue; }
  echo "=== ELO $ARCH start $(date) ==="
  conda run --no-capture-output -n rlhf_mahjong python scripts/run_elo_league.py rate \
    --ckpt $CK --label bc45_${ARCH}_T1 --deals 100 --seed0 45100001 \
    --allow_engine_mismatch || echo "!!! ELO $ARCH FAILED"
  CKPTS="$CKPTS $CK"
done
echo "=== DEFENSE PROBE start $(date) ==="
conda run --no-capture-output -n rlhf_mahjong python scripts/probe_defense.py \
  --ckpt $CKPTS --games 800 --seed0 45200001 --out $DIR/defense_iq.json \
  || echo "!!! DEFENSE PROBE FAILED"
echo "=== EVAL ALL DONE $(date) ==="
