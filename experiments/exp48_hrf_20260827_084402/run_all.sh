#!/bin/bash
set -uo pipefail
cd /home/hongdp/Workspace/LLM_Mahjong
until grep -q "EXP47 ALL DONE" experiments/exp47_factorial_20260827_082805/run.log 2>/dev/null; do sleep 300; done
run_arm() {  # arch label extra_args...
  ARCH=$1; LABEL=$2; shift 2
  echo "=== ARM $LABEL start $(date) ==="
  conda run --no-capture-output -n rlhf_mahjong python scripts/train_human_bc.py     --arch $ARCH --limit_games 2000 --holdout_games 400     --max_epochs 30 --patience 3 --min_delta 0.0005     --batch 1024 --lr 3e-4 --workers 10 --seed 0 --out experiments/exp48_hrf_20260827_084402 "$@"     || { echo "!!! ARM $LABEL FAILED"; return; }
  conda run --no-capture-output -n rlhf_mahjong python scripts/run_elo_league.py rate     --ckpt experiments/exp48_hrf_20260827_084402/bc_${ARCH}_best.pt --label bc48_${LABEL}_T1 --deals 100     --seed0 45100001 --allow_engine_mismatch || echo "!!! ELO $LABEL FAILED"
}
run_arm hrf_xl_v4_m46 hrfA_m46
run_arm hrf_xl_nocross_v4 hrfB_nocross
mv experiments/exp48_hrf_20260827_084402/bc_hrf_xl_v4_best.pt experiments/exp48_hrf_20260827_084402/keep_A 2>/dev/null || true
run_arm hrf_xl_v4 hrfC_rw5 --riichi_weight 5
echo "=== DEFENSE PROBE ==="
conda run --no-capture-output -n rlhf_mahjong python scripts/probe_defense.py   --ckpt hrfA=experiments/exp48_hrf_20260827_084402/bc_hrf_xl_v4_m46_best.pt hrfB=experiments/exp48_hrf_20260827_084402/bc_hrf_xl_nocross_v4_best.pt hrfC=experiments/exp48_hrf_20260827_084402/bc_hrf_xl_v4_best.pt   --games 800 --seed0 45200001 --out experiments/exp48_hrf_20260827_084402/defense_iq.json || echo "!!! DEFENSE FAILED"
echo "=== EXP48 ALL DONE $(date) ==="
