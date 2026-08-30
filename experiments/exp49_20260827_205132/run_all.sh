#!/bin/bash
set -uo pipefail
cd /home/hongdp/Workspace/LLM_Mahjong
DIR=experiments/exp49_20260827_205132
T() { conda run --no-capture-output -n rlhf_mahjong python scripts/train_human_bc.py   --limit_games "$1" --holdout_games "$2" --max_epochs 30 --patience 3 --min_delta 0.0005   --batch 1024 --workers 10 --seed 0 "${@:3}"; }
echo "=== A1 lr1e-4 start $(date) ==="
T 2000 400 --arch mortal_bb_xl_v3r_m46 --lr 1e-4 --out $DIR/A1 || echo "!!! A1 FAILED"
echo "=== A2 cosine start $(date) ==="
T 2000 400 --arch mortal_bb_xl_v3r_m46 --lr 3e-4 --lr_schedule cosine --out $DIR/A2 || echo "!!! A2 FAILED"
# ---- automatic pick per prereg rule
PICK=$(conda run --no-capture-output -n rlhf_mahjong python - <<'PY'
import json
base=0.7553
def best(p):
    try: return max(r['acc'] for r in json.load(open(p)))
    except Exception: return 0.0
a1=best("experiments/exp49_20260827_205132/A1/bc_mortal_bb_xl_v3r_m46_metrics.json")
a2=best("experiments/exp49_20260827_205132/A2/bc_mortal_bb_xl_v3r_m46_metrics.json")
print(f"a1={a1:.4f} a2={a2:.4f} base={base:.4f}", end=" ")
if max(a1,a2) < base+0.003: print("PICK:const3e4")
elif a2>=a1: print("PICK:cos3e4")
else: print("PICK:const1e4")
PY
)
echo "A-part verdict: $PICK"
SEFLAGS="--lr 3e-4"; CONVFLAGS="--lr 3e-4"
case "$PICK" in
  *PICK:cos3e4)  SEFLAGS="--lr 3e-4 --lr_schedule cosine"; CONVFLAGS="--lr 3e-4 --lr_schedule cosine";;
  *PICK:const1e4) SEFLAGS="--lr 1e-4";;
esac
echo "=== B 100% start $(date) SEFLAGS=$SEFLAGS CONVFLAGS=$CONVFLAGS ==="
T 0 1000 --arch convformer_m_r $CONVFLAGS --out $DIR/B || echo "!!! B conv FAILED"
T 0 1000 --arch convformer_m_v3r_m46 $CONVFLAGS --out $DIR/B || echo "!!! B conv46 FAILED"
T 0 1000 --arch mortal_bb_xl_v3r_m46 $SEFLAGS --out $DIR/B || echo "!!! B seres FAILED"
echo "=== B ELO ==="
for A in convformer_m_r convformer_m_v3r_m46 mortal_bb_xl_v3r_m46; do
  conda run --no-capture-output -n rlhf_mahjong python scripts/run_elo_league.py rate     --ckpt $DIR/B/bc_${A}_best.pt --label bc49_${A}_full_T1 --deals 300     --seed0 45400001 --allow_engine_mismatch || echo "!!! ELO $A FAILED"
done
conda run --no-capture-output -n rlhf_mahjong python scripts/probe_defense.py   --ckpt conv=$DIR/B/bc_convformer_m_r_best.pt conv46=$DIR/B/bc_convformer_m_v3r_m46_best.pt seres46=$DIR/B/bc_mortal_bb_xl_v3r_m46_best.pt   --games 800 --seed0 45500001 --out $DIR/B/defense_iq.json || echo "!!! DEFENSE FAILED"
echo "=== EXP49 ALL DONE $(date) ==="
