#!/bin/bash
cd /home/hongdp/Workspace/LLM_Mahjong/.claude/worktrees/rl-model-defense-learning-8fb5c0
until grep -q ALL_DONE experiments/akochan_dup_20261106_200951/run.log 2>/dev/null; do sleep 120; done
PY=/home/hongdp/miniconda3/envs/rlhf_mahjong/bin/python
for spec in "none 0" "G 8000" "GS 8000" "G 16000" "GS 16000"; do set -- $spec; tag=$1$(($2/1000))
  for i in $(seq 0 11); do w0=$((i*125)); w1=$((w0+125))
    $PY -W ignore experiments/placement_fold_1v3_20261106_230858/fold1v3_driver.py $1 $2 $w0 $w1 201000000 /home/hongdp/Workspace/LLM_Mahjong/experiments/probes/l6b_1v3/${tag}_shard$i.json > experiments/placement_fold_1v3_20261106_230858/${tag}_shard$i.log 2>&1 &
  done; wait
done
echo ALL_DONE
