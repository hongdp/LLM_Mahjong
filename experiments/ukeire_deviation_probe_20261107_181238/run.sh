#!/bin/bash
cd /home/hongdp/Workspace/LLM_Mahjong/.claude/worktrees/rl-model-defense-learning-8fb5c0
until grep -q ALL_DONE experiments/kokushi_opening_probe_20261107_175945/run.log 2>/dev/null; do sleep 60; done
PY=/home/hongdp/miniconda3/envs/rlhf_mahjong/bin/python
N=10500; per=$(( (N+11)/12 ))
for var in none UK2 UK1; do
  for i in $(seq 0 11); do w0=$((i*per)); w1=$((w0+per)); [ $w1 -gt $N ] && w1=$N
    $PY -W ignore experiments/ukeire_deviation_probe_20261107_181238/uk_driver.py $var $w0 $w1 101112000 /home/hongdp/Workspace/LLM_Mahjong/experiments/probes/uk_dev/${var}_shard$i.json > experiments/ukeire_deviation_probe_20261107_181238/${var}_shard$i.log 2>&1 &
  done; wait; echo "$var done $(date +%H:%M)" >> experiments/ukeire_deviation_probe_20261107_181238/run.log
done
echo ALL_DONE >> experiments/ukeire_deviation_probe_20261107_181238/run.log
