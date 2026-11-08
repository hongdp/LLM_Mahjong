#!/bin/bash
cd /home/hongdp/Workspace/LLM_Mahjong/.claude/worktrees/rl-model-defense-learning-8fb5c0
until grep -q ALL_DONE experiments/akochan_dup_20261106_200951/run.log 2>/dev/null; do sleep 120; done
mkdir -p /home/hongdp/Workspace/LLM_Mahjong/experiments/probes/r6_mortal
PY=/home/hongdp/miniconda3/envs/rlhf_mahjong/bin/python
for i in $(seq 0 7); do w0=$((i*1250)); w1=$((w0+1250))
  $PY -W ignore experiments/paper_ref_rows_20261106_062737/dup_driver.py mortal:/home/hongdp/Workspace/LLM_Mahjong/data/mortal_ext/mortal_298k.pth $w0 $w1 101152000 /home/hongdp/Workspace/LLM_Mahjong/experiments/probes/r6_mortal/mortal_shard$i.json > experiments/mortal_fresh_walls_20261107_020638/shard$i.log 2>&1 &
done; wait
$PY experiments/mortal_fresh_walls_20261107_020638/merge.py > experiments/mortal_fresh_walls_20261107_020638/result.txt 2>&1
echo ALL_DONE
