#!/bin/bash
# 11-07 rerun with the official default tactics (jun_pt 90/30/-30/-90): the first round's jun_pt=0 zeroed akochan's objective.
# Waits for every other local job (CPU: 6 akochan processes x 4 threads), then 2000 walls in 6 shards; fallbacks dumped.
cd /home/hongdp/Workspace/LLM_Mahjong/.claude/worktrees/rl-model-defense-learning-8fb5c0
E=experiments
for f in $E/placement_fold_1v3_20261106_230858/run.log $E/mortal_fresh_walls_20261107_020638/run.log $E/table4m_192M_run.log \
         $E/concepts_exp127_A1A2_20261107_043435/run.log $E/ukeire_deviation_probe_20261107_181238/run.log; do
  until grep -q ALL_DONE $f 2>/dev/null; do sleep 120; done
done
PY=/home/hongdp/miniconda3/envs/rlhf_mahjong/bin/python
T=akochan:/home/hongdp/Workspace/LLM_Mahjong/.claude/worktrees/rl-model-defense-learning-8fb5c0/experiments/akochan_dup_20261106_200951/tactics_default.json
P=/home/hongdp/Workspace/LLM_Mahjong/experiments/probes/akochan_dup2
b=(0 300 640 980 1320 1660 2000)
for i in 0 1 2 3 4 5; do
  AKO_THREADS=4 AKO_FALLBACK_LOG=$P/fallbacks_shard$i.jsonl $PY -W ignore experiments/akochan_dup_20261106_200951/ako_dup_driver.py $T ${b[$i]} ${b[$((i+1))]} 101112000 $P/default_shard$i.json > experiments/akochan_dup_20261106_200951/default_shard$i.log 2>&1 &
done
wait
echo ALL_DONE
