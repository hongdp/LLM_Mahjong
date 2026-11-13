#!/bin/bash
cd /home/hongdp/Workspace/LLM_Mahjong/.claude/worktrees/rl-model-defense-learning-8fb5c0
until grep -q ALL_DONE experiments/placement_fold_1v3_20261106_230858/run.log 2>/dev/null; do sleep 120; done
PY=/home/hongdp/miniconda3/envs/rlhf_mahjong/bin/python
M=mortal:/home/hongdp/Workspace/LLM_Mahjong/data/mortal_ext/mortal_298k.pth
for i in $(seq 0 11); do w0=$((i*34)); w1=$((w0+34)); [ $w1 -gt 400 ] && w1=400; [ $w0 -ge 400 ] && continue
  $PY -W ignore experiments/table_rates_20261106_082735/table_flows.py $w0 $w1 91000000 /home/hongdp/Workspace/LLM_Mahjong/experiments/probes/table_rates/R5_shard$i.json rot mortal=$M bc49=/home/hongdp/Workspace/LLM_Mahjong/experiments/_anchors_epoch6/bc49.pt exp124A_192M=/home/hongdp/Workspace/LLM_Mahjong/experiments/exp124_A_p2/games_192000000.pt exp113C_384M=/home/hongdp/Workspace/LLM_Mahjong/experiments/exp113_C/games_384000000.pt > experiments/table4m_192M_20261107_013503/shard$i.log 2>&1 &
done; wait
$PY experiments/table_rates_20261106_082735/summarize.py "/home/hongdp/Workspace/LLM_Mahjong/experiments/probes/table_rates/R5_shard*.json" /home/hongdp/Workspace/LLM_Mahjong/experiments/probes/table_rates/R5_rates.json > experiments/table4m_192M_20261107_013503/R5_rates_raw.txt 2>&1
echo ALL_DONE
