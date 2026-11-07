#!/bin/bash
cd /home/hongdp/Workspace/LLM_Mahjong/.claude/worktrees/rl-model-defense-learning-8fb5c0
export RAYON_NUM_THREADS=4 OMP_NUM_THREADS=2
/home/hongdp/miniconda3/envs/rlhf_mahjong/bin/python -W ignore experiments/yaku_census_20261107_134328/yaku_census.py /home/hongdp/Workspace/LLM_Mahjong/experiments/probes/yaku_census/models.json 200000 \
  A_192M=/home/hongdp/Workspace/LLM_Mahjong/experiments/exp124_A_p2/games_192000000.pt A_192M_T0=/home/hongdp/Workspace/LLM_Mahjong/experiments/exp124_A_p2/games_192000000.pt@0.0 bc49=/home/hongdp/Workspace/LLM_Mahjong/experiments/_anchors_epoch6/bc49.pt \
  CNN_392M=/home/hongdp/Workspace/LLM_Mahjong/experiments/exp101_L_p2/games_392000000.pt A_144M=/home/hongdp/Workspace/LLM_Mahjong/experiments/exp124_A_p2/games_144000000.pt A_96M=/home/hongdp/Workspace/LLM_Mahjong/experiments/exp124_A_p2/games_96000000.pt \
  A_64M=/home/hongdp/Workspace/LLM_Mahjong/experiments/exp124_A/games_64000000.pt A_32M=/home/hongdp/Workspace/LLM_Mahjong/experiments/exp124_A/games_32000000.pt A_8M=/home/hongdp/Workspace/LLM_Mahjong/experiments/exp124_A/games_8000000.pt >> experiments/yaku_census_20261107_134328/run.log 2>&1
echo ALL_DONE >> experiments/yaku_census_20261107_134328/run.log
