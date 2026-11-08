#!/bin/bash
cd /home/hongdp/Workspace/LLM_Mahjong/.claude/worktrees/rl-model-defense-learning-8fb5c0
export PROBE_WORKERS=8
/home/hongdp/miniconda3/envs/rlhf_mahjong/bin/python -W ignore experiments/kokushi_opening_probe_20261107_175945/kokushi_probe.py play /home/hongdp/Workspace/LLM_Mahjong/experiments/probes/kokushi_probe/seeds.json /home/hongdp/Workspace/LLM_Mahjong/experiments/probes/kokushi_probe/results.json \
  A_192M=/home/hongdp/Workspace/LLM_Mahjong/experiments/exp124_A_p2/games_192000000.pt@0.0 bc49=/home/hongdp/Workspace/LLM_Mahjong/experiments/_anchors_epoch6/bc49.pt@0.0 \
  A_192M_nokyu=/home/hongdp/Workspace/LLM_Mahjong/experiments/exp124_A_p2/games_192000000.pt@0.0@nokyu bc49_nokyu=/home/hongdp/Workspace/LLM_Mahjong/experiments/_anchors_epoch6/bc49.pt@0.0@nokyu \
  A_192M_T1=/home/hongdp/Workspace/LLM_Mahjong/experiments/exp124_A_p2/games_192000000.pt@1.0 A_192M_T1_nokyu=/home/hongdp/Workspace/LLM_Mahjong/experiments/exp124_A_p2/games_192000000.pt@1.0@nokyu \
  CNN_392M=/home/hongdp/Workspace/LLM_Mahjong/experiments/exp101_L_p2/games_392000000.pt@0.0 CNN_392M_nokyu=/home/hongdp/Workspace/LLM_Mahjong/experiments/exp101_L_p2/games_392000000.pt@0.0@nokyu \
  A_32M=/home/hongdp/Workspace/LLM_Mahjong/experiments/exp124_A/games_32000000.pt@0.0 A_96M=/home/hongdp/Workspace/LLM_Mahjong/experiments/exp124_A_p2/games_96000000.pt@0.0 >> experiments/kokushi_opening_probe_20261107_175945/run.log 2>&1
echo ALL_DONE >> experiments/kokushi_opening_probe_20261107_175945/run.log
