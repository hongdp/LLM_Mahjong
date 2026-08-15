#!/bin/bash
# Generic DNN experiment runner ON the VM. $1 = experiment name (GCS dir),
# rest = the full python command. Bootstraps a light env (torch/numpy/tb),
# syncs the exp dir to GCS every 10 min, uploads on exit, powers off.
set -uo pipefail
GCS=gs://llm-mahjong-experiments
NAME="$1"; shift
LOG="$HOME/${NAME}_nohup.log"

trap 'echo "[exit] salvaging...";
      gsutil -m -q rsync -r "$HOME/LLM_Mahjong/experiments/$NAME" "$GCS/$NAME/" 2>/dev/null;
      gsutil -q cp "$LOG" "$GCS/$NAME/" 2>/dev/null;
      sudo shutdown -h now' EXIT

cd "$HOME/LLM_Mahjong"
export PYTHONUNBUFFERED=1 PYTHONPATH="$HOME/LLM_Mahjong"
VENV="$HOME/venvs/dnn"
for i in $(seq 1 60); do nvidia-smi &>/dev/null && break; sleep 20; done
if [ ! -f "$VENV/bin/activate" ]; then
    sudo apt-get update -qq && sudo apt-get install -y -qq python3-venv
    python3 -m venv "$VENV"; source "$VENV/bin/activate"
    pip install -q --no-cache-dir --upgrade pip
    pip install -q --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu129 || \
    pip install -q --no-cache-dir torch
    pip install -q --no-cache-dir numpy tensorboard
else source "$VENV/bin/activate"; fi
python -c "import torch; assert torch.cuda.is_available()" || echo "[warn] no CUDA, updates on CPU"

( while true; do sleep 600
    gsutil -m -q rsync -r "experiments/$NAME" "$GCS/$NAME/" 2>/dev/null
    gsutil -q cp "$LOG" "$GCS/$NAME/" 2>/dev/null
  done ) & SYNC=$!

echo "[run] $*"
"$@"
RC=$?
kill $SYNC 2>/dev/null
echo "[run] exit=$RC"
