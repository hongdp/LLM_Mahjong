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
else source "$VENV/bin/activate"; fi
# idempotent top-up: the FULL third-party closure of the DNN path, computed
# by import tracing (torch, numpy, the riichi scoring lib, tqdm, tb). Two
# VMs self-terminated on missing modules found one at a time — never again.
pip install --no-cache-dir numpy tensorboard "mahjong==2.0.0" tqdm 2>&1 | tail -2
# hard preflight: refuse to start the run unless the FULL closure imports in
# the exact interpreter that will run it. a0 died on this silently once.
echo "[env] python=$(which python) pip=$(which pip)"
if ! python -c "import numpy, torch, tensorboard, mahjong, tqdm"; then
    echo "[fatal] dependency closure incomplete in $(which python)"
    pip list 2>/dev/null | grep -iE "torch|numpy|mahjong|tensorboard|tqdm"
    exit 1
fi
python -c "import torch; assert torch.cuda.is_available()" || echo "[warn] no CUDA, updates on CPU"

# two sync cadences: heavy artifacts (checkpoints) every 10 min; the light
# observability set (tensorboard events, train_log, nohup log) every 2 min so
# the local TB mirror is never more than ~3.5 min behind the run.
( while true; do sleep 600
    gsutil -m -q rsync -r "experiments/$NAME" "$GCS/$NAME/" 2>/dev/null
  done ) & SYNC=$!
( while true; do sleep 120
    gsutil -m -q rsync -r "experiments/$NAME/tensorboard" "$GCS/$NAME/tensorboard" 2>/dev/null
    gsutil -q cp "experiments/$NAME/train_log.json" "$GCS/$NAME/" 2>/dev/null
    gsutil -q cp "$LOG" "$GCS/$NAME/" 2>/dev/null
  done ) & SYNC2=$!

echo "[run] $*"
"$@"
RC=$?
kill $SYNC $SYNC2 2>/dev/null
echo "[run] exit=$RC"
