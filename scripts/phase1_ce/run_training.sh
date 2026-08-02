#!/bin/bash
# Runs the Phase 1 training INSIDE the VM, uploads all results to GCS, then
# shuts the VM down to stop billing. Upload + shutdown happen regardless of
# whether training succeeded, so nothing is lost and nothing keeps billing.
# Launch it detached from the local machine, e.g.:
#   ssh mahjong-a100.us-central1-b.workstation-185016 \
#     'nohup bash LLM_Mahjong/scripts/phase1_ce/run_training.sh > train_nohup.log 2>&1 & disown'
set -uo pipefail

# Belt-and-braces: whatever path exits this script (bootstrap failure, crash,
# normal completion), the VM powers off so it never idles on the meter.
trap 'echo "[shutdown] EXIT trap — powering off"; sudo shutdown -h now' EXIT

REPO="$HOME/LLM_Mahjong"
CONFIG="${1:-configs/v2_full_run.json}"   # pass a config path as $1 to override
export PYTHONUNBUFFERED=1   # stream prints to the nohup log in real time
VENV="$HOME/venvs/rlhf"
GCS_BUCKET="gs://llm-mahjong-experiments"

cd "$REPO"

# --- one-time environment bootstrap (idempotent) -------------------------
if [ ! -f "$VENV/bin/activate" ]; then
    echo "[bootstrap] creating venv (python 3.10, pinned packages)..."
    sudo apt-get update -qq && sudo apt-get install -y -qq python3-venv python3-pip
    python3 -m venv "$VENV"
    source "$VENV/bin/activate"
    pip install --no-cache-dir --upgrade pip
    pip install --no-cache-dir -r scripts/phase1_ce/requirements_pinned.txt || exit 1
else
    source "$VENV/bin/activate"
fi
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('GPU:', torch.cuda.get_device_name(0))" || exit 1

# --- provenance for the experiment record --------------------------------
nvidia-smi > "$HOME/gpu_info.txt" 2>&1 || true
pip freeze > "$HOME/pip_freeze.txt" 2>&1 || true

# --- training -------------------------------------------------------------
echo "[train] starting: python -m src.core.trainer --config $CONFIG"
python -m src.core.trainer --config "$CONFIG"
TRAIN_EXIT_CODE=$?
echo "[train] finished with exit code $TRAIN_EXIT_CODE"

# --- persist results to GCS (success OR failure) ---------------------------
LATEST_EXP=$(ls -td experiments/*/ 2>/dev/null | head -1)
if [ -n "$LATEST_EXP" ]; then
    EXP_NAME=$(basename "$LATEST_EXP")
    cp "$HOME/gpu_info.txt" "$HOME/pip_freeze.txt" "$LATEST_EXP" 2>/dev/null || true
    echo "exit_code=$TRAIN_EXIT_CODE $(date -u +%FT%TZ)" > "${LATEST_EXP}TRAIN_EXIT"
    echo "[gcs] uploading $EXP_NAME to $GCS_BUCKET/$EXP_NAME/ ..."
    gsutil -m rsync -r "$LATEST_EXP" "$GCS_BUCKET/$EXP_NAME/"
    gsutil cp "$HOME/train_nohup.log" "$GCS_BUCKET/$EXP_NAME/" 2>/dev/null || true
    echo "[gcs] upload done."
else
    echo "[gcs] WARNING: no experiment directory found to upload."
    gsutil cp "$HOME/train_nohup.log" "$GCS_BUCKET/orphan_logs/train_nohup_$(date -u +%Y%m%d_%H%M%S).log" 2>/dev/null || true
fi

# --- stop the meter --------------------------------------------------------
echo "[shutdown] powering off to stop billing..."
sudo shutdown -h now
