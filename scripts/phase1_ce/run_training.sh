#!/bin/bash
# Runs the Phase 1 training INSIDE the VM, uploads all results to GCS, then
# shuts the VM down to stop billing. Upload + shutdown happen regardless of
# whether training succeeded, so nothing is lost and nothing keeps billing.
# Launch it detached from the local machine, e.g.:
#   ssh mahjong-a100.us-central1-b.workstation-185016 \
#     'nohup bash LLM_Mahjong/scripts/phase1_ce/run_training.sh > train_nohup.log 2>&1 & disown'
set -uo pipefail

REPO="$HOME/LLM_Mahjong"
CONFIG="${1:-configs/v2_full_run.json}"   # pass a config path as $1 to override
export PYTHONUNBUFFERED=1   # stream prints to the nohup log in real time
VENV="$HOME/venvs/rlhf"
GCS_BUCKET="gs://llm-mahjong-experiments"

# Belt-and-braces: whatever path exits this script (bootstrap failure, crash,
# normal completion), the VM powers off so it never idles on the meter.
# On flex-start VMs poweroff => DELETE, so salvage the log to GCS first —
# a bootstrap failure would otherwise vanish without a trace.
trap 'echo "[shutdown] EXIT trap — salvaging log, powering off";
      gsutil cp "$HOME/train_nohup.log" "$GCS_BUCKET/orphan_logs/$(hostname)_train_nohup.log" 2>/dev/null || true;
      sudo shutdown -h now' EXIT

cd "$REPO"

# --- wait for the NVIDIA driver (first boot installs it asynchronously) ---
for i in $(seq 1 60); do
    nvidia-smi &>/dev/null && break
    echo "[bootstrap] waiting for NVIDIA driver ($i/60)..."; sleep 20
done
nvidia-smi &>/dev/null || { echo "[bootstrap] FATAL: driver never came up"; exit 1; }

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
# Fast-path kernel: prebuilt causal-conv1d (matches torch 2.12.1+cu129, py3.10);
# untar into site-packages — avoids a ~10min source build on every ephemeral VM.
if ! python -c "import causal_conv1d" 2>/dev/null; then
    SITE=$(python -c "import site; print(site.getsitepackages()[0])")
    gsutil cp "$GCS_BUCKET/_infra/cc1d_pkg_torch2121cu129_py310.tgz" /tmp/cc1d.tgz \
        && tar xzf /tmp/cc1d.tgz -C "$SITE" && rm /tmp/cc1d.tgz
    python -c "import causal_conv1d; print('causal-conv1d OK')" || echo "[bootstrap] WARN: cc1d unavailable, falling back to eager path"
fi
# SFT anchor: RL-only configs load a frozen adapter; pull it from GCS if absent.
ANCHOR="experiments/v2_engine_full_run_20260802_005918/checkpoints_sft_warmup_mahjong"
if [ ! -f "$ANCHOR/adapter_model.safetensors" ]; then
    echo "[bootstrap] pulling SFT anchor from GCS..."
    mkdir -p "$ANCHOR"
    gsutil -m rsync -r "$GCS_BUCKET/v2_engine_full_run_20260802_005918/checkpoints_sft_warmup_mahjong" "$ANCHOR" || exit 1
fi
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('GPU:', torch.cuda.get_device_name(0))" || exit 1

# --- provenance for the experiment record --------------------------------
nvidia-smi > "$HOME/gpu_info.txt" 2>&1 || true
pip freeze > "$HOME/pip_freeze.txt" 2>&1 || true

# --- periodic incremental sync: bounds data loss to ~10 min if the VM is
# --- terminated by GCE (host event / max-run-duration) before the EXIT path
( while true; do
      sleep 600
      L=$(ls -td experiments/*/ 2>/dev/null | head -1)
      [ -n "$L" ] && gsutil -m -q rsync -r "$L" "$GCS_BUCKET/$(basename "$L")/" 2>/dev/null
      gsutil -q cp "$HOME/train_nohup.log" "$GCS_BUCKET/$(basename "${L:-orphan}")/" 2>/dev/null
  done ) &
SYNC_PID=$!

# --- training -------------------------------------------------------------
echo "[train] starting: python -m src.core.trainer --config $CONFIG"
python -m src.core.trainer --config "$CONFIG"
TRAIN_EXIT_CODE=$?
kill "$SYNC_PID" 2>/dev/null
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
