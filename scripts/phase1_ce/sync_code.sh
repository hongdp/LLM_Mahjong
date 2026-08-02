#!/bin/bash
# Syncs the working tree + SFT data + .env to the VM, then verifies the SFT
# dataset sha256 on the remote side (hard requirement from the handoff doc).
# Relies on the ssh Host alias written by start_vm.sh (gcloud compute config-ssh).
set -euo pipefail

# Overridable for multi-VM experiments: VM_NAME=... ZONE=... bash sync_code.sh
PROJECT_ID="workstation-185016"
ZONE="${ZONE:-us-central1-b}"
VM_NAME="${VM_NAME:-mahjong-a100}"
SSH_HOST="$VM_NAME.$ZONE.$PROJECT_ID"
REMOTE_DIR="LLM_Mahjong"
SFT_SHA256="b3eefd6d144e662b6ed4239cfbdb62197a2c4a941264ae360ab5a250615becf6"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Local pre-flight: never ship a dataset that doesn't match the handoff hash.
echo "Verifying local data/sft_mahjong.jsonl sha256..."
echo "$SFT_SHA256  data/sft_mahjong.jsonl" | sha256sum -c -

echo "Syncing code + data to $SSH_HOST:~/$REMOTE_DIR ..."
rsync -avz --delete \
    --exclude='.git/' \
    --exclude='__pycache__/' \
    --exclude='experiments/' \
    --exclude='checkpoints/' \
    --exclude='logs/' \
    --exclude='.antigravitycli/' \
    ./ "$SSH_HOST:$REMOTE_DIR/"

echo "Verifying remote sha256..."
ssh "$SSH_HOST" "echo '$SFT_SHA256  $REMOTE_DIR/data/sft_mahjong.jsonl' | sha256sum -c -"

echo "Sync complete and dataset hash verified."
