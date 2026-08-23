#!/bin/bash
# g2-standard-32 (32 vCPU + 1x L4) for DNN self-play experiments.
# PROVISIONING=spot (default): preemptible, termination action STOP —
#   incremental GCS sync + resume make preemption a pause, not a loss.
# PROVISIONING=ondemand: for days when spot is being harvested zone-wide
#   (2026-08-15: three double-preemptions across -a AND -b in 2.5h).
# Usage: VM_NAME=dnn-x ZONE=us-central1-b PROVISIONING=ondemand bash start_g2.sh
set -euo pipefail
PROJECT_ID="workstation-185016"
ZONE="${ZONE:-us-central1-a}"
VM_NAME="${VM_NAME:-mahjong-dnn}"
PROVISIONING="${PROVISIONING:-spot}"
PROV_FLAGS=()
if [ "$PROVISIONING" = "spot" ]; then
    PROV_FLAGS=(--provisioning-model=SPOT --instance-termination-action=STOP)
fi
gcloud compute instances create "$VM_NAME" \
    --project="$PROJECT_ID" --zone="$ZONE" \
    --machine-type=g2-standard-32 \
    --image-family=common-cu129-ubuntu-2204-nvidia-580 \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=100GB --boot-disk-type=pd-balanced \
    --maintenance-policy=TERMINATE \
    "${PROV_FLAGS[@]}" \
    --scopes=storage-rw,logging-write \
    --metadata="install-nvidia-driver=True"
gcloud compute config-ssh --project="$PROJECT_ID" > /dev/null
echo "up: $VM_NAME.$ZONE.$PROJECT_ID"
