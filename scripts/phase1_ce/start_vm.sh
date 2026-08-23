#!/bin/bash
# Creates (first run) or starts (subsequent runs) the Phase 1 GPU VM.
# Machine: a2-highgpu-1g = 1x A100 40GB, 12 vCPU, 85GB RAM (fits A2_CPUS=12 quota).
set -euo pipefail

# Overridable for multi-VM experiments: VM_NAME=mahjong-a100-e ZONE=us-east1-b bash start_vm.sh
# PROVISIONING=flex (default per docs/gcp_compute_cost_and_quota.md): DWS flex-start,
#   -45% cost, no preemption within max-run-duration. Termination action STOP
#   (NOT DELETE): a host event mid-run then only stops the VM — disk survives
#   for salvage (2026-08-10 incident: DELETE vaporized a 17h run's disk).
#   Flex VMs cannot restart after stop; delete manually after salvage.
# PROVISIONING=ondemand: legacy standard VM (1 A100/region quota cap).
PROJECT_ID="workstation-185016"
PROVISIONING="${PROVISIONING:-flex}"
ZONE="${ZONE:-us-central1-b}"   # -a was STOCKOUT on 2026-08-01; capacity reported in -b, -f
VM_NAME="${VM_NAME:-mahjong-a100}"
MACHINE_TYPE="a2-highgpu-1g"
IMAGE_FAMILY="common-cu129-ubuntu-2204-nvidia-580"  # driver 580 -> supports cu130 torch wheels
IMAGE_PROJECT="deeplearning-platform-release"
BOOT_DISK_GB="200"

if gcloud compute instances describe "$VM_NAME" --project="$PROJECT_ID" --zone="$ZONE" &>/dev/null; then
    echo "VM $VM_NAME exists — starting it..."
    gcloud compute instances start "$VM_NAME" --project="$PROJECT_ID" --zone="$ZONE"
else
    echo "Creating VM $VM_NAME ($MACHINE_TYPE, 1x A100 40GB) in $ZONE..."
    gcloud compute instances create "$VM_NAME" \
        --project="$PROJECT_ID" \
        --zone="$ZONE" \
        --machine-type="$MACHINE_TYPE" \
        --image-family="$IMAGE_FAMILY" \
        --image-project="$IMAGE_PROJECT" \
        --boot-disk-size="${BOOT_DISK_GB}GB" \
        --boot-disk-type=pd-balanced \
        --maintenance-policy=TERMINATE \
        --scopes=storage-rw,logging-write,monitoring-write,pubsub,trace \
        $( [ "$PROVISIONING" = flex ] && echo "--provisioning-model=FLEX_START --instance-termination-action=STOP --max-run-duration=${MAX_RUN_DURATION:-48h} --request-valid-for-duration=2h" ) \
        --metadata="install-nvidia-driver=True"
fi

# Make plain ssh/rsync work against the VM (writes a Host alias into ~/.ssh/config).
gcloud compute config-ssh --project="$PROJECT_ID" > /dev/null

echo "VM up. SSH alias: $VM_NAME.$ZONE.$PROJECT_ID"
echo "  gcloud compute ssh $VM_NAME --project=$PROJECT_ID --zone=$ZONE"
