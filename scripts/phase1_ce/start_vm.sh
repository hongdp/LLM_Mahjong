#!/bin/bash
# Starts the GCP Compute Engine VM.

PROJECT_ID="your-gcp-project-id"
ZONE="us-central1-a"
VM_NAME="rlhf-gpu-vm"

echo "Starting VM: $VM_NAME in zone $ZONE..."
gcloud compute instances start $VM_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE

echo "VM Started!"
echo "To SSH into the VM, run:"
echo "gcloud compute ssh $VM_NAME --project=$PROJECT_ID --zone=$ZONE"
