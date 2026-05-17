#!/bin/bash
# Syncs local code to the GCP VM using gcloud compute scp.

PROJECT_ID="your-gcp-project-id"
ZONE="us-central1-a"
VM_NAME="rlhf-gpu-vm"
REMOTE_PATH="~/gcloud_llm"

echo "Syncing code to $VM_NAME..."
gcloud compute scp --recurse ../../src ../../requirements.txt $VM_NAME:$REMOTE_PATH \
    --project=$PROJECT_ID \
    --zone=$ZONE

echo "Sync complete!"
