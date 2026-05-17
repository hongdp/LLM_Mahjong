#!/bin/bash
# Runs the RLHF training and then automatically shuts down the VM.
# IMPORTANT: This script is intended to be run INSIDE the VM.

echo "Starting training..."
# Activate environment if necessary
# source ~/miniconda3/bin/activate rlhf_env

# Install dependencies (if not already installed)
pip install -r ~/gcloud_llm/requirements.txt

# Run the training script
python ~/gcloud_llm/src/train_rlhf.py

TRAIN_EXIT_CODE=$?

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully."
else
    echo "Training failed with exit code $TRAIN_EXIT_CODE."
    # Optionally: send a slack/email notification here
fi

echo "Shutting down VM to save costs..."
sudo shutdown -h now
