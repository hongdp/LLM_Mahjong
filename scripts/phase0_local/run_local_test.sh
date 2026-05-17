#!/bin/bash
# Runs a small-scale local dry-run of the RLHF training loop.
# This script is used to ensure the model compiles and trains locally without OOM.

echo "🚀 Starting Phase 0: Local Sanity Check..."

# Ensure we are in the root directory
cd "$(dirname "$0")/../../"

# Optional: set environment variables for local testing
export CUDA_VISIBLE_DEVICES=0 # Use first local GPU if available, or fallback to CPU
export WANDB_MODE=disabled    # Disable wandb sync during local tests

echo "Loading tiny proxy dataset and model..."
# We pass arguments to train_rlhf.py to run in "debug" mode (e.g. fewer steps, tiny batch size)
python src/train_rlhf.py --debug --max_steps=5 --batch_size=1

TEST_EXIT_CODE=$?

if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "✅ Local test passed! Forward and backward passes succeeded without crashing."
    echo "You are ready to deploy to GCP for Phase 1."
else
    echo "❌ Local test failed with exit code $TEST_EXIT_CODE."
    echo "Please check the stack trace above and fix the code before running on GCP."
fi
