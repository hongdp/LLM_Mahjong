#!/bin/bash
# Runs a small-scale local dry-run of the RLHF training loop.
# This script is used to ensure the model compiles and trains locally without OOM.

echo "🚀 Starting Phase 0: Local Sanity Check..."

# Ensure we are in the root directory
cd "$(dirname "$0")/../../"

# Initialize conda and activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate rlhf_mahjong

# Optional: set environment variables for local testing
export CUDA_VISIBLE_DEVICES=0 # Use first local GPU if available, or fallback to CPU
export WANDB_MODE=disabled    # Disable wandb sync during local tests
export PYTHONPATH=$(pwd)      # Ensure python can find the src module

echo "Loading tiny proxy dataset and model..."
# We pass arguments to core trainer to run in "debug" mode
python -m src.core.trainer --task mahjong --model_name gpt2 --debug --epochs 1 --batch_size 2 --num_episodes 2

TEST_EXIT_CODE=$?

if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "✅ Local test passed! Forward and backward passes succeeded without crashing."
    echo "You are ready to deploy to GCP for Phase 1."
else
    echo "❌ Local test failed with exit code $TEST_EXIT_CODE."
    echo "Please check the stack trace above and fix the code before running on GCP."
fi
