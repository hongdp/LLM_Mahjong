#!/bin/bash
# Runs a small-scale local dry-run of the RLHF training loop using a Gemma model with QLoRA.
# This script ensures the model compiles and trains locally without OOM on a 16GB VRAM GPU.

echo "🚀 Starting Phase 0: Local Sanity Check with Gemma..."

# Ensure we are in the root directory
cd "$(dirname "$0")/../../"

# Initialize conda and activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate rlhf_mahjong

# Optional: set environment variables for local testing
export CUDA_VISIBLE_DEVICES=0 # Use first local GPU if available, or fallback to CPU
export WANDB_MODE=disabled    # Disable wandb sync during local tests
export PYTHONPATH=$(pwd)      # Ensure python can find the src module

# Set the model name (Change this to your specific Gemma 4 variant)
MODEL_NAME="google/gemma-2-2b-it"

echo "Loading $MODEL_NAME with QLoRA (4-bit)..."
# We pass arguments to train_rlhf.py to run in "debug" mode (e.g. fewer steps, tiny batch size)
# and importantly, we enable --use_qlora to save VRAM.
python src/train_rlhf.py \
    --model_name $MODEL_NAME \
    --use_qlora \
    --batch_size 1 \
    --mini_batch_size 1 \
    --max_steps 5 \
    --debug

TEST_EXIT_CODE=$?

if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "✅ Local test passed! Forward and backward passes succeeded without OOM."
    echo "You are ready to scale up the batch size or deploy to GCP."
else
    echo "❌ Local test failed with exit code $TEST_EXIT_CODE."
    echo "Please check the stack trace above."
fi
