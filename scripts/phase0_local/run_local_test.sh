#!/bin/bash
# Runs a small-scale local dry-run of the RLHF training loop using a local model with QLoRA.
# This script ensures the model compiles and trains locally without OOM on a 16GB VRAM GPU.

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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Set the model name (Using Qwen 0.5B locally to fit in 16GB VRAM, will use Gemma on GCP)
MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"

echo "Loading $MODEL_NAME with QLoRA (4-bit)..."
# SFT warm-up from pre-generated data, then full strategy RL in one run.
if [ -n "$1" ]; then
    echo "Running with config file: $1"
    python -m src.core.trainer --config "$1"
else
    python -m src.core.trainer \
        --model_name $MODEL_NAME \
        --task mahjong \
        --use_qlora \
        --batch_size 1 \
        --num_episodes 1 \
        --epochs 1 \
        --learning_rate 1e-6 \
        --training_phase 2 \
        --sft_epochs 1 \
        --sft_batch_size 2 \
        --sft_data data/sft_mahjong.jsonl
fi

TEST_EXIT_CODE=$?

if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "✅ Local test passed! SFT warm-up + strategy RL completed without OOM."
    echo "You are ready to scale up the batch size or deploy to GCP."
else
    echo "❌ Local test failed with exit code $TEST_EXIT_CODE."
    echo "Please check the stack trace above."
fi
