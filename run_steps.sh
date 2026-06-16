#!/bin/bash

# HybridSense-Agentic: Master Training Script
# Automates the steps outlined in docs/TRAINING_STEPS.md

set -e # Exit on error

echo "=== HybridSense-Agentic: Starting Training Pipeline Orchestrator ==="

# Step 1: Environment Setup
echo ""
echo "--- Step 1: Checking/Installing Dependencies ---"
# Set Hugging Face mirror to avoid connectivity issues (Errno 99)
export HF_ENDPOINT="https://hf-mirror.com"
pip install torch transformers peft datasets accelerate bitsandbytes gymnasium stable-baselines3 pandas numpy scipy sentencepiece wandb

# Step 2: Configuration Check
echo ""
echo "--- Step 2: Configuration Check ---"
if [ ! -f "configs/pipeline_config.yaml" ]; then
    echo "ERROR: configs/pipeline_config.yaml not found!"
    exit 1
fi
echo "Configuration found. Please ensure wandb is logged in (run 'wandb login' if needed)."

# Step 3: Auto-detect DoRA checkpoint for resume
echo ""
echo "--- Step 3: Checking for Existing Checkpoints ---"

# Read output_dir and ablation_mode from pipeline config to build the expected checkpoint dir
DORA_OUTPUT_DIR=$(python3 -c "
import yaml, sys
with open('configs/pipeline_config.yaml') as f:
    c = yaml.safe_load(f)
d = c['dora_training']
mode = d.get('ablation_mode', 'full')
print(d['output_dir'] + '_' + mode)
")

RESUME_FLAG=""
if [ -d "$DORA_OUTPUT_DIR" ]; then
    # Find the checkpoint subdirectory with the highest step number
    LATEST_CKPT=$(ls -d "${DORA_OUTPUT_DIR}"/checkpoint-* 2>/dev/null \
        | awk -F'checkpoint-' '{print $2, $0}' \
        | sort -n \
        | tail -1 \
        | awk '{print $2}')

    if [ -n "$LATEST_CKPT" ]; then
        echo "Found checkpoint: $LATEST_CKPT"
        echo "Training will resume from this checkpoint."
        RESUME_FLAG="--resume $LATEST_CKPT"
    else
        echo "No checkpoints found in $DORA_OUTPUT_DIR — starting fresh."
    fi
else
    echo "No existing model directory found — starting fresh."
fi

# Steps 4-7: Unified Pipeline Execution
echo ""
echo "--- Steps 3-6: Executing Unified Pipeline ---"
echo "This will handle Data Prep, DoRA Training, PPO Training, and Evaluation."
# shellcheck disable=SC2086
python3 src/pipeline.py --config configs/pipeline_config.yaml $RESUME_FLAG
echo ""
echo "=== All Steps Completed Successfully ==="
