#!/bin/bash

# HybridSense-Agentic: Master Training Script
# Automates the steps outlined in docs/TRAINING_STEPS.md

set -e # Exit on error

echo "=== HybridSense-Agentic: Starting Training Pipeline Orchestrator ==="

# Step 1: Environment Setup
echo ""
echo "--- Step 1: Checking/Installing Dependencies ---"
pip install torch transformers peft datasets accelerate bitsandbytes gymnasium stable-baselines3 pandas numpy scipy sentencepiece wandb

# Step 2: Configuration Check
echo ""
echo "--- Step 2: Configuration Check ---"
if [ ! -f "configs/pipeline_config.yaml" ]; then
    echo "ERROR: configs/pipeline_config.yaml not found!"
    exit 1
fi
echo "Configuration found. Please ensure wandb is logged in (run 'wandb login' if needed)."

# Steps 3-6: Unified Pipeline Execution
echo ""
echo "--- Steps 3-6: Executing Unified Pipeline ---"
echo "This will handle Data Prep, DoRA Training, PPO Training, and Evaluation."
python3 src/pipeline.py --config configs/pipeline_config.yaml

echo ""
echo "=== All Steps Completed Successfully ==="
