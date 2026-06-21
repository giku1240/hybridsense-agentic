#!/bin/bash

# HybridSense-Agentic: Master Training Script
# Automates the steps outlined in docs/TRAINING_STEPS.md
#
# Usage:
#   bash run_steps.sh                          # Full mode (default)
#   bash run_steps.sh --ablation no_hrv        # Ablation: remove HRV
#   bash run_steps.sh --ablation text_only     # Ablation: text only
#   bash run_steps.sh --ablation all           # Run all 3 variants sequentially

set -e # Exit on error

# Parse --ablation argument
ABLATION_ARG="full"
for arg in "$@"; do
    case $arg in
        --ablation=*) ABLATION_ARG="${arg#*=}" ;;
        --ablation)   shift; ABLATION_ARG="$1" ;;
    esac
done

echo "=== HybridSense-Agentic: Starting Training Pipeline Orchestrator ==="
echo "Ablation mode: $ABLATION_ARG"

# Step 1: Environment Setup
echo ""
echo "--- Step 1: Checking/Installing Dependencies ---"
# Set Hugging Face mirror to avoid connectivity issues (Errno 99)
export HF_ENDPOINT="https://hf-mirror.com"
pip install "torch>=2.6" transformers peft datasets accelerate bitsandbytes gymnasium stable-baselines3 pandas numpy scipy sentencepiece wandb

# Step 2: Configuration Check
echo ""
echo "--- Step 2: Configuration Check ---"
if [ ! -f "configs/pipeline_config.yaml" ]; then
    echo "ERROR: configs/pipeline_config.yaml not found!"
    exit 1
fi
echo "Configuration found. Please ensure wandb is logged in (run 'wandb login' if needed)."

# Step 2.5: Clear stale Python bytecode caches so updated source files take effect
echo ""
echo "--- Clearing Python __pycache__ directories ---"
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
echo "Python caches cleared."
# -----------------------------------------------------------------------
# Helper: run a single ablation variant
# -----------------------------------------------------------------------
run_variant() {
    local MODE="$1"
    echo ""
    echo "====== Running variant: $MODE ======"

    # Read output_dir for this variant from pipeline config
    DORA_OUTPUT_DIR=$(python3 -c "
import yaml
with open('configs/pipeline_config.yaml') as f:
    c = yaml.safe_load(f)
d = c['dora_training']
print(d['output_dir'] + '_' + '$MODE')
")

    # Auto-detect latest checkpoint for resume
    RESUME_FLAG=""
    if [ -d "$DORA_OUTPUT_DIR" ]; then
        LATEST_CKPT=$(ls -d "${DORA_OUTPUT_DIR}"/checkpoint-* 2>/dev/null \
            | awk -F'checkpoint-' '{print $2, $0}' \
            | sort -n \
            | tail -1 \
            | awk '{print $2}')
        if [ -n "$LATEST_CKPT" ]; then
            echo "Found checkpoint: $LATEST_CKPT — will resume."
            RESUME_FLAG="--resume $LATEST_CKPT"
        else
            echo "No checkpoint found — starting fresh."
        fi
    else
        echo "No existing model directory — starting fresh."
    fi

    # shellcheck disable=SC2086
    python3 src/pipeline.py \
        --config configs/pipeline_config.yaml \
        --ablation "$MODE" \
        $RESUME_FLAG
}

# -----------------------------------------------------------------------
# Step 3-7: Execute pipeline for requested ablation variant(s)
# -----------------------------------------------------------------------
echo ""
echo "--- Steps 3-7: Executing Unified Pipeline ---"

if [ "$ABLATION_ARG" = "all" ]; then
    echo "Running all 3 ablation variants sequentially: full → no_hrv → text_only"
    run_variant "full"
    run_variant "no_hrv"
    run_variant "text_only"
else
    run_variant "$ABLATION_ARG"
fi

echo ""
echo "=== All Steps Completed Successfully ==="
