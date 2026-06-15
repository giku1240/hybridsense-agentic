# HybridSense-Agentic: Step-by-Step Training Guide

This guide provides a detailed walkthrough of the unified training pipeline for the HybridSense-Agentic system, covering physiological data extraction, LLM fine-tuning, and Reinforcement Learning.

## Step 1: Environment Setup
Ensure all dependencies are installed. The pipeline requires PyTorch, Transformers, PEFT, and Stable-Baselines3.

```bash
pip install torch transformers peft datasets accelerate bitsandbytes gymnasium stable-baselines3 pandas numpy scipy sentencepiece wandb
```

## Step 2: Configuration
The pipeline is governed by `configs/pipeline_config.yaml`. Key parameters include:
- `model_name_or_path`: Target LLM (e.g., Qwen2.5-7B-Instruct).
- `use_dora`: Set to `true` to enable Weight-Decomposed Low-Rank Adaptation.
- `ablation_mode`: Choose between `full`, `no_hrv`, or `text_only`.

## Step 3: Data Preparation (Stage 0)
The pipeline automatically checks for processed data. If missing, it runs:
1. **Physio Extraction**: `src/data_processing/deep_extractor.py` processes raw RESILIENT data into individualized Z-scores.
2. **SFT Synthesis**: `src/data_processing/synthesizer.py` merges physiological states with conversational templates to create `data/processed/sft_train_data.jsonl`.

## Step 4: DoRA Fine-Tuning (Stage 1)
The core LLM is trained to respond to physiological triggers.
- **Process**: Loads the base model in 4-bit (if configured), integrates DoRA adapters, and trains on the synthesized dataset.
- **Monitoring**: Real-time metrics are logged to **Weights & Biases (wandb)**.
- **Output**: LoRA weights are saved to `models/hybrid-sense-dora_{ablation_mode}/`.

## Step 5: PPO Action Policy Training (Stage 2)
Trains the JITAI (Just-In-Time Adaptive Intervention) router using Reinforcement Learning.
- **Environment**: A digital twin sandbox simulates patient physiological responses.
- **Algorithm**: Proximal Policy Optimization (PPO) discovers the optimal timing for interventions.
- **Output**: RL policy saved to `models/rl/ppo_jitai_policy.zip`.

## Step 6: System Integration & Evaluation (Stage 3)
The `AgentHarness` integrates the DoRA model and the PPO policy.
- **Evaluation**: The system is tested against a hold-out dataset to measure routing accuracy and clinical adherence.
- **Command**:
  ```bash
  python src/pipeline.py --config configs/pipeline_config.yaml
  ```

## Monitoring & Debugging
- **Logs**: Detailed pipeline logs are written to `training_pipeline.log`.
- **WandB**: Visit your wandb dashboard to view training loss, learning rate, and hardware utilization.
