# HybridSense-Agentic: Unified Training Pipeline Documentation

This document provides a comprehensive guide to the remade, automated training pipeline for the HybridSense-Agentic project.

## 🚀 Overview

The Unified Training Pipeline orchestrates the transition from raw physiological data to a fully integrated, proactive mental health agent. It integrates two primary AI training paradigms:
1.  **Supervised Fine-Tuning (SFT)** via **DoRA** for language understanding.
2.  **Reinforcement Learning (RL)** via **PPO** for action policy optimization.

---

## 🏗️ Pipeline Architecture

The pipeline is divided into four distinct stages:

### Stage 0: Data Preparation & Synthesis
*   **Physiological Extraction**: Processes raw data from the RESILIENT dataset to calculate individualized longitudinal baselines (Z-scores) for HR, HRV, and Sleep Wakeups.
*   **Instruction Synthesis**: Pairs physiological "personas" with conversational data and generates proactive JITAI (Just-In-Time Adaptive Intervention) scenarios.
*   **Output**: `data/processed/sft_train_data.jsonl`

### Stage 1: Phase 1 Training (DoRA SFT)
*   **Model**: Qwen2.5-7B-Instruct.
*   **Technique**: Weight-Decomposed Low-Rank Adaptation (DoRA).
*   **Goal**: Fine-tune the LLM to modulate its empathy and clinical depth based on the input physiological gating signal.
*   **Ablation Support**: Supports `full`, `no_hrv`, and `text_only` modes for research validation.

### Stage 2: Phase 2 Training (PPO RL)
*   **Environment**: Digital Twin Sandbox (Gymnasium).
*   **Algorithm**: Proximal Policy Optimization (PPO).
*   **Goal**: Train the Action Layer to select the optimal intervention (e.g., ACT Grounding, Breathing exercises) based on real-time physiological deviation.

### Stage 3: System Integration & Evaluation
*   **Harnessing**: Merges the trained LoRA adapters and the PPO policy into a single `AgentHarness`.
*   **Cross-Validation**: Runs the integrated system against a 20% hold-out real-world dialogue dataset and high-risk physiological profiles.
*   **Metrics**: Reports Routing Accuracy, True Positive Rates for Crisis Escalation, and JITAI adherence.

---

## ⚙️ Configuration

The entire pipeline is controlled via `configs/pipeline_config.yaml`. Key sections include:

- `data_prep`: Paths to raw/processed data and synthesis limits.
- `dora_training`: Hyperparameters for the LLM fine-tuning (Rank, Alpha, Learning Rate).
- `ppo_training`: RL parameters (Timesteps, Batch Size).
- `evaluation`: Metrics and test dataset paths.

---

## 🛠️ Usage Instructions

### 1. Requirements
Ensure you have the following installed:
- `torch`, `transformers`, `peft`, `bitsandbytes` (for DoRA)
- `stable-baselines3`, `gymnasium` (for PPO)
- `pyyaml`, `pandas`, `datasets`

### 2. Execution
Run the complete pipeline from the project root:

```bash
python3 src/pipeline.py --config configs/pipeline_config.yaml
```

### 3. Monitoring
Check `training_pipeline.log` for real-time progress, error logs, and the final Evaluation Report.

---

## 📊 Artifacts
- **Logs**: `training_pipeline.log`
- **Models**: `models/hybrid-sense-dora_full/` and `models/rl/ppo_jitai_policy.zip`
- **Data**: `data/processed/sft_train_data.jsonl`
