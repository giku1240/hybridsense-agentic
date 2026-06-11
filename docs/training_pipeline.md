# PRAA Architecture: End-to-End Training Pipeline

This document outlines the complete training and optimization pipeline for the **HybridSense-Agentic** project. The pipeline consists of two primary phases: Supervised Fine-Tuning (SFT) for the language model and Deep Reinforcement Learning (RL) for the action policy.

## Phase 1: Data Preparation & SFT (DoRA)
The first phase involves training the core Large Language Model (Qwen2.5) to understand physiological Z-Vectors and generate appropriate clinical responses. We use **Weight-Decomposed Low-Rank Adaptation (DoRA)** for parameter-efficient fine-tuning.

### 1.1 Dataset Preparation
1.  **Raw Data Splitting**: Split the Kaggle conversational dataset into training and testing sets.
    ```bash
    python src/evaluation/real_data_splitter.py
    ```
2.  **Physiological Baseline Extraction**: Extract Z-Scores (HR, HRV, Wakeups) from the RESILIENT dataset.
    ```bash
    python src/data_processing/deep_extractor.py
    ```
3.  **SFT Dataset Synthesis**: Combine the Z-Scores and conversational data to generate instruction pairs.
    ```bash
    python src/data_processing/synthesizer.py
    ```
    *Output*: `data/processed/sft_train_data.jsonl`

### 1.2 DoRA Model Training
Train the model using the synthesized dataset. The training script supports ablation studies to verify the contribution of specific physiological markers.

*   **Requirements**: High VRAM GPU (e.g., 24GB+ for 7B models, or adjust batch size/quantization for smaller GPUs).
*   **Command**:
    ```bash
    # Full multi-modal gating
    python src/model/train.py --ablation full
    
    # Ablation: Remove HRV
    python src/model/train.py --ablation no_hrv
    
    # Baseline: Text-only
    python src/model/train.py --ablation text_only
    ```
*   *Output*: LoRA weights saved in `models/hybrid-sense-dora_full/`.

---

## Phase 2: Deep Reinforcement Learning (PPO Sandbox)
The second phase trains the **Action Layer** of the PRAA architecture. Instead of relying on static rules or simple bandit algorithms, we use Proximal Policy Optimization (PPO) in a simulated environment to learn the optimal intervention strategy.

### 2.1 The Digital Twin Sandbox
We have built a Gymnasium environment (`src/rl/env.py`) that simulates a patient (`src/rl/patient_sim.py`).
*   **Observation Space**: The patient's current physiological state `[HR_Z, HRV_Z, Wakeup_Z]`.
*   **Action Space**: 4 discrete Just-In-Time Adaptive Interventions (JITAI) (e.g., Breathing, ACT Grounding).
*   **Reward**: The agent is rewarded for stabilizing the patient's physiology (lowering HR, restoring HRV) and penalized for escalating stress.

### 2.2 PPO Training Execution
Run the self-play simulation to train the policy network. The agent will interact with the simulated patient for tens of thousands of episodes to discover the optimal intervention timing and selection.

*   **Requirements**: `gymnasium`, `stable-baselines3`.
*   **Command**:
    ```bash
    python src/rl/train_ppo.py
    ```
*   *Output*: PPO policy model saved as `models/rl/ppo_jitai_policy.zip`.

---

## Phase 3: System Integration & Evaluation
Once both models are trained, they are integrated into the `AgentHarness` for real-time inference.

1.  **Update Harness**: Modify `src/safety/harness.py` to load the trained PPO policy instead of the basic Thompson Sampling class.
2.  **Evaluate**: Run the cross-validation script against the 20% hold-out dataset to measure routing accuracy and clinical adherence.
    ```bash
    python src/evaluation/metrics.py
    ```
3.  **Prototype**: Launch the Gradio UI to interact with the fully trained system.
    ```bash
    python app/app.py
    ```
