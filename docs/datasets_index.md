# Research Dataset Index: Personalized Physiological Risk-Aware Agents

This index categorizes datasets suitable for training and evaluating the PRAA (Perception-Retrieval-Adaptation-Action) architecture.

---

## 1. Physiological Stress & Emotion (Perception)
*Best for training Z-Vector extraction and stress classifiers.*

*   **WESAD (Wearable Stress and Affect Detection):**
    *   **Data:** HR, HRV, EDA, PPG, ECG, EMG, Temp.
    *   **Context:** Lab-induced stress (public speaking, math).
    *   **Value:** The gold standard for physiological stress detection.
*   **LifeSnaps (2024):**
    *   **Data:** Fitbit Sense (HR, Sleep, Stress) + Ecological Momentary Assessments (EMA).
    *   **Context:** 4 months of "in-the-wild" longitudinal data.
    *   **Value:** Ideal for building personalized baselines and detecting "vulnerable states."
*   **VitaStress (2025):**
    *   **Data:** ECG, EDA from wearable sensors.
    *   **Context:** Recent lab-based stress monitoring.
*   **AddHRVr Dataset (2025):**
    *   **Focus:** Non-metabolic HRV reductions.
    *   **Value:** Precise trigger for psychological stress vs. physical exertion.

## 2. Multimodal: Physio + Text/Video (Alignment)
*Best for cross-modal alignment (e.g., Lingo-Aura or Emotion-LLaMA styles).*

*   **EEVR (Emotion Elicitation in VR, NeurIPS 2024):**
    *   **Data:** Physio (EDA, PPG) + **Natural Language Interviews**.
    *   **Value:** First dataset to pair raw textual descriptions of feelings with physiological spikes.
*   **AFFEC (2025):**
    *   **Data:** Physio (EEG, GSR) + Video (Facial) + **Emotional Dialogues**.
    *   **Value:** Excellent for training conversational agents to distinguish between "felt" and "perceived" emotion.
*   **egoEMOTION (2025):**
    *   **Data:** Head-mounted PPG + Egocentric Video (Project Aria).
    *   **Value:** Captures emotions in real-world naturalistic activities.

## 3. Mental Health Dialogue (Cognition & Adaptation)
*Best for SFT (Supervised Fine-Tuning) and RAG evaluation.*

*   **MentalChat16K:**
    *   **Structure:** 16k QA pairs (Anonymized clinical transcripts + Synthetic).
    *   **Value:** High-quality clinical grounding for LLMs.
*   **CounselChat:**
    *   **Structure:** Licensed therapist responses to user questions.
    *   **Value:** Teaches the model a professional, empathetic, and non-directive "voice."
*   **EmoCare:**
    *   **Focus:** Multi-turn empathetic conversations.
    *   **Value:** Fine-tuning for emotional intelligence.
*   **Mental Health Sensemaking (MHSD, 2025):**
    *   **Structure:** Synthetic multi-turn patient-LLM interactions.
    *   **Value:** Evaluating multi-turn consistency and "sensemaking" abilities.

## 4. JITAI & Longitudinal Behavior (Action)
*Best for training Reinforcement Learning (RL) policies and JITAI logic.*

*   **GLOBEM (Multi-Year):**
    *   **Data:** 700+ user-years of smartphone + wearable data.
    *   **Value:** Testing generalizability of depression detection models over long timeframes.
*   **StudentLife:**
    *   **Data:** Passive sensing (GPS, Audio, Activity) + Academic performance.
    *   **Value:** Early-stage depression/stress correlation modeling.
*   **StepCountJITAI:**
    *   **Type:** Simulation Environment (NeurIPS).
    *   **Value:** Training RL policies on synthetic users before deployment.

## 5. Safety & Ethics Benchmarks (Protection)
*Best for evaluating the Safety Guardrails.*

*   **HealthBench (OpenAI):** 5,000 conversations evaluated by 262 physicians.
*   **MedSafetyBench:** 1,800 requests based on AMA Ethics.
*   **PsychiatryBench:** Modular platform for psychiatrist-level evaluation of LLMs.

---
*Updated: June 2026*
