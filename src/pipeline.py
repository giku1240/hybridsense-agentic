"""
HybridSense-Agentic: Unified Training & Evaluation Pipeline
Orchestrates Phase 1 (DoRA SFT), Phase 2 (PPO RL), and Phase 3 (Evaluation).
"""

import os
import sys
import yaml
import time
import argparse
import logging
from pathlib import Path


# Set Hugging Face mirror to avoid connectivity issues
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model.train import train as run_dora_training
from src.rl.train_ppo import train_ppo_policy as run_ppo_training
from src.data_processing.deep_extractor import DeepPhysioExtractor
from src.data_processing.synthesizer import DatasetSynthesizer
from src.safety.harness import AgentHarness
from src.evaluation.metrics import HybridSenseEvaluator

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training_pipeline.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def run_pipeline(config_path):
    # 1. Load Configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    logger.info(f"=== Starting HybridSense Training Pipeline [{config['experiment_version']}] ===")
    start_time = time.time()

    # Create artifacts directory
    os.makedirs(config['output_base_dir'], exist_ok=True)

    # --- STAGE 0: DATA PREPARATION ---
    logger.info("--- Stage 0: Data Preparation ---")
    
    # Run Extractor (If vitals aligned file doesn't exist or force_prep is true)
    vitals_path = config['data_prep']['processed_vitals_path']
    if not os.path.exists(vitals_path):
        logger.info("Extracting physiological Z-scores from raw data...")
        if os.path.exists(config['data_prep']['raw_vitals_path']):
            extractor = DeepPhysioExtractor(config['data_prep']['raw_vitals_path'])
            df = extractor.run()
            df.to_csv(vitals_path, index=False)
            logger.info(f"Physio data extracted and saved to {vitals_path}")
        else:
            logger.warning(f"Raw vitals path {config['data_prep']['raw_vitals_path']} not found. Skipping extraction.")
    else:
        logger.info(f"Using existing vitals file at {vitals_path}")
    
    # Run Synthesizer
    sft_dataset_path = config['data_prep']['sft_dataset_path']
    if not os.path.exists(sft_dataset_path):
        logger.info("Synthesizing SFT instruction dataset...")
        dialogues_path = config['data_prep']['dialogue_dataset_path']
        if os.path.exists(vitals_path) and os.path.exists(dialogues_path):
            synthesizer = DatasetSynthesizer(vitals_path, dialogues_path)
            count = synthesizer.synthesize(sft_dataset_path, limit=config['data_prep']['sft_limit'])
            logger.info(f"Synthesized {count} samples and saved to {sft_dataset_path}")
        else:
            logger.warning("Vitals or dialogue data missing. Skipping synthesis.")
    else:
        logger.info(f"Using existing SFT dataset at {sft_dataset_path}")

    # --- STAGE 1: DORA FINE-TUNING ---
    logger.info("--- Stage 1: DoRA Fine-Tuning (Phase 1) ---")
    dora_config = config['dora_training']
    # Ensure dataset path is correctly set for training
    dora_config['dataset_path'] = sft_dataset_path
    
    model_path = None
    if os.path.exists(sft_dataset_path):
        try:
            model_path = run_dora_training(
                config_dict=dora_config, 
                ablation_mode=dora_config.get('ablation_mode', 'full')
            )
            logger.info(f"DoRA Training successful. Model saved to {model_path}")
        except Exception as e:
            logger.error(f"DoRA Training failed: {e}")
    else:
        logger.error("SFT dataset missing. Skipping Stage 1.")
    
    # --- STAGE 2: PPO RL TRAINING ---
    logger.info("--- Stage 2: PPO Action Policy Training (Phase 2) ---")
    try:
        run_ppo_training()
        logger.info("PPO Training successful.")
    except Exception as e:
        logger.error(f"PPO Training failed: {e}")

    # --- STAGE 3: EVALUATION ---
    logger.info("--- Stage 3: System Integration & Evaluation ---")
    try:
        # Use newly trained model or fallback to default
        target_model = model_path if model_path else dora_config['output_dir'] + "_" + dora_config['ablation_mode']
        logger.info(f"Initializing Harness with model: {target_model}")
        harness = AgentHarness(model_path=target_model)
        
        # Initialize Evaluator
        eval_data_path = config['evaluation']['test_dataset_path']
        evaluator = HybridSenseEvaluator(
            harness=harness,
            text_data_path=eval_data_path,
            physio_data_path=vitals_path
        )
        
        # Run routing evaluation
        accuracy = evaluator.evaluate_harness_routing()
        logger.info(f"Evaluation Completed. Overall Routing Accuracy: {accuracy:.2f}%")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")

    end_time = time.time()
    logger.info(f"=== Pipeline Completed in {end_time - start_time:.2f} seconds ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HybridSense Training Pipeline")
    parser.add_argument("--config", type=str, default="configs/pipeline_config.yaml", help="Path to pipeline config")
    args = parser.parse_args()
    
    run_pipeline(args.config)
