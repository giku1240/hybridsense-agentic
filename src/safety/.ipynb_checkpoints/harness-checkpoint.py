import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import re
import yaml
import os


# Set Hugging Face mirror to avoid connectivity issues
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from src.safety.rl_policy import PPOIntervention


def _load_model_config():
    """
    Load model configuration with priority:
      1. configs/pipeline_config.yaml  -> dora_training section
      2. configs/train_config.yaml     -> top-level fallback
    Returns a flat dict containing at least 'model_name_or_path'
    plus any quantization fields if present.
    """
    for cfg_path, key in [
        ("configs/pipeline_config.yaml", "dora_training"),
        ("configs/train_config.yaml", None),
    ]:
        if os.path.exists(cfg_path):
            with open(cfg_path, "r") as f:
                raw = yaml.safe_load(f)
            cfg = raw[key] if key and key in raw else raw
            if cfg and cfg.get("model_name_or_path"):
                return cfg
    raise FileNotFoundError(
        "Neither configs/pipeline_config.yaml nor configs/train_config.yaml "
        "contains a usable model_name_or_path."
    )


def _build_bnb_config(cfg):
    """Build a BitsAndBytesConfig from config fields, or return None.

    BitsAndBytesConfig is imported lazily to keep this module importable
    in environments where bitsandbytes / quantization is unavailable.
    """
    if not (cfg.get("load_in_4bit") or cfg.get("load_in_8bit")):
        return None
    from transformers import BitsAndBytesConfig
 
    return BitsAndBytesConfig(
        load_in_4bit=cfg.get("load_in_4bit", False),
        load_in_8bit=cfg.get("load_in_8bit", False),
        bnb_4bit_compute_dtype=(
            torch.float16
            if cfg.get("bnb_4bit_compute_dtype") == "float16"
            else torch.float32
        ),
        bnb_4bit_quant_type=cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=cfg.get("bnb_4bit_use_double_quant", True),
    )

class SafetySentry:
    """
    Chapter 5.1: 0.5B Sentry Model for Proactive Guardrails (Z-Score Upgraded).
    Responsible for high-speed risk detection and continuous physiological state monitoring.
    """
    def __init__(self, model_path="Qwen/Qwen2.5-0.5B"):
        print(f"Initializing Safety Sentry with {model_path}...")
        self.crisis_keywords = ["suicide", "self-harm", "kill myself", "end it all", "prescribe"]

    def scan_text(self, text):
        if not text or text == "[SILENT/NO_INPUT]":
            return {"status": "PASS"}
            
        text = text.lower()
        found_triggers = [w for w in self.crisis_keywords if w in text]
        if found_triggers:
            return {"status": "RISK_DETECTED", "triggers": found_triggers}
        return {"status": "PASS"}

    def scan_vitals(self, physio_data):
        """Monitors physiological deviation using Z-scores (Individualized Baselines)."""
        alerts = []
        jitai_triggers = []
        
        hr_z = physio_data.get('hr_z_score', 0)
        hrv_z = physio_data.get('hrv_z_score', 0)
        wakeup_z = physio_data.get('wakeup_z_score', 0)
        
        # 1. Critical Crisis Alerts (> 3 Standard Deviations)
        if hr_z > 3.0:
            alerts.append(f"CRITICAL_TACHYCARDIA (Z={hr_z:.1f})")
            
        # 2. JITAI Proactive Triggers (Sub-critical: 1.5 to 3.0 Std Devs)
        if hrv_z < -2.0 or hr_z > 2.0 or wakeup_z > 2.0:
            jitai_triggers.append("PHYSIO_DEVIATION_DETECTED")
            
        if alerts:
            return {"status": "PHYSIO_ALERT", "alerts": alerts}
        if jitai_triggers:
            return {"status": "JITAI_TRIGGER", "triggers": jitai_triggers}
            
        return {"status": "PASS"}

class IntentRouter:
    """
    Chapter 5.2: Intent Routing and Continuous State Management.
    Decides between RAG, Crisis Mode, JITAI, or General Empathy.
    """
    def route(self, text, safety_status, physio_status):
        if safety_status['status'] == 'RISK_DETECTED' or physio_status['status'] == 'PHYSIO_ALERT':
            return "CRISIS_ESCALATION" 
            
        if physio_status['status'] == 'JITAI_TRIGGER' and (not text or text == "[SILENT/NO_INPUT]"):
            return "PROACTIVE_JITAI" 
        
        text = str(text).lower()
        # Expanded keywords to match both real-world symptoms and simulated data
        clinical_keywords = [
            "help", "advice", "symptom", "treatment", "anxious", "sad",
            "edge", "stressing", "worrying", "exhausted", "bed"
        ]
        if any(w in text for w in clinical_keywords):
            return "CLINICAL_RAG"             
        return "GENERAL_SUPPORT"
class AgentHarness:
    """
    The Orchestrator. Coordinates Sentry, Router, and Main LLM with Physiological Gating.
    """
    def __init__(self, model_path="models/hybrid-sense-dora_full", base_model_name=None):
        """
        Args:
            model_path: Path to the trained LoRA adapter directory.
            base_model_name: HuggingFace model id for the base LLM. If None,
                it is auto-resolved from configs/pipeline_config.yaml
                (dora_training.model_name_or_path), with a fallback to
                configs/train_config.yaml.
        """
        self.sentry = SafetySentry()
        self.router = IntentRouter()
        # Updated to PPO Policy
        self.rl_policy = PPOIntervention()

        # Load Config (pipeline_config preferred → train_config fallback)
        self.config = _load_model_config()

        # Resolve base model: explicit arg > config file
        resolved_base = base_model_name or self.config["model_name_or_path"]

        # Build quantization config from training-time settings to keep
        # base model + LoRA adapters dimensionally and dtype-compatible.
        bnb_config = _build_bnb_config(self.config)

        print(f"Loading main LLM ({resolved_base}) with LoRA adapters...")
        self.tokenizer = AutoTokenizer.from_pretrained(resolved_base, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            resolved_base,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

        if os.path.exists(model_path):
            print(f"Applying LoRA adapters from {model_path}...")
            self.model = PeftModel.from_pretrained(base_model, model_path)
        else:
            print("Warning: LoRA adapters not found. Using base model.")
            self.model = base_model

    def process_query(self, user_text, user_vitals):
        safety_check = self.sentry.scan_text(user_text)
        physio_check = self.sentry.scan_vitals(user_vitals)
        
        mode = self.router.route(user_text, safety_check, physio_check)
        
        if mode == "CRISIS_ESCALATION":
            return self._emergency_response(safety_check, physio_check)
        elif mode == "PROACTIVE_JITAI":
            return self._jitai_response(user_vitals)
        
        # Mode is CLINICAL_RAG or GENERAL_SUPPORT
        return self._generate_response(user_text, user_vitals, mode)

    def _generate_response(self, text, vitals, mode):
        # Construct prompt with Physiological Gating
        persona = f"Current Physiology: HR_Z={vitals['hr_z_score']:.1f}, HRV_Z={vitals['hrv_z_score']:.1f}, Wakeup_Z={vitals['wakeup_z_score']:.1f}"
        prompt = f"System: You are a professional mental health assistant. Mode: {mode}.\n\n{persona}\n\nClient: {text}\n\nAssistant:"
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=150, do_sample=True, temperature=0.7)
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the assistant's part
        response = response.split("Assistant:")[-1].strip()
        return f"[{mode}] {response}"

    def _emergency_response(self, text_risk, physio_risk):
        response = "[CRITICAL INTERVENTION] Safety protocol activated. "
        if physio_risk['status'] == 'PHYSIO_ALERT':
            response += f"Your physiological deviation ({', '.join(physio_risk['alerts'])}) indicates a severe anomaly relative to your baseline. "
        response += "Please contact emergency services immediately."
        return response
        
    def _jitai_response(self, user_vitals):
        # Use RL Policy (PPO) to select the best intervention
        arm_idx = self.rl_policy.select_arm(user_vitals)
        response = self.rl_policy.get_intervention_text(arm_idx)
        return response


if __name__ == "__main__":
    harness = AgentHarness()
    
    print("Test Case 1 (Crisis - Z-score > 3):")
    vitals_risky = {'hr_z_score': 3.5, 'hrv_z_score': -1.0, 'wakeup_z_score': 0}
    query_risky = "I can't take this."
    print(harness.process_query(query_risky, vitals_risky))
    
    print("\nTest Case 2 (Proactive JITAI - HRV Z-score < -2):")
    vitals_jitai = {'hr_z_score': 0.5, 'hrv_z_score': -2.5, 'wakeup_z_score': 0}
    query_jitai = "[SILENT/NO_INPUT]"
    print(harness.process_query(query_jitai, vitals_jitai))
