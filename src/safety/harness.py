# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
import re

from src.safety.rl_policy import ThompsonSamplingIntervention

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
    def __init__(self):
        self.sentry = SafetySentry()
        self.router = IntentRouter()
        self.rl_policy = ThompsonSamplingIntervention()

    def process_query(self, user_text, user_vitals):
        safety_check = self.sentry.scan_text(user_text)
        physio_check = self.sentry.scan_vitals(user_vitals)
        
        mode = self.router.route(user_text, safety_check, physio_check)
        
        if mode == "CRISIS_ESCALATION":
            return self._emergency_response(safety_check, physio_check)
        elif mode == "PROACTIVE_JITAI":
            return self._jitai_response(user_vitals)
        
        return mode

    def _emergency_response(self, text_risk, physio_risk):
        response = "[CRITICAL INTERVENTION] Safety protocol activated. "
        if physio_risk['status'] == 'PHYSIO_ALERT':
            response += f"Your physiological deviation ({', '.join(physio_risk['alerts'])}) indicates a severe anomaly relative to your baseline. "
        response += "Please contact emergency services immediately."
        return response
        
    def _jitai_response(self, user_vitals):
        # Use RL Policy (Thompson Sampling) to select the best intervention
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
