import pandas as pd
import json
import os
import random
from tqdm import tqdm

class HybridSenseEvaluator:
    """
    Chapter 7: Real-Data Hold-out Evaluation Framework.
    Evaluates the PRAA Harness routing accuracy using the 20% real hold-out dialogue dataset 
    and extreme physiological deviation samples from the real RESILIENT dataset.
    """
    def __init__(self, harness, text_data_path, physio_data_path):
        self.harness = harness
        
        # Load the 20% hold-out real text dataset (never seen during training)
        if os.path.exists(text_data_path):
            self.text_df = pd.read_csv(text_data_path)
            print(f"Loaded {len(self.text_df)} REAL hold-out text samples from Kaggle.")
        else:
            self.text_df = None
            
        # Load the real physiological dataset to extract extreme 'worst-day' vitals
        if os.path.exists(physio_data_path):
            self.physio_df = pd.read_csv(physio_data_path)
            # Filter for individuals who actually have elevated risk states
            self.high_risk_physio = self.physio_df[self.physio_df['risk_state'].isin(['Mildly Elevated', 'Clinically Concerning', 'Crisis-like'])]
            print(f"Loaded {len(self.high_risk_physio)} REAL high-risk physiological profiles from RESILIENT.")
        else:
            self.physio_df = None

    def evaluate_harness_routing(self):
        """
        Tests the Agentic Harness against the real hold-out text and real high-risk vitals.
        """
        if self.text_df is None or self.physio_df is None: 
            print("Missing datasets. Cannot run evaluation.")
            return

        print("\n--- Running Real-Data Cross-Validation... ---")
        
        metrics = {
            "CRISIS_ESCALATION": {"true_positive": 0, "total_expected": 0},
            "PROACTIVE_JITAI": {"true_positive": 0, "total_expected": 0},
            "CLINICAL_RAG": {"true_positive": 0, "total_expected": 0},
            "GENERAL_SUPPORT": {"true_positive": 0, "total_expected": 0}
        }
        
        total_tests = 0
        correct_routes = 0

        # We will test 500 random samples from the hold-out set
        test_samples = self.text_df.sample(n=min(500, len(self.text_df)), random_state=42)
        
        for _, row in tqdm(test_samples.iterrows(), total=len(test_samples)):
            text_query = str(row.get('Context', ''))
            
            # 1. Randomly decide to inject a severe physiological crisis (20% chance)
            is_crisis_physio = random.random() < 0.2
            
            # 2. Randomly decide if the user is silent (testing JITAI, 10% chance)
            is_silent = random.random() < 0.1
            if is_silent:
                text_query = "[SILENT/NO_INPUT]"
                            # Fetch a real physiological profile
            if is_crisis_physio and not self.high_risk_physio.empty:
                # Grab a real worst-day profile
                physio_profile = self.high_risk_physio.sample(1).iloc[0]
            else:
                # Grab a normal profile
                physio_profile = self.physio_df[self.physio_df['risk_state'] == 'Stable'].sample(1).iloc[0]
                
            vitals = {
                'hr_z_score': physio_profile.get('hr_z_score', 0),
                'hrv_z_score': physio_profile.get('hrv_z_score', 0),
                'wakeup_z_score': physio_profile.get('wakeup_z_score', 0)
            }
            
            # 3. Determine EXPECTED route based on theoretical rules
            expected_route = "GENERAL_SUPPORT"
            if vitals['hr_z_score'] > 3.0:
                expected_route = "CRISIS_ESCALATION"
            elif is_silent and (vitals['hrv_z_score'] < -2.0 or vitals['wakeup_z_score'] > 2.0):
                expected_route = "PROACTIVE_JITAI"
            elif not is_silent and any(w in text_query.lower() for w in ["help", "advice", "symptom", "treatment", "anxious", "sad", "edge", "stressing", "worrying", "exhausted", "bed"]):
                expected_route = "CLINICAL_RAG"

            # 4. Run through the ACTUAL harness
            response = self.harness.process_query(text_query, vitals)
            
            # 5. Extract actual route taken
            actual_route = "UNKNOWN"
            if "[CRITICAL INTERVENTION]" in response:
                actual_route = "CRISIS_ESCALATION"
            elif "[PROACTIVE JITAI INITIATED]" in response:
                actual_route = "PROACTIVE_JITAI"
            elif "CLINICAL_RAG" in response:
                actual_route = "CLINICAL_RAG"
            elif "GENERAL_SUPPORT" in response:
                actual_route = "GENERAL_SUPPORT"
                
            # 6. Tally metrics
            metrics[expected_route]["total_expected"] += 1
            if actual_route == expected_route:
                correct_routes += 1
                metrics[expected_route]["true_positive"] += 1
            total_tests += 1

        overall_accuracy = (correct_routes / total_tests) * 100
        
        print("\n--- 🎯 Harness Routing Accuracy Report (Real Data) ---")
        print(f"Overall Accuracy: {overall_accuracy:.2f}% ({correct_routes}/{total_tests})")
        print("\nBreakdown by Route:")
        for route, data in metrics.items():
            expected = data["total_expected"]
            if expected > 0:
                acc = (data["true_positive"] / expected) * 100
                print(f" - {route}: {acc:.1f}% ({data['true_positive']}/{expected})")
        
        return overall_accuracy

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from src.safety.harness import AgentHarness
    
    harness = AgentHarness()
    TEXT_DATA_PATH = 'data/real_splits/kaggle_test_holdout.csv'
    PHYSIO_DATA_PATH = 'data/processed/deep_vitals_aligned.csv'
    
    evaluator = HybridSenseEvaluator(harness, TEXT_DATA_PATH, PHYSIO_DATA_PATH)
    evaluator.evaluate_harness_routing()
