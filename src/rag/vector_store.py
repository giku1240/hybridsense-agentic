import numpy as np

class PhysioFAISSRetriever:
    """
    Chapter 5.2: Risk-Conditioned Retrieval (RCR).
    Implements the mathematical retrieval model:
    Score(q,d) = α*Sim_text(q,d) + β*Sim_risk(r_q, r_d) + γ*RoutePrior(d|state)
    """
    def __init__(self):
        print("Initializing Risk-Conditioned Retrieval (RCR) Engine...")
        # Mock database with associated risk target profiles
        self.mock_db = {
            "doc_001": {
                "content": "CBT Protocol for Panic Attacks: Grounding techniques (5-4-3-2-1 method).",
                "target_risk_profile": {"hr_z_score": 2.5, "hrv_z_score": 0, "wakeup_z_score": 0},
                "prior_weight": 1.2 # High priority for acute conditions
            },
            "doc_002": {
                "content": "CBT-I Protocol for Insomnia: Sleep restriction therapy and stimulus control.",
                "target_risk_profile": {"hr_z_score": 0, "hrv_z_score": 0, "wakeup_z_score": 2.0},
                "prior_weight": 1.0
            },
            "doc_003": {
                "content": "ACT Protocol for Chronic Stress: Defusion and acceptance exercises.",
                "target_risk_profile": {"hr_z_score": 0.5, "hrv_z_score": -2.0, "wakeup_z_score": 0.5},
                "prior_weight": 1.1
            },
            "doc_004": {
                "content": "General Empathy Guidelines: Listen actively, validate feelings.",
                "target_risk_profile": {"hr_z_score": 0, "hrv_z_score": 0, "wakeup_z_score": 0},
                "prior_weight": 0.8
            }
        }
        
        # Retrieval hyperparameters (α, β, γ)
        self.alpha = 0.4  # Weight for text similarity
        self.beta = 0.5   # Weight for physiological risk match (emphasized in clinical setting)
        self.gamma = 0.1  # Weight for document prior

    def _sim_text(self, query, doc_content):
        """Mock text similarity based on keyword overlap."""
        query = query.lower()
        content = doc_content.lower()
        score = 0.0
        if 'panic' in query and 'panic' in content: score += 0.8
        if 'sleep' in query and 'insomnia' in content: score += 0.8
        if 'stress' in query and 'stress' in content: score += 0.8
        return min(score + 0.1, 1.0) # Add base similarity

    def _sim_risk(self, user_z_scores, doc_target_z_scores):
        """Calculates physiological risk similarity using negative L2 distance."""
        # user_z_scores: e.g., {'hr_z_score': 1.5, ...}
        dist = 0
        for key in ['hr_z_score', 'hrv_z_score', 'wakeup_z_score']:
            user_val = user_z_scores.get(key, 0)
            doc_val = doc_target_z_scores.get(key, 0)
            dist += (user_val - doc_val) ** 2
        
        # Convert distance to similarity (0 to 1)
        similarity = 1.0 / (1.0 + np.sqrt(dist))
        return similarity

    def retrieve(self, text_intent, physio_data, top_k=1):
        """
        Executes Risk-Conditioned Retrieval (RCR).
        """
        best_doc = None
        best_score = -1.0
        
        # Extract Z-scores from physio_data (assuming they are passed from the Harness/Extractor)
        user_z_scores = {
            'hr_z_score': physio_data.get('hr_z_score', 0),
            'hrv_z_score': physio_data.get('hrv_z_score', 0),
            'wakeup_z_score': physio_data.get('wakeup_z_score', 0)
        }

        for doc_id, doc_data in self.mock_db.items():
            sim_text = self._sim_text(text_intent, doc_data["content"])
            sim_risk = self._sim_risk(user_z_scores, doc_data["target_risk_profile"])
            route_prior = doc_data["prior_weight"]
            
            # Score(q,d) = α·Sim_text + β·Sim_risk + γ·RoutePrior
            final_score = (self.alpha * sim_text) + (self.beta * sim_risk) + (self.gamma * route_prior)
            
            if final_score > best_score:
                best_score = final_score
                best_doc = doc_data["content"]
                
        return best_doc

if __name__ == "__main__":
    retriever = PhysioFAISSRetriever()
    print("\nTesting Risk-Conditioned Retrieval (RCR):")
    
    # Case 1: High stress physiology (Low HRV), vague text
    vitals_stress = {'hr_z_score': 0.5, 'hrv_z_score': -2.5, 'wakeup_z_score': 1.0}
    context1 = retriever.retrieve(text_intent="I just feel weird.", physio_data=vitals_stress)
    print(f"Case 1 (Vague text, High Stress Vitals) -> Retrieved: {context1}")
    
    # Case 2: Explicit panic text, matching physiology
    vitals_panic = {'hr_z_score': 3.0, 'hrv_z_score': -0.5, 'wakeup_z_score': 0}
    context2 = retriever.retrieve(text_intent="Help, my heart is racing, panic attack!", physio_data=vitals_panic)
    print(f"Case 2 (Panic text, High HR Vitals) -> Retrieved: {context2}")
