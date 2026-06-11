import numpy as np

class ThompsonSamplingIntervention:
    """
    Chapter 5.1: LLM+TS (Thompson Sampling) Hybrid Reinforcement Learning.
    Dynamically selects the best JITAI 'arm' (intervention type) based on 
    physiological state and historical 'success' (simulated).
    """
    def __init__(self):
        # Intervention Arms: 0: Breathing, 1: Grounding (ACT), 2: Sleep Hygiene, 3: Empathy Only
        self.arms = ["Breathing Exercise", "ACT Grounding", "Sleep Hygiene", "Empathy & Validation"]
        self.n_arms = len(self.arms)
        
        # Beta distribution parameters (Alpha: Successes, Beta: Failures)
        # Initialized with 1 to provide a uniform prior
        self.alphas = np.ones(self.n_arms)
        self.betas = np.ones(self.n_arms)

    def select_arm(self, physio_state):
        """
        Selects an intervention arm using Thompson Sampling.
        Weights the priors based on physiological state (Contextual Bandit Lite).
        """
        # Contextual adjustment: Increase priority of specific arms based on Z-scores
        state_priors = np.zeros(self.n_arms)
        if physio_state.get('hr_z_score', 0) > 1.5:
            state_priors[0] += 0.2 # Favor Breathing for high HR
        if physio_state.get('hrv_z_score', 0) < -1.5:
            state_priors[1] += 0.2 # Favor Grounding for low HRV
        if physio_state.get('wakeup_z_score', 0) > 1.5:
            state_priors[2] += 0.3 # Favor Sleep for fragmentation
            
        # Thompson Sampling: Sample from Beta(alpha, beta) + Contextual Prior
        samples = [np.random.beta(self.alphas[i], self.betas[i]) + state_priors[i] for i in range(self.n_arms)]
        return np.argmax(samples)

    def update(self, arm_idx, reward):
        """
        Updates the success/failure counts (Simulated reward).
        Reward: 1 (Effective), 0 (Ineffective).
        """
        if reward > 0.5:
            self.alphas[arm_idx] += 1
        else:
            self.betas[arm_idx] += 1

    def get_intervention_text(self, arm_idx):
        texts = [
            "[JITAI - Breathing] I've noticed your heart rate is slightly elevated. Let's try 3 cycles of box breathing (inhale 4, hold 4, exhale 4).",
            "[JITAI - ACT Grounding] Your stress markers (HRV) are showing a dip. Let's try a quick grounding exercise: name 3 things you can see and 2 things you can hear right now.",
            "[JITAI - Sleep] Your sleep data indicates significant restlessness. Would you like to review some sleep hygiene tips or try a relaxing visualization?",
            "[JITAI - Empathy] I can see your body is going through some stress. I'm here to listen. What's on your mind?"
        ]
        return texts[arm_idx]

if __name__ == "__main__":
    ts_agent = ThompsonSamplingIntervention()
    
    # Simulation: High stress user (Low HRV)
    physio = {'hr_z_score': 0.5, 'hrv_z_score': -2.5, 'wakeup_z_score': 0}
    
    print("Simulating 5 trials for a high-stress user:")
    for i in range(5):
        arm = ts_agent.select_arm(physio)
        text = ts_agent.get_intervention_text(arm)
        print(f"Trial {i+1}: Selected {ts_agent.arms[arm]} -> {text}")
        
        # Simulate reward (Assume ACT Grounding is most effective for this user)
        reward = 1.0 if arm == 1 else 0.2
        ts_agent.update(arm, reward)
    
    print(f"\nUpdated Alphas: {ts_agent.alphas}")
    print(f"Updated Betas: {ts_agent.betas}")
