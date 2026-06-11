import numpy as np

class PatientSimulator:
    """
    Chapter 5.1 (Advanced): Digital Twin Patient Simulator.
    Simulates physiological drift based on WESAD/RESILIENT distributions and reacts to interventions.
    """
    def __init__(self, persona_type="Anxiety"):
        self.persona_type = persona_type
        self.reset()

    def reset(self):
        # Initial physiological state (Z-Scores) - simulating an elevated risk baseline
        self.state = {
            'hr_z': np.random.uniform(1.0, 2.5),   # Elevated HR
            'hrv_z': np.random.uniform(-2.5, -1.0), # Dropped HRV
            'wakeup_z': np.random.uniform(0, 2.0),  # Mild sleep issues
            'internal_stress': 0.8 # Hidden stress index (0 to 1)
        }
        return self.get_observation()

    def get_observation(self):
        return np.array([self.state['hr_z'], self.state['hrv_z'], self.state['wakeup_z']], dtype=np.float32)

    def react(self, action_idx):
        """
        Calculates the physiological reaction to a specific JITAI action.
        Action 0: Breathing, 1: Grounding, 2: Sleep, 3: Empathy
        """
        # Clinical priors: Breathing lowers HR immediately; Grounding improves HRV.
        if action_idx == 0: # Breathing
            reduction = 0.4 if self.state['hr_z'] > 1.5 else 0.1
            self.state['hr_z'] -= reduction
            self.state['internal_stress'] -= 0.2
        elif action_idx == 1: # Grounding
            reduction = 0.5 if self.state['hrv_z'] < -1.5 else 0.1
            self.state['hrv_z'] += reduction # HRV recovers (moves toward 0)
            self.state['internal_stress'] -= 0.25
        elif action_idx == 2: # Sleep Hygiene
            reduction = 0.4 if self.state['wakeup_z'] > 1.0 else 0.1
            self.state['wakeup_z'] -= reduction
            self.state['internal_stress'] -= 0.15
        elif action_idx == 3: # Empathy
            self.state['internal_stress'] -= 0.1
            
        # Add natural physiological drift (non-stationarity)
        self.state['hr_z'] += np.random.normal(0, 0.05)
        self.state['hrv_z'] -= np.random.normal(0, 0.05) # Natural tendency for stress to increase slightly
        
        self.state['internal_stress'] = np.clip(self.state['internal_stress'], 0, 1)
        
        return self.get_observation(), self.state['internal_stress']
