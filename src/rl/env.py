import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .patient_sim import PatientSimulator

class MentalHealthEnv(gym.Env):
    """
    Gymnasium environment for training the PRAA Agent using PPO.
    """
    def __init__(self):
        super(MentalHealthEnv, self).__init__()
        # Action Space: 4 discrete JITAI actions
        self.action_space = spaces.Discrete(4)
        
        # Observation Space: [HR_Z, HRV_Z, Wakeup_Z]
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(3,), dtype=np.float32)
        
        self.patient = PatientSimulator()
        self.max_steps = 10 # An episode consists of 10 interaction steps
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        obs = self.patient.reset()
        return obs, {}

    def step(self, action):
        self.current_step += 1
        
        # 1. Execute action and get patient reaction
        obs, stress_level = self.patient.react(action)
        
        # 2. Reward Design
        # Objective: Lower HR_Z, increase HRV_Z (closer to 0), and reduce overall stress
        reward = (1.0 - stress_level) * 2.0  # Base reward for stress reduction
        
        if obs[0] < 1.0: reward += 1.0       # Reward for stabilizing HR
        if obs[1] > -1.0: reward += 1.0      # Reward for stabilizing HRV
        
        # 3. Termination conditions
        done = False
        truncated = False
        
        if stress_level < 0.2:
            done = True
            reward += 10.0 # Large bonus for successfully 'curing' the acute state
        elif self.current_step >= self.max_steps:
            truncated = True
            
        return obs, reward, done, truncated, {}
