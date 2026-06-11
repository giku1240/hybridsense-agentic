from stable_baselines3 import PPO
from .env import MentalHealthEnv
import os

def train_ppo_policy():
    """
    Trains the Deep RL Action Policy using Proximal Policy Optimization (PPO).
    Uses the Digital Twin Sandbox to simulate 50,000 interactions.
    """
    print("Initializing Digital Twin Patient Environment...")
    env = MentalHealthEnv()

    print("Building PPO Model Architecture (MlpPolicy for Vector States)...")
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        tensorboard_log="./logs/ppo_mental_health/"
    )

    print("Starting Patient-Agent Self-Play Simulation (50,000 timesteps)...")
    # In a real run, this would collect trajectories and optimize the neural network
    model.learn(total_timesteps=50000)

    # Save the trained policy
    os.makedirs("models/rl", exist_ok=True)
    model.save("models/rl/ppo_jitai_policy")
    print("\nTraining Complete! PPO Model saved to models/rl/ppo_jitai_policy.zip")

if __name__ == "__main__":
    # Note: Requires 'gymnasium' and 'stable-baselines3' to be installed.
    # pip install gymnasium stable-baselines3
    train_ppo_policy()
