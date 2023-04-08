import gym
from stable_baselines3 import PPO, A2C
import os
import yaml


#### Load the config in a dict called "config" 
CONFIG_PATH = "./config/"

def load_yaml(config_name):
    """Function to load yaml configuration file"""
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
    return config

config = load_yaml("my_config.yaml")

models_dir = "models/"

env = gym.make(config["env"], render_mode = config["render_mode"], full_action_space = config["full_action_space"], repeat_action_probability = config["repeat_action_probability"])
env.reset()

# model_path = f"{models_dir}/3000000.zip"
model_path = "./models/PPO/2023_04_08_at_10h-17m-48s/4.zip" 
model = PPO.load(model_path, env=env)

episodes = 5
print(f"AI will play {episodes} games")
for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        print(rewards)
