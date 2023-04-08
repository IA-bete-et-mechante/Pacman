import gym
from stable_baselines3 import PPO
import os
import os
import yaml
from datetime import datetime



#### Load the config in a dict called "config" 
CONFIG_PATH = "./config/"

def load_yaml(config_name):
    """Function to load yaml configuration file"""
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
    return config

config = load_yaml("my_config.yaml")


# datetime object containing current date and time
now = datetime.now().strftime("%Y_%m_%d %Hh-%Mm-%Ss")

models_dir = "models/" + config['model'] + "/" + now
logdir = "logs/" + config['model'] + "/" + now

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

# env = gym.make(config["env"], frame_skip = 0)
env = gym.make(config["env"], full_action_space = config["full_action_space"], repeat_action_probability = config["repeat_action_probability"])
env.reset()

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)


total_timesteps = config['total_timesteps']
checkpoint = config['checkpoint_freq']
nb_iter = total_timesteps // checkpoint
# iters = 0
for i in range(nb_iter):
    model.learn(total_timesteps=checkpoint, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{models_dir}/{checkpoint*i}")
