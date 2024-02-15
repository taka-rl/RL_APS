'''
# https://gymnasium.farama.org/environments/toy_text/frozen_lake/
'''
import gymnasium as gym
from ray.rllib.algorithms.ppo import PPO
import time

from path_select import select_path


env_name = "FrozenLake-v1"
env = gym.make(env_name, render_mode="human")
observation, info = env.reset()
checkpoint = select_path(env_name)
# algo = PPO.from_checkpoint("C:/Users/is12f/AppData/Local/Temp/tmpe8e_n0uq") # trained for 50 iterations with 2 rollout workers
algo = PPO.from_checkpoint(checkpoint + "/tmp4mapp34h") # trained for 50 iterations with 4 rollout workers

# algo = PPOConfig()
episode_reward = 0
terminated = truncated = False

obs, info = env.reset()
env.render()

while not terminated and not truncated:
    action = algo.compute_single_action(obs) # Algorithm.compute_single_action() is to programmatically compute actions from a trained agent.
    # action = env.action_space.sample() # env.action_space.sample() is to sample random actions.
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    time.sleep(0.1)
    episode_reward += reward
    print("Episode reward:", episode_reward)


