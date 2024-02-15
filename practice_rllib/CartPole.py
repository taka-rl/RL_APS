'''
This script is used for understanding Ray RLlib.
Ray RLlib: https://docs.ray.io/en/latest/rllib/rllib-training.html
・Computing Actions
The simplest way to programmatically compute actions from a trained agent is to use Algorithm.compute_single_action().
This method preprocesses and filters the observation before passing it to the agent policy.
Here is a simple example of testing a trained agent for one episode:

Cart Pole: https://gymnasium.farama.org/environments/classic_control/cart_pole/
'''

import gymnasium as gym
from ray.rllib.algorithms.ppo import PPO
import time
from path_select import select_path


env_name = "CartPole-v1"
env = gym.make(env_name, render_mode="human")
checkpoint = select_path(env_name)
algo = PPO.from_checkpoint(checkpoint + "/tmpbgedyug4")

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
