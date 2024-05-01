import time
from ray.rllib.algorithms.ppo import PPO
from parking_env import Parking
from training.utility import set_path

env_config = {"render_mode": "human",
              "action_type": "discrete",
              "parking_type": "perpendicular",
              "training_mode": False}
env = Parking(env_config)

# folder_path = ""  # trained_agent folder
# checkpoint = set_path(env_config)
# algo = PPO.from_checkpoint(checkpoint + folder_path)

episode_reward = 0
terminated = truncated = False
obs, info = env.reset()
env.render()
while not terminated and not truncated:
    # action = algo.compute_single_action(obs)  # Algorithm.compute_single_action() is to programmatically compute actions from a trained agent.
    action = env.action_space.sample()  # env.action_space.sample() is to sample random actions.
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    time.sleep(0.1)
    episode_reward += reward
    print("Episode reward:", episode_reward)
