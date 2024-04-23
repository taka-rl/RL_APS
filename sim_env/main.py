import time
from ray.rllib.algorithms.ppo import PPO
from parking_env import Parking
from training.utility import set_path

env_config = dict(render_mode="human", action_type="continuous", parking_type="perpendicular")
env = Parking(env_config)

observation, info = env.reset()
checkpoint = set_path()
algo = PPO.from_checkpoint(checkpoint + "PPO_parallel_2024-04-24_00-11-07")

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
