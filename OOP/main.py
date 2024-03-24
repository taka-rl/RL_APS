import time
from parking_env_OOP import Parking
import ray
from ray.rllib.algorithms.ppo import PPOConfig


# ray.init()
env = Parking(render_mode="human", action_type="continuous", parking_type="parallel")

#algo = (
#    PPOConfig()
#    .environment(env=env)
#    .rollouts(num_rollout_workers=2)
#    .resources(num_gpus=0)
#    .framework("torch")
#    .evaluation(evaluation_num_workers=1)
#    .build()
#)

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
