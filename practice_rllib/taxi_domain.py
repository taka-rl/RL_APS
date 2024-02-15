'''
# https://www.gymlibrary.dev/environments/toy_text/taxi/
add display for action,reward, environment
# https://docs.ray.io/en/latest/rllib/rllib-env.html

'''
import gymnasium as gym
from ray.rllib.algorithms.ppo import PPO
import time

env = gym.make("Taxi-v3", render_mode="human")
observation, info = env.reset()
# algo = PPO.from_checkpoint("C:/Users/is12f/AppData/Local/Temp/tmpjqg21cth") # trained for 5 iterations
# algo = PPO.from_checkpoint("C:/Users/is12f/AppData/Local/Temp/tmpxqocx617") # trained for 50 iterations
# algo = PPO.from_checkpoint("C:/Users/is12f/AppData/Local/Temp/tmpji63oszk") # trained for 150 iterations .training(model={"fcnet_hiddens": [64, 64]})
# algo = PPO.from_checkpoint("C:/Users/is12f/AppData/Local/Temp/tmpcyj_lqnr") # trained for 300 iterations .training(model={"fcnet_hiddens": [128, 128]})
# algo = PPO.from_checkpoint("C:/Users/is12f/AppData/Local/Temp/tmpdvegclxk") # trained for 1000 iterations .training(model={"fcnet_hiddens": [64, 64]})
algo = PPO.from_checkpoint("C:/Users/is12f/AppData/Local/Temp/tmp00uazhh2") # trained for 50 iterations with 4 rollout workers
# C:\Users\is12f\AppData\Local\Temp\tmpdafqkenc # trained for 50 iterations with 6 rollout workers
# C:\Users\is12f\AppData\Local\Temp\tmpyy6m6pma # trained for 50 iterations with 2 rollout workers

episode_reward = 0
terminated = truncated = False

obs, info = env.reset()
env.render()

while not terminated and not truncated:
    action = algo.compute_single_action(obs) # Algorithm.compute_single_action() is to programmatically compute actions from a trained agent.
    # action = env.action_space.sample() # env.action_space.sample() is to sample random actions.
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    time.sleep(0.01)
    episode_reward += reward
    print("Episode reward:", episode_reward)


algo.evaluate()  # 4. and evaluate it.
env.close()
