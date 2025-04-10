import time
from ray.rllib.algorithms.ppo import PPO
from parameters import Config, PI
from parking_env import Parking
from training.utility import create_folder_path

config = Config(car_length=4.0, car_width=2.0,
                wheel_length=0.75, wheel_width=0.35,
                parking_length=6.0, parking_width=4.0,
                max_distance=25.0, max_steps=80,
                acceleration_limit=1.0, steering_limit=PI/4, velocity_limit=10.0,
                max_angle_error=PI/12, center_threshold=1.0, penalty_ratio={'angle': 0.5, 'velocity': 0.0},
                reward_type='type2', state_type='type2',
                side=1, car_loc_randomize_range=(5, 7), initial_distance_range=(5.0, 7.0)
                )
env_config = {"render_mode": "human",
              "action_type": "continuous",
              "parking_type": "parallel",
              "training_mode": "off",
              'config': config}

env = Parking(env_config)

folder_name = ''  # trained_agent folder
folder_path = create_folder_path(env_config, is_training=False)
folder_path = folder_path.replace('sim_env', 'training')

algo = PPO.from_checkpoint(folder_path + folder_name)
episode_reward = 0
for i in range(10):
    episode_reward = 0
    terminated = truncated = False
    obs, info = env.reset()
    env.render()
    actions = []
    while not terminated and not truncated:
        # Algorithm.compute_single_action() is to programmatically compute actions from a trained agent.
        action = algo.compute_single_action(obs)
        # action = env.action_space.sample()  # env.action_space.sample() is to sample random actions.
        # action = int(input("Action: "))
        actions.append(action)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        # print("obs: ", obs, "reward: ", reward, "info: ", info)
        time.sleep(0.1)
        episode_reward += reward
        # print("Episode reward:", episode_reward)
    print(f'{i}: {episode_reward}')
