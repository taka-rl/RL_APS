import time
from pathlib import Path
from ray.rllib.algorithms.ppo import PPO

from parameters import Config, PI
from parking_env import Parking
from training.utility import create_folder_path


if __name__ == '__main__':

    config = Config(car_length=4.0, car_width=2.0,
                    wheel_length=0.75, wheel_width=0.35,
                    parking_length=6.0, parking_width=4.0,
                    max_distance=25.0, max_steps=80,
                    acceleration_limit=1.0, steering_limit=PI/4, velocity_limit=10.0,
                    max_angle_error=PI/12, center_threshold=0.5, penalty_ratio={'angle': 0.25, 'velocity': 0.25},
                    reward_type='type1', state_type='type1',
                    side=(1, 2, 3, 4), car_loc_randomize_range=(-5, 5), initial_distance_range=(7.5, 15.0)
                    )
    env_config = {"render_mode": "human",
                  "action_type": "continuous",
                  "parking_type": "perpendicular",
                  "training_mode": "off",
                  'config': config}

    env = Parking(env_config)

    folder_name = 'PPO_perpendicular_continuous_10_r1_s1_all'  # trained_agent folder
    folder_path = create_folder_path(env_config, is_training=False)
    checkpoint_path = Path(folder_path) / folder_name

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_path}")

    algo = PPO.from_checkpoint(str(checkpoint_path))
    episode_reward = 0
    for i in range(10):
        episode_reward = 0
        terminated = truncated = False
        obs, info = env.reset()
        actions = []
        while not terminated and not truncated:
            # Algorithm.compute_single_action() is to programmatically compute actions from a trained agent.
            action = algo.compute_single_action(obs, explore=False)
            # action = env.action_space.sample()  # env.action_space.sample() is to sample random actions.
            # action = int(input("Action: "))
            actions.append(action)
            obs, reward, terminated, truncated, info = env.step(action)
            # print("obs: ", obs, "reward: ", reward, "info: ", info)
            time.sleep(0.1)
            episode_reward += reward
            # print("Episode reward:", episode_reward)
        print(f'{i}: {episode_reward}')

    env.close()
    algo.stop()
