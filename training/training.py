import os
import time
from numbers import Number

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS, EPISODE_RETURN_MEAN, EPISODE_RETURN_MIN, EPISODE_RETURN_MAX, EPISODE_LEN_MEAN
)
from tensorboardX import SummaryWriter

from sim_env.parking_env import Parking
from sim_env.parameters import Config, PI
from utility import create_folder_path, create_folder_name, create_training_result_dir, create_checkpoint_dir


SCENARIOS_PERPENDICULAR = [
    # parking type1
    {
        'config':
            Config(car_length=4.0, car_width=2.0,
                wheel_length=0.75, wheel_width=0.35,
                parking_length=6.0, parking_width=4.0,
                max_distance=25.0, max_steps=80,
                acceleration_limit=1.0, steering_limit=PI/4, velocity_limit=2.77778,
                max_angle_error=PI/12, center_threshold=0.5, penalty_ratio={'angle': 0.25, 'velocity': 0.25},
                reward_type='type1', state_type='type1',
                side=(1, 2, 3, 4), car_loc_randomize_range=(-2, 2), initial_distance_range=(5.0, 7.5)
                ),
        'num_train': 10,
        "parking_type": "perpendicular",
    },
    """
    # parking type2
    {
        'config':
            Config(car_length=4.0, car_width=2.0,
                   wheel_length=0.75, wheel_width=0.35,
                   parking_length=6.0, parking_width=4.0,
                   max_distance=25.0, max_steps=80,
                   acceleration_limit=1.0, steering_limit=PI / 4, velocity_limit=10.0,
                   max_angle_error=PI / 12, center_threshold=0.5, penalty_ratio={'angle': 0.5, 'velocity': 0.0},
                   reward_type='type2', state_type='type2',
                   side=(1, 2, 3, 4), car_loc_randomize_range=(-5, 5), initial_distance_range=(7.5, 15.0)
                   ),
        'num_train': 100,
        "parking_type": "perpendicular",
    },
    # parking type3
    {
        'config':
            Config(car_length=4.0, car_width=2.0,
                   wheel_length=0.75, wheel_width=0.35,
                   parking_length=6.0, parking_width=4.0,
                   max_distance=25.0, max_steps=80,
                   acceleration_limit=1.0, steering_limit=PI / 4, velocity_limit=10.0,
                   max_angle_error=PI / 12, center_threshold=0.5, penalty_ratio={'angle': 0.0, 'velocity': 0.5},
                   reward_type='type3', state_type='type3',
                   side=(1, 2, 3, 4), car_loc_randomize_range=(-5, 5), initial_distance_range=(7.5, 15.0)
                   ),
        'num_train': 100,
    "parking_type": "perpendicular",
    },

    # parking type4
    {
        'config':
            Config(car_length=4.0, car_width=2.0,
                   wheel_length=0.75, wheel_width=0.35,
                   parking_length=6.0, parking_width=4.0,
                   max_distance=25.0, max_steps=80,
                   acceleration_limit=1.0, steering_limit=PI / 4, velocity_limit=10.0,
                   max_angle_error=PI / 12, center_threshold=0.5, penalty_ratio={'angle': 0.25, 'velocity': 0.25},
                   reward_type='type4', state_type='type4',
                   side=(1, 2, 3, 4), car_loc_randomize_range=(-5, 5), initial_distance_range=(7.5, 15.0)
                   ),
        'num_train': 100,
        "parking_type": "perpendicular",
    }
    """
]

SCENARIOS_PARALLEL = [
    # parking type1
    {
        'config':
            Config(car_length=4.0, car_width=2.0,
                   wheel_length=0.75, wheel_width=0.35,
                   parking_length=6.0, parking_width=4.0,
                   max_distance=25.0, max_steps=80,
                   acceleration_limit=1.0, steering_limit=PI / 4, velocity_limit=10.0,
                   max_angle_error=PI / 12, center_threshold=0.5, penalty_ratio={'angle': 0, 'velocity': 0.5},
                   reward_type='type1', state_type='type1',
                   side=1, car_loc_randomize_range=(6.0, 7.0), initial_distance_range=(5.0, 7.0)
                   ),
        'num_train': 100,
        "parking_type": "parallel",
    }
]

def train_and_evaluate(config: Config, num_train: int,  parking_type: str,):

    env_config = {"render_mode": "no_render",
                  "action_type": "continuous",
                  "parking_type": parking_type,
                  "training_mode": "on",
                  'config': config}

    # for folder names
    folder_name = create_folder_name('PPO',
                                     env_config,
                                     config.reward_type,
                                     config.state_type,
                                     num_train,
                                     config.side,
                                     create_folder_path(env_config, is_training=True),
                                     config.center_threshold,
                                     config.penalty_ratio['angle'],
                                     config.penalty_ratio['velocity']
                                     )

    # --------------------------------------------------
    # TensorBoard training-result folder
    # --------------------------------------------------
    training_root = create_folder_path(env_config, is_training=True)
    training_result_dir = create_training_result_dir(training_root, folder_name)
    writer = SummaryWriter(training_result_dir)

    # --------------------------------------------------
    # Build algorithm
    # --------------------------------------------------
    algo = (
        PPOConfig()
        .environment(env=Parking, env_config=env_config)
        .env_runners(num_env_runners=1)
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
        .framework("torch")
        .evaluation(evaluation_num_env_runners=1)
        .build_algo()
    )

    start_time = time.time()

    try:
        for iteration in range(num_train):
            result = algo.train()

            env_metrics = result.get(ENV_RUNNER_RESULTS, {})

            metrics = {
                EPISODE_RETURN_MEAN: env_metrics.get(EPISODE_RETURN_MEAN),
                EPISODE_RETURN_MIN: env_metrics.get(EPISODE_RETURN_MIN),
                EPISODE_RETURN_MAX: env_metrics.get(EPISODE_RETURN_MAX),
                EPISODE_LEN_MEAN: env_metrics.get(EPISODE_LEN_MEAN),
            }

            for metric_name, value in metrics.items():
                if isinstance(value, Number):
                    writer.add_scalar(f"training/{metric_name}", float(value), iteration)

            writer.flush()

            print(
                f"Iteration {iteration}: "
                f"return mean={metrics[EPISODE_RETURN_MEAN]}, "
                f"return min={metrics[EPISODE_RETURN_MIN]}, "
                f"return max={metrics[EPISODE_RETURN_MAX]}, "
                f"length mean={metrics[EPISODE_LEN_MEAN]}"
            )

    finally:
        writer.close()

    end_time = time.time()
    print(f"Total execution time of the script: {end_time - start_time} second")

    algo.evaluate()

    # --------------------------------------------------
    # Separate trained-agent checkpoint folder
    # --------------------------------------------------
    checkpoint_root = create_folder_path(env_config, is_training=False)
    checkpoint_dir = create_checkpoint_dir(checkpoint_root, folder_name)
    saved_checkpoint_path = algo.save_to_path(checkpoint_dir)
    print(f"Checkpoint saved in directory {saved_checkpoint_path}")

    # release the resources
    algo.stop()


if __name__ == '__main__':
    ray.init()

    try:
        for scenario in SCENARIOS_PERPENDICULAR:
            train_and_evaluate(**scenario)
    finally:
        ray.shutdown()
