import os
import ray
import time
from ray.rllib.algorithms.ppo import PPOConfig
from sim_env.parking_env import Parking
from sim_env.parameters import Config, PI
from utility import custom_log_creator, custom_log_checkpoint, create_folder_path


ray.init()
env_name = Parking
config = Config(car_length=4.0, car_width=2.0,
                wheel_length=0.75, wheel_width=0.35,
                parking_length=6.0, parking_width=4.0,
                max_distance=25.0, max_steps=80,
                acceleration_limit=1.0, steering_limit=PI/4, velocity_limit=10.0,
                max_angle_error=PI/12, center_threshold=1.0,
                reward_type='type1', state_type='type1',
                side=(1, 2, 3, 4), car_loc_randomize_range=(-5, 5), initial_distance_range=(7.5, 15.0)
                )
env_config = {"render_mode": "no_render",
              "action_type": "continuous",
              "parking_type": "perpendicular",
              "training_mode": "on",
              'config': config}


# for folder names
num_train = "100"
folder_path = create_folder_path(env_config, config.reward_type, config.state_type, is_training=True)
if config.state_type == 'type1':
    folder_name = env_config["parking_type"] + "_" + env_config["action_type"] + "_" + num_train

if config.state_type == 'type2':
    guidance = "guidance_15"
    folder_name = env_config["parking_type"] + "_" + env_config["action_type"] + "_" + num_train + "_" + guidance
if config.state_type == 'type3':
    ratio = "035_015"
    guidance = "guidance_15"
    folder_name = (env_config["parking_type"] + "_" + env_config["action_type"] + "_"
                  + num_train + "_" + guidance + "_" + ratio)

algo = (
    PPOConfig()
    .environment(env=env_name, env_config=env_config)
    .rollouts(num_rollout_workers=1)
    .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    .framework("torch")
    .evaluation(evaluation_num_workers=1)
    .build(logger_creator=custom_log_creator(folder_path, folder_name))
)

start_time = time.time()
# training
for i in range(int(num_train)):
    print("Iterations:", i, ":", algo.train())
end_time = time.time()
print(f"Total execution time of the script: {end_time - start_time} second")

algo.evaluate()

# save the checkpoint
checkpoint_dir = custom_log_checkpoint(folder_path, folder_name, algo)
checkpoint_dir = algo.save(checkpoint_dir)
print(f"Checkpoint saved in directory {checkpoint_dir}")
