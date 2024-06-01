import os
import ray
import time
from ray.rllib.algorithms.ppo import PPOConfig
from sim_env.parking_env import Parking
from utility import custom_log_creator, custom_log_checkpoint


ray.init()
env_name = Parking
env_config = {"render_mode": "no_render",
              "action_type": "continuous",
              "parking_type": "perpendicular",
              "training_mode": "on"}

# for folder names
num_train = "500"
guidance = "guidance_15"
custom_str = env_config["parking_type"] + "_" + env_config["action_type"] + "_" + num_train + "_" + guidance

algo = (
    PPOConfig()
    .environment(env=env_name, env_config=env_config)
    .rollouts(num_rollout_workers=1)
    .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    .framework("torch")
    .evaluation(evaluation_num_workers=1)
    .build(logger_creator=custom_log_creator(custom_str, env_config))
)

start_time = time.time()
# training
for i in range(int(num_train)):
    print("Iterations:", i, ":", algo.train())
end_time = time.time()
print(f"Total execution time of the script: {end_time - start_time} second")

algo.evaluate()

# save the checkpoint
checkpoint_dir = custom_log_checkpoint(custom_str, env_config, algo)
checkpoint_dir = algo.save(checkpoint_dir)
print(f"Checkpoint saved in directory {checkpoint_dir}")