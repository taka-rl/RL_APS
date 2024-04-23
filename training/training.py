import os
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from sim_env.parking_env import Parking
from utility import custom_log_creator, custom_log_checkpoint


ray.init()
env_name = Parking
env_config = {"render_mode": "no_render",
              "action_type": "continuous",
              "parking_type": "parallel"}
custom_str = env_config.get("parking_type")
algo = (
    PPOConfig()
    .environment(env=env_name, env_config=env_config)
    .rollouts(num_rollout_workers=1)
    .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    .framework("torch")
    .evaluation(evaluation_num_workers=1)
    .build(logger_creator=custom_log_creator(custom_str))
)

# training
for i in range(10):
    print("Iterations:", i, ":", algo.train())

algo.evaluate()

# save the checkpoint
checkpoint_dir = custom_log_checkpoint(custom_str, algo)
checkpoint_dir = algo.save(checkpoint_dir)
print(f"Checkpoint saved in directory {checkpoint_dir}")