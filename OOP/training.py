import os
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from parking_env_OOP import Parking


ray.init()

env_config = {"render_mode": "human",
              "action_type": "continuous",
              "parking_type": "parallel"}

config = (
        PPOConfig()
        .environment(env=Parking, env_config=env_config)
        .rollouts(num_rollout_workers=1)
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
        .framework("torch")
    )

algo = config.build()

while True:
    print(algo.train())