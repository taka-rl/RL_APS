from parking_env_OOP import Parking
import ray
from ray.rllib.algorithms import ppo


ray.init()
env_config = dict(render_mode="human", action_type="continuous", parking_type="parallel")
algo = ppo.PPO(env=Parking(env_config))
while True:
    print(algo.train())