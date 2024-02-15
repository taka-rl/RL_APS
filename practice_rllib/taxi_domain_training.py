# https://www.gymlibrary.dev/environments/toy_text/taxi/

'''
add display for action,reward, environment
# https://docs.ray.io/en/latest/rllib/rllib-env.html

'''

from ray.rllib.algorithms.ppo import PPOConfig
from log_creator import custom_log_creator
from log_checkpoint import custom_log_checkpoint
from path_select import get_current_path

env_name = "Taxi-v3"
custom_path = get_current_path() + "/training_result/"

config = (  # 1. Configure the algorithm,
    PPOConfig()
    .environment(env=env_name)
    .rollouts(num_rollout_workers=2)
    .framework("torch")
    .training(model={"fcnet_hiddens": [64, 64]})
    # .training(model={"fcnet_hiddens": [128, 128]})
    .evaluation(evaluation_num_workers=1)
)
algo = config.build(logger_creator=custom_log_creator(custom_path, env_name))  # 2. build the algorithm,
# algo = config.build()

for i in range(50):
    print("Iterations:", i, ":", algo.train())  # 3. train it,

algo.evaluate()  # 4. and evaluate it.

# save the checkpoint
checkpoint_dir = custom_log_checkpoint(env_name, algo)
checkpoint_dir = algo.save(checkpoint_dir)
print(f"Checkpoint saved in directory {checkpoint_dir}")

