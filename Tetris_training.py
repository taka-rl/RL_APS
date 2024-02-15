# https://www.gymlibrary.dev/environments/toy_text/taxi/

'''
add display for action,reward, environment
# https://docs.ray.io/en/latest/rllib/rllib-env.html

'''

from ray.rllib.algorithms.ppo import PPOConfig
from log_creator import custom_log_creator

env_name = "ALE/Tetris-v5"
custom_path = "C:/Users/is12f/Documents/programming/pythonProject/trained_agent"

config = (  # 1. Configure the algorithm,
    PPOConfig()
    .environment(env=env_name)
    .rollouts(num_rollout_workers=2)
    .framework("torch")
    .training(model={"fcnet_hiddens": [64, 64]})
    .evaluation(evaluation_num_workers=1)
)
# algo = config.build(logger_creator=custom_log_creator(custom_path, env_name))  # 2. build the algorithm,
algo = config.build()

for i in range(10):
    print("Iterations:", i, ":", algo.train())  # 3. train it,

algo.evaluate()  # 4. and evaluate it.


'''
need to make a function or use custom_log_creator function
to select a folder for the checkpoint result

checkpoint_dir = custom_log_checkpoint(custom_path, env_name)
checkpoint_dir = custom_path + "/" + env_name
checkpoint_dir = algo.save(checkpoint_dir)
'''
checkpoint_dir = algo.save().checkpoint.path
print(f"Checkpoint saved in directory {checkpoint_dir}")
