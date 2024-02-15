from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
from log_creator import custom_log_creator
from log_checkpoint import custom_log_checkpoint
from path_select import get_current_path

env_name = "CartPole-v1"
custom_path = get_current_path() + "/training_result/"

algo = (
    PPOConfig()
    .rollouts(num_rollout_workers=1)
    .resources(num_gpus=0)
    .environment(env=env_name)
    .framework("torch")
    .evaluation(evaluation_num_workers=1)
    .build(logger_creator=custom_log_creator(custom_path, env_name))
)

for i in range(1):
    result = algo.train()
    print(pretty_print(result))

# save the checkpoint
checkpoint_dir = custom_log_checkpoint(env_name, algo)
checkpoint_dir = algo.save(checkpoint_dir)
print(f"Checkpoint saved in directory {checkpoint_dir}")
