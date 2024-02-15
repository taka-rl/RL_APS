
import gymnasium as gym
from ray.rllib import SampleBatch
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.util.client import ray
from torch.distributed.fsdp.wrap import CustomPolicy

# Setup policy and rollout workers.
env = gym.make("CartPole-v1")
policy = CustomPolicy(env.observation_space, env.action_space, {})
workers = WorkerSet(
    policy_class=CustomPolicy,
    env_creator=lambda c: gym.make("CartPole-v1"),
    num_workers=10)

while True:
    # Gather a batch of samples.
    T1 = SampleBatch.concat_samples(
        ray.get([w.sample.remote() for w in workers.remote_workers()]))

    # Improve the policy using the T1 batch.
    policy.learn_on_batch(T1)

    # The local worker acts as a "parameter server" here.
    # We put the weights of its `policy` into the Ray object store once (`ray.put`)...
    weights = ray.put({"default_policy": policy.get_weights()})
    for w in workers.remote_workers():
        # ... so that we can broacast these weights to all rollout-workers once.
        w.set_weights.remote(weights)