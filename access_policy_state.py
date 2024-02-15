from ray.rllib.algorithms.dqn import DQNConfig

algo = DQNConfig().environment(env="CartPole-v1").build()

# Get weights of the default local policy
print(algo.get_policy().get_weights())

# Same as above
print(algo.workers.local_worker().policy_map["default_policy"].get_weights())

# Get list of weights of each worker, including remote replicas
print(algo.workers.foreach_worker(lambda worker: worker.get_policy().get_weights()))

# Same as above, but with index.
print(algo.workers.foreach_worker_with_id(
    lambda _id, worker: worker.get_policy().get_weights()
))