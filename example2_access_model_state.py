# Get a reference to the policy
import numpy as np
from ray.rllib.algorithms.dqn import DQNConfig

algo = (
    DQNConfig()
    .environment("CartPole-v1")
    .framework("tf2")
    .rollouts(num_rollout_workers=0)
    .build()
)
# <ray.rllib.algorithms.ppo.PPO object at 0x7fd020186384>

policy = algo.get_policy()
print("policy:", policy)
# <ray.rllib.policy.eager_tf_policy.PPOTFPolicy_eager object at 0x7fd020165470>

# Run a forward pass to get model output logits. Note that complex observations
# must be preprocessed as in the above code block.
logits, _ = policy.model({"obs": np.array([[0.1, 0.2, 0.3, 0.4]])})
print("logits:", logits)
# (<tf.Tensor: id=1274, shape=(1, 2), dtype=float32, numpy=...>, [])

# Compute action distribution given logits
print("policy.dist_class:", policy.dist_class)
# <class_object 'ray.rllib.models.tf.tf_action_dist.Categorical'>
dist = policy.dist_class(logits, policy.model)
print("dist:", dist)
# <ray.rllib.models.tf.tf_action_dist.Categorical object at 0x7fd02301d710>

# Query the distribution for samples, sample logps
print("dist.sample:", dist.sample())
# <tf.Tensor: id=661, shape=(1,), dtype=int64, numpy=..>
print("dist.logp:", dist.logp([1]))
# <tf.Tensor: id=1298, shape=(1,), dtype=float32, numpy=...>

# Get the estimated values for the most recent forward pass
print("policy.model.value_function():", policy.model.value_function())
# <tf.Tensor: id=670, shape=(1,), dtype=float32, numpy=...>

print("policy.model.base_model.summary():", policy.model.base_model.summary())