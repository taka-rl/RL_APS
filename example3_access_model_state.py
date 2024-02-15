# Get a reference to the model through the policy
import numpy as np
from ray.rllib.algorithms.dqn import DQNConfig

algo = DQNConfig().environment("CartPole-v1").framework("tf2").build()
model = algo.get_policy().model
# <ray.rllib.models.catalog.FullyConnectedNetwork_as_DistributionalQModel ...>

# List of all model variables
print("model.variables():", model.variables())

# Run a forward pass to get base model output. Note that complex observations
# must be preprocessed. An example of preprocessing is examples/saving_experiences.py
model_out = model({"obs": np.array([[0.1, 0.2, 0.3, 0.4]])})
print("model_out:", model_out)
# (<tf.Tensor: id=832, shape=(1, 256), dtype=float32, numpy=...)

# Access the base Keras models (all default models have a base)
print("model.base_model.summary():", model.base_model.summary())

"""
Model: "model"
_______________________________________________________________________
Layer (type)                Output Shape    Param #  Connected to
=======================================================================
observations (InputLayer)   [(None, 4)]     0
_______________________________________________________________________
fc_1 (Dense)                (None, 256)     1280     observations[0][0]
_______________________________________________________________________
fc_out (Dense)              (None, 256)     65792    fc_1[0][0]
_______________________________________________________________________
value_out (Dense)           (None, 1)       257      fc_1[0][0]
=======================================================================
Total params: 67,329
Trainable params: 67,329
Non-trainable params: 0
______________________________________________________________________________
"""

# Access the Q value model (specific to DQN)
print(model.get_q_value_distributions(model_out)[0])
# tf.Tensor([[ 0.13023682 -0.36805138]], shape=(1, 2), dtype=float32)
# ^ exact numbers may differ due to randomness

print(model.q_value_head.summary())

# Access the state value model (specific to DQN)
print(model.get_state_value(model_out))
# tf.Tensor([[0.09381643]], shape=(1, 1), dtype=float32)
# ^ exact number may differ due to randomness

print(model.state_value_head.summary())