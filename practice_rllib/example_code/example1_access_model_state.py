# Example: Preprocessing observations for feeding into a model

import gymnasium as gym

env = gym.make("ALE/Pong-v5")
obs, info = env.reset()
print(obs)

# RLlib uses preprocessors to implement transforms such as one-hot encoding
# and flattening of tuple and dict observations.
from ray.rllib.models.preprocessors import get_preprocessor

prep = get_preprocessor(env.observation_space)(env.observation_space)
print(prep)
# <ray.rllib.models.preprocessors.GenericPixelPreprocessor object at 0x7fc4d049de80>

# Observations should be preprocessed prior to feeding into a model
print(obs.shape)
# (210, 160, 3)
print(prep.transform(obs).shape)
# (84, 84, 3)