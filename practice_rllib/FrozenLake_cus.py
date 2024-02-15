'''
# https://gymnasium.farama.org/environments/toy_text/frozen_lake/

to learn the code in Ray module
https://applied-rl-course.netlify.app/en/module3

modify the random start position and the goal

'''

import gymnasium as gym
# from.gymnasium.wrappers import TimeLimit
import numpy as np
from ray.rllib.algorithms.ppo import PPOConfig

class FrozenPond(gym.Env):
    def __init__(self, env_config=None):
        self.observation_space = gym.spaces.Discrete(16)
        self.action_space = gym.spaces.Discrete(4)

    def reset(self):
        self.player = (0,0) # the player
        self.goal = (3,3) # goal is at the bottom-right

        self.holes = np.array([
            [0,0,0,0], # FFFF
            [0,1,0,1], # FHFH
            [0,0,0,1], # FFFH
            [1,0,0,0]  # HFFF
        ])
        self.stepcount = 0

        return 0 # to be changed to return self.observation()

    def observation(self):
        return 4*self.player[0] + self.player[1]

    def reward(self):
        return int(self.player == self.goal)

    def done(self):
        return self.player == self.goal or self.holes[self.player] == 1

    def is_valid_loc(self, location):
        if 0 <= location[0] <= 3 and 0 <= location[1] <= 3:
            return True
        else:
            return False

    def step(self, action):
        # Compute the new player location
        if action == 0: # left
            new_loc = (self.player[0], self.player[1]-1)
        elif action == 1: # down
            new_loc = (self.player[0]+1, self.player[1])
        elif action == 2: # right
            new_loc = (self.player[0], self.player[1]+1)
        elif action == 3: # up
            new_loc = (self.player[0]-1, self.player[1])
        else:
            raise ValueError("Action must be in {0, 1, 2, 3}")

        # update the player location only if you stayed in bounds
        # (if you try to move out of bounds, the action does nothing)
        if self.is_valid_loc(new_loc):
            self.player = new_loc

        # Return observation/reward/done
        return self.observation(), self.reward(), self.done(), {}

    def render(self):
        for i in range(4):
            for j in range(4):
                if (i,j) == self.goal:
                    print("⛳️", end="")
                elif (i,j) == self.player:
                    print("🧑", end="")
                elif self.holes[i,j]:
                    print("🕳", end="")
                else:
                    print("🧊", end="")
            print()



pond = FrozenPond()
pond.reset()
pond.render()

lake_default_config = (
    PPOConfig()
    .environment(env=FrozenPond)
    .rollouts(num_rollout_workers=2)
    .framework("torch")
    .training(model={"fcnet_hiddens": [64, 64]})
    .evaluation(evaluation_num_workers=1)
    .build()
)

ppo = lake_default_config
for i in range(8):
    ppo.train() # more than 90%
# ppo.train() # less than 20%
print(ppo.evaluate()["evaluation"]["episode_reward_mean"])
