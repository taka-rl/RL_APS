import numpy as np
import pygame
import time
import gymnasium as gym
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from typing import Optional


'''
## todo
- general
Update parameters to tailor this script
Implement Ray RLlib algorithm

# https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/#registering-envs
- step function
Consider a maximum/minimum steering angle and Vehicle speed
Add kinematic bicycle model
Consider a collision function, a function which can check if the parking is success or not

- render function
Learn how to use Pygame
Consider the parking environemnt and car size and so on
Render the environment
Draw the car path

- reset function
reset the parking environment
reset car location, reward and so on

- reward function
Consider reward functions
Implement them

'''


# parameters
PI = np.pi
STEERING_LIMIT = PI / 4
SPEED_LIMIT = 4.0
N_SAMPLES_LAT_ACTION = 5
LON_ACTIONS = np.array([-SPEED_LIMIT, SPEED_LIMIT])
LAT_ACTIONS = np.linspace(-STEERING_LIMIT, STEERING_LIMIT, N_SAMPLES_LAT_ACTION)
SCREEN_H = 400
SCREEN_W = 500

SCALE = 8

STATE_SCALE = np.array(
    [SCREEN_W / SCALE, SCREEN_H / SCALE, 1, 1, SPEED_LIMIT]
    + [SCREEN_W / SCALE, SCREEN_H / SCALE] * 4
)

# position[2], heading, speed, length, width, type
INIT_STATE = np.array([0, 10, -PI, 4, 4.8, 1.8, 0])
GOAL_STATE = np.array([7 * 3 / 2, -20, 0, 0, 2, 2, 1])
STATIONARY_STATE = np.array(
    [
        [0, 0, 0, 0, SCREEN_W / SCALE / 2, 0.5, 1],
        [0, (SCREEN_H - 3) / SCALE / 2, 0, 0, SCREEN_W / SCALE, 1, 1],
        [0, -SCREEN_H / SCALE / 2, 0, 0, SCREEN_W / SCALE, 1, 1],
        [(SCREEN_W - 3) / SCALE / 2, 0, 0, 0, 1, SCREEN_H / SCALE, 1],
        [-SCREEN_W / SCALE / 2, 0, 0, 0, 1, SCREEN_H / SCALE, 1],
        [-5 * 3 / 2, -20, PI / 2, 0, 4.8, 1.8, 0],
        [-3 * 3 / 2, -20, PI / 2, 0, 4.8, 1.8, 0],
        [-5 * 3 / 2, -5, -PI / 2, 0, 4.8, 1.8, 0],
        [7 * 3 / 2, -5, -PI / 2, 0, 4.8, 1.8, 0],
        [-3 / 2, 5, PI / 2, 0, 4.8, 1.8, 0],
        [-3 * 3 / 2, 20, -PI / 2, 0, 4.8, 1.8, 0],
        [7 * 3 / 2, 20, -PI / 2, 0, 4.8, 1.8, 0],
    ]
)

FPS = 30
RED = (255, 100, 100)
GREEN = (50, 200, 0)
BLUE = (100, 200, 255)
YELLOW = (200, 200, 0)
BLACK = (0, 0, 0)
GREY = (100, 100, 100)

class Parking(gym.Env):

    metadata = {
        "render_modes": ["human", "rgb_array", "no_render"],
        "render_fps": FPS,
        "action_types": ["discrete", "continuous"]
    }

    def __init__(
            self,
            render_mode: Optional[str] = None,
            action_type: Optional[str] = None,

    ) -> None:
        super().__init__()
        assert render_mode in self.metadata["render_modes"]
        assert action_type in self.metadata["action_types"]
        self.render_mode = render_mode
        self.action_type = action_type
        self.observation_space = gym.spaces.Box(-1, 1, dtype=np.float32)

        if action_type == "discrete":
            self.action_space = gym.spaces.Discrete(N_SAMPLES_LAT_ACTION)

        if action_type == "continuous":
            self.action_space = gym.spaces.Box(
                np.array([-SPEED_LIMIT, -STEERING_LIMIT]),
                np.array([SPEED_LIMIT, STEERING_LIMIT]),
                dtype=np.float32,
            )

        self.screen = None
        self.surf = None
        self.surf_movable = None
        self.surf_stationary = None
        self.clock = None

    def step(self, action):
        if action is not None:
            if self.action_type == "discrete":
                action = np.array([SPEED_LIMIT, LAT_ACTIONS[action]])
            if self.action_type == "continuous":
                action = np.clip(action, -1, 1) * STEERING_LIMIT
                action = np.array([SPEED_LIMIT, action.item()])

            # add kinematic bicycle model later
            self.obs = 1 # temporary

            reward = self._reward()

        if self.render_mode == "human":
            self.render()

        return self.obs, reward, self.terminated, self.truncated, {}

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return
        else:
            return self._render(self.render_mode)

    def _render(self, mode:str, SCREEN_W, SCREEN_H):
        if mode == "human" and self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
            if self.clock is None:
                self.clock = pygame.time.Clock()

        if mode == "human" or mode == "rgb_array":
            if self.surf_stationary is None:
                self.surf_stationary = pygame.Surface(
                    (SCREEN_W, SCREEN_H), flags=pygame.SRCALPHA
                )
                #for i in range(self.stationary.shape[0]):
                    #draw_rectangle(
                    #    self.surf_stationary,
                    #    to_pixel(self.stationary_vertices[i]),
                    #    obj_type=self.stationary[i, -1],
                    #)
                    # draw_direction_pattern(self.surf_stationary, self.stationary[i])
                # draw_rectangle(self.surf_stationary, to_pixel(self.goal_vertices), BLUE)

            if self.surf_movable is None:
                self.surf_movable = pygame.Surface(
                    (SCREEN_W, SCREEN_H), flags=pygame.SRCALPHA
                )
            self.surf_movable.fill((0, 0, 0, 0))
            # draw_rectangle(self.surf_movable, to_pixel(self.movable_vertices), GREEN)
            # draw_direction_pattern(self.surf_movable, self.movable[0])

            surf = self.surf_stationary.copy()
            surf.blit(self.surf_movable, (0, 0))
            surf = pygame.transform.flip(surf, False, True)

        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            assert self.screen is not None
            self.screen.fill(BLACK)
            self.screen.blit(surf, (0, 0))
            pygame.display.flip()
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):

        super().reset(seed=seed)
        self.stationary = STATIONARY_STATE
        self.stationary_vertices = np.zeros((self.stationary.shape[0], 4, 2))
        #for i in range(self.stationary.shape[0]):
        #    self.stationary_vertices[i] = compute_vertices(self.stationary[i])

        self.movable = np.array([randomise_state(INIT_STATE)])
        self.movable_vertices = compute_vertices(self.movable[0])
        #self.movable = np.array([randomise_state(INIT_STATE)])
        #self.movable_vertices = compute_vertices(self.movable[0])
        #while collision_check(
        #       self.movable_vertices,
        #        self.movable[0, 2],
        #        self.stationary_vertices,
        #        self.stationary[:, 2],
        #):
        #    self.movable = np.array([randomise_state(INIT_STATE)])
        #    self.movable_vertices = compute_vertices(self.movable[0])
        #self.goal_vertices = compute_vertices(GOAL_STATE)

        #if self.observation_type == "vector":
        self.obs = np.zeros(
                (self.movable.shape[0] + self.stationary.shape[0] + 1, 13),
                dtype=np.float32,
            )
        self.obs[1, :2] = GOAL_STATE[:2]
        self.obs[1, 2:5] = [
                np.cos(GOAL_STATE[2]),
                np.sin(GOAL_STATE[2]),
                GOAL_STATE[3],
            ]
        self.obs[1, 5:] = self.goal_vertices.reshape(1, 8)
        self.obs[2:, :2] = self.stationary[:, :2]
        self.obs[2:, 2] = np.cos(self.stationary[:, 2])
        self.obs[2:, 3] = np.sin(self.stationary[:, 2])
        self.obs[2:, 4] = self.stationary[:, 3]
        self.obs[2:, 5:] = self.stationary_vertices.reshape(-1, 8)
        self.obs /= STATE_SCALE

        self.terminated = False
        self.truncated = False
        self.run_steps = 0

        if self.render_mode == "human":
            self.render()
        return self.step(None)[0], {}

    def _reward(self):

        # add reward later
        reward = 0
        return reward

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None


if __name__ == "__main__":
    # ray.init()

    env = Parking(render_mode="human", action_type="continuous")

    #algo = (
    #    PPOConfig()
    #    .environment(env=env)
    #    .rollouts(num_rollout_workers=2)
    #    .resources(num_gpus=0)
    #    .framework("torch")
    #    .evaluation(evaluation_num_workers=1)
    #    .build()
    #)

    episode_reward = 0
    terminated = truncated = False
    obs, info = env.reset()
    env.render()
    while not terminated and not truncated:
        # action = algo.compute_single_action(obs)  # Algorithm.compute_single_action() is to programmatically compute actions from a trained agent.
        action = env.action_space.sample() # env.action_space.sample() is to sample random actions.
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        time.sleep(0.1)
        episode_reward += reward
        print("Episode reward:", episode_reward)

