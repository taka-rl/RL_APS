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
Consider a collision function, a function which can check if the parking is success or not
Consider obs information

- render function
Consider the parking environemnt and car size and so on
Draw the car path

- reset function
reset the parking environment
reset car location, reward and so on

- reward function
Consider reward functions
Implement them

'''

# parameters for actions
PI = np.pi
STEERING_LIMIT = PI / 4
SPEED_LIMIT = 4.0
N_SAMPLES_LAT_ACTION = 5
LON_ACTIONS = np.array([-SPEED_LIMIT, SPEED_LIMIT])
LAT_ACTIONS = np.linspace(-STEERING_LIMIT, STEERING_LIMIT, N_SAMPLES_LAT_ACTION)

# parameters for rendering the simulation environment
FPS = 30
RED = (255, 100, 100)
GREEN = (0, 255, 0)
BLUE = (100, 200, 255)
YELLOW = (200, 200, 0)
BLACK = (0, 0, 0)
GREY = (100, 100, 100)
WHITE = (255, 255, 255)
GRID_SIZE = 20
GRID_COLOR = (200, 200, 200)
WINDOW_W, WINDOW_H = 800, 600

# parameters for cars in the parking environment
CAR_L = 80
CAR_W = 40
WHEEL_L = 15
WHEEL_W = 7
WHEEL_POS = np.array([[25, 15], [25, -15], [-25, 15], [-25, -15]])
PARKINGLOT_LOC = [300, 10, 90, 50]
DT = 1

MAX_STEPS = 200


def kinematic_act(action, state, DT):
    """
    action: [0]:v, [1]:delta
    state: [x,y,v,psi]

    Kinematic bicycle model
    x_dot = v * np.cos(psi)
    y_dot = v * np.sin(psi)
    v_dot = v
    psi_dot = v * np.tan(delta) / CAR_L
    or
    x_dot = v * cos(psi + beta)
    y_dot = v * sin(psi + beta)
    psi_dot = v * cos(beta) * tan(delta)/car length
    beta = tan^-1(Lr/Lf+Lr * tan(delta))
    """

    x_dot = action[0] * np.cos(state[3])
    y_dot = action[0] * np.sin(state[3])
    v_dot = state[2]
    psi_dot = action[0] * np.tan(action[1]) / CAR_L
    state_dot = np.array([x_dot, y_dot, v_dot, psi_dot]).T
    state = update_state(state, state_dot, DT)
    return state


def update_state(state, state_dot, dt):
    state[0] += dt * state_dot[0]
    state[1] += dt * state_dot[1]
    state[2] = state_dot[2]
    state[3] += dt * state_dot[3]
    return state


def draw_vehicle(screen, car_loc, psi):
    # the car(agent)
    # get the car(agent) vertexes from the center point
    car_vertices = compute_vertices(car_loc, CAR_L, CAR_W)
    # get the rotated car location
    car_vertices = rotate_car(car_vertices, angle=psi)
    # draw the car(agent)
    pygame.draw.polygon(screen, GREEN, car_vertices)

    # wheels
    # get the center of each wheel point
    wheel_points = compute_wheel_points(car_loc)
    # draw each wheel
    for i, wheel_point in enumerate(wheel_points):
        wheel_vertices = compute_vertices(wheel_point, WHEEL_L, WHEEL_W)
        if i < 2:
            # add delta(steering angle) into the front wheels
            wheel_vertices = rotate_car(wheel_vertices, angle=psi)
        else:
            wheel_vertices = rotate_car(wheel_vertices, angle=psi)
        pygame.draw.polygon(screen, RED, wheel_vertices)


def compute_vertices(loc, length, width):
    vertices = np.array([
        [loc[0] + length / 2, loc[1] + width / 2],
        [loc[0] + length / 2, loc[1] - width / 2],
        [loc[0] - length / 2, loc[1] - width / 2],
        [loc[0] - length / 2, loc[1] + width / 2]
    ])
    return vertices


def compute_wheel_points(car_loc):
    wheel_points = np.array([
        [car_loc[0] + WHEEL_POS[0, 0], car_loc[1] + WHEEL_POS[0, 1]],
        [car_loc[0] + WHEEL_POS[1, 0], car_loc[1] + WHEEL_POS[1, 1]],
        [car_loc[0] + WHEEL_POS[2, 0], car_loc[1] + WHEEL_POS[2, 1]],
        [car_loc[0] + WHEEL_POS[3, 0], car_loc[1] + WHEEL_POS[3, 1]]
    ])
    return wheel_points


def rotate_car(pos, angle=0):
    R = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ])
    rotated_pos = (R @ pos.T).T

    return rotated_pos


class Parking(gym.Env):
    """
    A Gymnasium environment for the parking simulation.

    Attributes:
        render_modes (list): List of rendering modes including "human", "rgb_array", "no_render".
        action_types (list): List of action types including "discrete", "continuous".
        window: A reference to the Pygame window to render the environment.
        surf: A surface object used for rendering graphics.
        surf_car: A surface object representing the car(agent) in the environment.
        surf_parkinglot: A surface object representing the parking lot in the environment
        clock: An object representing the game clock for managing time in the environment.
    """

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
        """
        Initializes a parking instance.

        Parameters:
            render_modes: the drawing mode for visualization
            action_types: the type of action for the agent
        """
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

        self.window = None
        self.surf = None
        self.surf_car = None
        self.surf_parkinglot = None
        self.clock = None

    def step(self, action):
        """
        Let the car(agent) take an action in the parking environment.

        Parameters:
            action(list): [𝑣, 𝜑], 𝑣 is velocity, 𝜑 is steering angle.
            state(list): [x,y,v,psi]

        Returns:
            obs (list):
        """
        if action is not None:
            if self.action_type == "discrete":
                action = np.array([SPEED_LIMIT, LAT_ACTIONS[action]])
            if self.action_type == "continuous":
                action = np.clip(action, [-1, -1], [1, 1]) * [
                    SPEED_LIMIT,
                    STEERING_LIMIT,
                ]

                # calculate by Kinematic model
                self.state = kinematic_act(action, self.state, DT)

            # update observation
            # the current vehicle info, goal info
            self.obs = 1  # temporary

            # self.obs[] = state

            reward = self._reward()

        if self.render_mode == "human":
            self.render()

        return self.obs, reward, self.terminated, self.truncated, {}

    def render(self):
        """
        Draw the parking environment.

        """
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return
        else:
            return self._render(self.render_mode, WINDOW_W, WINDOW_H)

    def _render(self, mode: str, WINDOW_W, WINDOW_H):
        if mode == "human" and self.window is None:
            # Initialize the parking environment window
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((WINDOW_W, WINDOW_H))
            pygame.display.set_caption("Parking Environment")
            if self.clock is None:
                self.clock = pygame.time.Clock()

        # for the parking lot
        if mode == "human" or mode == "rgb_array":
            if self.surf_parkinglot is None:
                self.surf_parkinglot = pygame.Surface(
                    (WINDOW_W, WINDOW_H), flags=pygame.SRCALPHA
                )

                # Clear the screen
                self.surf_parkinglot.fill(WHITE)

                # Draw the grid lines
                for x in range(0, WINDOW_W, GRID_SIZE):
                    pygame.draw.line(self.surf_parkinglot, GRID_COLOR, (x, 0), (x, WINDOW_H), 1)
                for y in range(0, WINDOW_H, GRID_SIZE):
                    pygame.draw.line(self.surf_parkinglot, GRID_COLOR, (0, y), (WINDOW_W, y), 1)

                # Draw the targeted parking space
                pygame.draw.rect(self.surf_parkinglot, YELLOW, PARKINGLOT_LOC)  # [X, Y, Width, Height]

            # for the car(agent)
            if self.surf_car is None:
                self.surf_car = pygame.Surface(
                    (WINDOW_W, WINDOW_H), flags=pygame.SRCALPHA
                )
            self.surf_car.fill((0, 0, 0, 0))

            # draw the car(agent) movement
            draw_vehicle(self.surf_car, self.state[0:2], self.state[3])

            surf = self.surf_parkinglot.copy()
            surf.blit(self.surf_car, (0, 0))
            surf = pygame.transform.flip(surf, False, True)

        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            assert self.window is not None
            self.window.fill(BLACK)
            self.window.blit(surf, (0, 0))
            pygame.display.flip()

    def reset(
            self,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):
        super().reset(seed=seed)

        self.state = [300, 150, 4, 0]
        self.window = None
        self.surf = None
        self.surf_car = None
        self.surf_parkinglot = None
        self.clock = None

        self.terminated = False
        self.truncated = False
        self.run_steps = 0
        obs = 0
        return obs, {}

    def _reward(self):
        self.run_steps += 1

        if self.run_steps == MAX_STEPS:
            self.truncated = True

        # add reward later
        reward = 0
        return reward

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None


if __name__ == "__main__":
    # ray.init()

    env = Parking(render_mode="human", action_type="continuous")

    # algo = (
    #    PPOConfig()
    #    .environment(env=env)
    #    .rollouts(num_rollout_workers=2)
    #    .resources(num_gpus=0)
    #    .framework("torch")
    #    .evaluation(evaluation_num_workers=1)
    #    .build()
    # )

    episode_reward = 0
    terminated = truncated = False
    obs, info = env.reset()
    env.render()
    while not terminated and not truncated:
        # action = algo.compute_single_action(obs)  # Algorithm.compute_single_action() is to programmatically compute actions from a trained agent.
        action = env.action_space.sample()  # env.action_space.sample() is to sample random actions.
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        time.sleep(0.1)
        episode_reward += reward
        print("Episode reward:", episode_reward)
