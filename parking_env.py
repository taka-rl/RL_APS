import numpy as np
import random
import pygame
import time
import gymnasium as gym
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from typing import Optional


# parameters for actions
PI = np.pi
ACCELERATION_LIMIT = 1.0
STEERING_LIMIT = PI / 4
VELOCITY_LIMIT = 10.0

# parameters for rendering the simulation environment
FPS = 30
COLORS = {
    "RED": (255, 100, 100),
    "GREEN": (0, 255, 0),
    "BLUE": (100, 200, 255),
    "YELLOW": (200, 200, 0),
    "BLACK": (0, 0, 0),
    "GREY": (100, 100, 100),
    "WHITE": (255, 255, 255),
    "GRID_COLOR": (200, 200, 200)
}
GRID_SIZE = 20
WINDOW_W, WINDOW_H = 800, 600

# parameters for cars in the parking environment
CAR_L = 80
CAR_W = 40
CAR_STRUCT = np.array([[+CAR_L / 2, +CAR_W / 2],
                       [+CAR_L / 2, -CAR_W / 2],
                       [-CAR_L / 2, -CAR_W / 2],
                       [-CAR_L / 2, +CAR_W / 2]],
                      np.int32)
WHEEL_L = 15
WHEEL_W = 7
WHEEL_STRUCT = np.array([[+WHEEL_L / 2, +WHEEL_W / 2],
                         [+WHEEL_L / 2, -WHEEL_W / 2],
                         [-WHEEL_L / 2, -WHEEL_W / 2],
                         [-WHEEL_L / 2, +WHEEL_W / 2]],
                        np.int32)
WHEEL_POS = np.array([[25, 15], [25, -15], [-25, 15], [-25, -15]])
DT = 1

MAX_STEPS = 300


def kinematic_act(action, loc, v, psi, DT):
    """
    Parameters:
        action(list): [a, δ]: a is acceleration, δ(delta) is steering angle.
        loc: the center of the car (x,y) location
        v : velocity
        ψ(psi): the heading angle of the car
        DT: time step

    Kinematic bicycle model:
    x_dot = v * np.cos(psi)
    y_dot = v * np.sin(psi)
    v_dot = a
    psi_dot = v * np.tan(delta) / CAR_L
    """

    state = np.array([loc[0], loc[1], v, psi])
    x_dot = v * np.cos(psi)
    y_dot = v * np.sin(psi)
    v_dot = action[0]
    psi_dot = v * np.tan(action[1]) / CAR_L
    state_dot = np.array([x_dot, y_dot, v_dot, psi_dot]).T
    state = update_state(state, state_dot, DT)
    return state[:2], state[2], state[3]


def update_state(state, state_dot, dt):
    state += dt * state_dot
    state[2] = np.clip(state[2], -VELOCITY_LIMIT, VELOCITY_LIMIT)
    return state


def draw_car(screen, car_loc, psi, delta):
    """
    Parameters:
        screen: pygame.Surface
        car_loc: the center of the car (x,y) location
        psi: the heading angle of the car
        delta: the steering angle
    """
    # the car(agent)
    # get the car vertices
    car_vertices = get_car_vertices(car_loc, psi)
    # draw the car(agent)
    pygame.draw.polygon(screen, COLORS["GREEN"], car_vertices)

    # wheels
    # calculate the rotation of the wheels
    wheel_points = rotate_car(WHEEL_POS, angle=psi)
    # draw each wheel
    for i, wheel_point in enumerate(wheel_points):
        if i < 2:
            wheel_vertices = rotate_car(WHEEL_STRUCT, angle=psi + delta)
        else:
            wheel_vertices = rotate_car(WHEEL_STRUCT, angle=psi)
        wheel_vertices += wheel_point + car_loc
        pygame.draw.polygon(screen, COLORS["RED"], wheel_vertices)


def rotate_car(pos, angle=0):
    R = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ])
    rotated_pos = (R @ pos.T).T
    return rotated_pos

def get_car_vertices(car_loc, psi):
    # calculate the rotation of the car itself
    car_vertices = rotate_car(CAR_STRUCT, angle=psi)
    # add the center of the car x,y location
    car_vertices += car_loc
    return car_vertices


def is_valid_loc(loc, width, height):
    if loc[0] < 0 or loc[0] > width or loc[1] < 0 or loc[1] > height:
        return True
    else:
        return False


def is_parking_successful(loc, parking_loc, psi):
    car_vertices = get_car_vertices(loc, psi)
    pa_top_right, pa_bottom_right, pa_bottom_left, pa_top_left = parking_loc

    # Define the edges of the parking area
    pa_left_edge = pa_top_left[0]
    pa_right_edge = pa_top_right[0]
    pa_top_edge = pa_top_left[1]
    pa_bottom_edge = pa_bottom_left[1]

    # Check if all car corners are within the parking area
    for corner in car_vertices:
        if not (pa_left_edge <= corner[0] <= pa_right_edge and
                pa_bottom_edge <= corner[1] <= pa_top_edge):
            return False
    return True


def set_random_loc():
    x = random.uniform(10, WINDOW_W-100)
    y = random.uniform(10, WINDOW_H-100)
    loc = [x,y]
    return loc


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

        if action_type == "continuous":
            self.action_space = gym.spaces.Box(
                np.array([-ACCELERATION_LIMIT, -STEERING_LIMIT]),
                np.array([ACCELERATION_LIMIT, STEERING_LIMIT]),
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
            action(list): [a, δ]: a is acceleration, δ(delta) is steering angle.

        Returns:
            state (list): velocity, the 4 corner points of the parking area
            reward:
            terminated:
            truncated:
        """
        if action is not None:
            if self.action_type == "continuous":
                action = np.clip(action, [-1, -1], [1, 1]) * [
                    ACCELERATION_LIMIT,
                    STEERING_LIMIT,
                ]
                self.loc_old = self.loc
                self.loc, self.velocity, self.psi = kinematic_act(action, self.loc, self.velocity, self.psi, DT)
                self.delta = action[1]

            reward = self._reward()
            self.state = [self.velocity, self.parking_lot - self.loc]

        if self.render_mode == "human":
            self.render()

        return self.state, reward, self.terminated, self.truncated, {}

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
                self.surf_parkinglot.fill(COLORS["WHITE"])

                # Draw the grid lines
                for x in range(0, WINDOW_W, GRID_SIZE):
                    pygame.draw.line(self.surf_parkinglot, COLORS["GRID_COLOR"], (x, 0), (x, WINDOW_H), 1)
                for y in range(0, WINDOW_H, GRID_SIZE):
                    pygame.draw.line(self.surf_parkinglot, COLORS["GRID_COLOR"], (0, y), (WINDOW_W, y), 1)

                # Draw the targeted parking space
                pygame.draw.polygon(self.surf_parkinglot, COLORS["YELLOW"], self.parking_lot)

            # for the car(agent)
            if self.surf_car is None:
                self.surf_car = pygame.Surface(
                    (WINDOW_W, WINDOW_H), flags=pygame.SRCALPHA
                )
            self.surf_car.fill((0, 0, 0, 0))

            # draw the car(agent) movement
            draw_car(self.surf_car, self.loc, self.psi, self.delta)

            # draw the car path
            # pygame.draw.line(self.surf_parkinglot, BLACK, self.loc_old, self.loc)

            surf = self.surf_parkinglot.copy()
            surf.blit(self.surf_car, (0, 0))
            surf = pygame.transform.flip(surf, False, True)

        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            assert self.window is not None
            self.window.fill(COLORS["BLACK"])
            self.window.blit(surf, (0, 0))
            pygame.display.flip()

    def reset(
            self,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):
        super().reset(seed=seed)

        self.loc = set_random_loc()
        self.velocity = 0
        self.psi = 0
        self.loc_old = self.loc
        self.delta = 0
        self.parking_lot = set_random_loc() + np.array([
                                [+CAR_L / 2 + 10, +CAR_W / 2 + 10],
                                [+CAR_L / 2 + 10, -CAR_W / 2],
                                [-CAR_L / 2, -CAR_W / 2],
                                [-CAR_L / 2, +CAR_W / 2 + 10]],
                                np.int32)
        self.state = [self.velocity, self.parking_lot - self.loc]
        self.terminated = False
        self.truncated = False
        self.run_steps = 0
        self.reward = 0

        self.window = None
        self.surf = None
        self.surf_car = None
        self.surf_parkinglot = None
        self.clock = None

        return self.state, {}

    def _reward(self):
        self.run_steps += 1

        if self.run_steps == MAX_STEPS:
            self.truncated = True

        # check the location
        if is_valid_loc(self.loc, WINDOW_W, WINDOW_H):
            self.reward -= 1
            self.terminated = True
            print("The car is not a valid location")

        # check the parking
        if is_parking_successful(self.loc, self.parking_lot, self.psi):
            self.reward += 1
            self.truncated = True
            print("successful parking")

        return self.reward

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None


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
        action = env.action_space.sample()  # env.action_space.sample() is to sample random actions.
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        time.sleep(0.1)
        episode_reward += reward
        print("Episode reward:", episode_reward)