import numpy as np
import pygame
import math
import gymnasium as gym
from typing import Optional
from sim_env.car import Car
from sim_env.com_fcn import meters_to_pixels, draw_object
from sim_env.parameters import PI
from sim_env.parking import ParallelParking, PerpendicularParking
from sim_env.init_state import set_init_position


class Parking(gym.Env):
    """
    A Gymnasium environment for the parking simulation.

    Attributes:
        render_mode (list): List of rendering modes including "human", "no_render".
        action_type (list): List of action types including "continuous".
        window: A reference to the Pygame window to render the environment.
        surf: A surface object used for rendering graphics.
        surf_car: A surface object representing the car(agent) in the environment.
        surf_parkinglot: A surface object representing the parking lot in the environment
        clock: An object representing the game clock for managing time in the environment.
    """

    metadata = {
        "render_modes": ["human", "no_render"],
        "action_types": ["continuous", "discrete"],
        "parking_types": ["parallel", "perpendicular"],
        "training_modes": ["on", "off"]
    }

    def __init__(self, env_config) -> None:
        """
        Initializes a parking instance.

        Parameters:
            env_config: contains the action type, render mode and parking type
        """
        super().__init__()

        # Check env_config
        if env_config["render_mode"] not in self.metadata["render_modes"]:
            raise ValueError(
                f"Invalid render mode: {env_config['render_mode']}. Valid options are {self.metadata['render_modes']}")

        if env_config["parking_type"] not in self.metadata["parking_types"]:
            raise ValueError(
                f"Invalid parking type: {env_config['parking_type']}. "
                f"Valid options are {self.metadata['parking_types']}")

        if env_config["action_type"] not in self.metadata["action_types"]:
            raise ValueError(
                f"Invalid action type: {env_config['action_type']}. Valid options are {self.metadata['action_types']}")

        # for training temporary
        if env_config["training_mode"] not in self.metadata["training_modes"]:
            raise ValueError(
                f"Invalid training mode: {env_config['training_mode']}. "
                f"Valid options are {self.metadata['training_modes']}"
            )
        self.training_mode = env_config["training_mode"]

        # Config setting
        self.config = env_config['config']

        # environment setting
        self.render_mode = env_config["render_mode"]
        self.parking_type = env_config["parking_type"]
        self.action_type = env_config["action_type"]
        if self.config.state_type == 'type1':
            self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)
        elif self.config.state_type == 'type2':
            self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32)
        elif self.config.state_type == 'type3':
            self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(9,), dtype=np.float32)

        # Action type
        if self.action_type == "continuous":
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        elif self.action_type == "discrete":
            self.action_space = gym.spaces.Discrete(6)

        # Rendering settings
        self.window = None
        self.surf = None
        self.surf_car = None
        self.surf_parkinglot = None
        self.surf_text = None
        self.clock = None

        # Parking type settings
        if self.parking_type == "parallel":
            self.parking_strategy = ParallelParking(self.config)
        else:
            self.parking_strategy = PerpendicularParking(self.config)

        # training
        self.state = None
        self.terminated = None
        self.truncated = None
        self.run_steps = None
        self.side = None
        self.parking_lot = None
        self.parking_lot_vertices = None
        self.car = None
        self.static_cars_vertices = None
        self.static_parking_lot_vertices = None

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
                    self.config.acceleration_limit,
                    self.config.steering_limit,
                ]
            if self.action_type == "discrete":
                if action == 0:  # move forward
                    action = np.array([1, 0])
                elif action == 1:  # move right forward
                    action = np.array([1, -PI/6])
                elif action == 2:  # move left forward
                    action = np.array([1, PI/6])
                elif action == 3:  # move backward
                    action = np.array([-1, 0])
                elif action == 4:  # move right backward
                    action = np.array([-1, -PI/6])
                elif action == 5:  # move left backward
                    action = np.array([-1, PI/6])
                else:
                    raise ValueError(
                        f"Invalid action value: {action}. "
                        f"Valid values are from 0 to 5")

            self.car.loc_old = self.car.car_loc
            self.car.kinematic_act(action)

            if self.render_mode == "human":
                self.render()
            reward = self._reward()
            self.state = self.get_normalized_state()

        return self.state, reward, self.terminated, self.truncated, {"step": self.run_steps}

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
            return self._render(self.render_mode, self.config.window_width, self.config.window_height)

    def _render(self, mode: str, window_w: int, window_h: int):
        if mode == "human":
            if self.window is None:
                # Initialize the parking environment window
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode((window_w, window_h))
                pygame.display.set_caption("Parking Environment")
                if self.clock is None:
                    self.clock = pygame.time.Clock()

                # Initialize the text display
                if self.surf_text is None:
                    pygame.font.init()
                    self.surf_text = pygame.Surface((window_w, window_h), flags=pygame.SRCALPHA)
            font = pygame.font.SysFont('Times New Roman', 15)
            self.surf_text.fill((0, 0, 0, 0))

            # Initialize the parking lot surface
            if self.surf_parkinglot is None:
                self.surf_parkinglot = self._create_parking_surface(window_w, window_h,
                                                                    self.config.colors, self.config.grid_size)
                # draw the static obstacles
                self._draw_static_obstacles()
                # Draw the targeted parking space
                draw_object(self.surf_parkinglot, self.config.colors["RED"], self.parking_lot_vertices)

            # Initialize the car(agent)
            if self.surf_car is None:
                self.surf_car = pygame.Surface((window_w, window_h), flags=pygame.SRCALPHA)
            self.surf_car.fill((0, 0, 0, 0))

            # draw the car(agent) movement
            self.car.draw_car(self.surf_car)

            # draw the car path
            car_loc_old = meters_to_pixels(self.car.loc_old)
            car_loc = meters_to_pixels(self.car.car_loc)
            pygame.draw.line(self.surf_parkinglot, self.config.colors["BLACK"], car_loc_old, car_loc)

            # display Multi-line text
            text_str = (f"Car location: {self.car.car_loc}\nVelocity: {self.car.v}\n"
                        f"Heading angle: {self.car.psi}\nDegree: {self.car.psi * (180 / PI)}")
            text_rect = pygame.Rect(400, 500, 100, 100)  # Define the rectangle area for text
            self.draw_multiline_text(self.surf_text, text_str, self.config.colors["BLACK"], text_rect, font)

            # Compose the final surface
            surf = self.surf_parkinglot.copy()
            surf.blit(self.surf_car, (0, 0))
            surf = pygame.transform.flip(surf, False, True)
            surf.blit(self.surf_text, (0, 0))

            # Update the display
            pygame.event.pump()
            self.clock.tick(self.config.fps)
            # assert self.window is not None
            self.window.fill(self.config.colors["BLACK"])
            self.window.blit(surf, (0, 0))
            pygame.display.flip()

    @staticmethod
    def draw_multiline_text(screen, text, color, rect, font, aa=False, bkg=None):
        lines = text.splitlines()
        rendered_lines = []
        for line in lines:
            line_surface = font.render(line, aa, color, bkg)
            rendered_lines.append(line_surface)

        y = rect.top
        for line_surface in rendered_lines:
            line_height = line_surface.get_height()
            screen.blit(line_surface, (rect.left, y))
            y += line_height  # Move y down to start the next line

    @staticmethod
    def _create_parking_surface(window_w: int, window_h: int, color: dict, grid_size: int):
        surf_parkinglot = pygame.Surface((window_w, window_h), flags=pygame.SRCALPHA)
        surf_parkinglot.fill(color["WHITE"])
        for x in range(0, window_w, grid_size):
            pygame.draw.line(surf_parkinglot, color["GRID_COLOR"], (x, 0), (x, window_h))
        for y in range(0, window_h, grid_size):
            pygame.draw.line(surf_parkinglot, color["GRID_COLOR"], (0, y), (window_w, y))
        return surf_parkinglot

    def _draw_static_obstacles(self):
        for parking_lot_vertex in self.static_parking_lot_vertices:
            draw_object(self.surf_parkinglot, "YELLOW", parking_lot_vertex)
        for car_vertex in self.static_cars_vertices:
            draw_object(self.surf_parkinglot, "GREY", car_vertex)

    def reset(
            self,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):
        super().reset(seed=seed)

        # choose the side
        self.side = 1  # self.parking_strategy.set_initial_loc()

        # set the initial positions
        if self.training_mode == "off":
            self.parking_lot = self.parking_strategy.set_initial_parking_loc(self.side,
                                                                             self.config.window_width,
                                                                             self.config.window_height)
            self.parking_lot_vertices = (self.parking_lot +
                                         self.parking_strategy.get_parking_struct(self.parking_type, self.side))
            while True:
                car_loc = self.parking_strategy.set_initial_car_loc(self.side, self.parking_lot)
                if not self.check_max_distance(self.parking_lot_vertices, car_loc, self.config.max_distance):
                    break
            self.car = Car(car_loc, self.parking_strategy.set_initial_heading(self.side), self.config)
        else:  # for training
            car_loc, self.parking_lot, heading_angle = set_init_position(self.side, self.parking_type, randomized=True)
            self.parking_lot_vertices = (self.parking_lot +
                                         self.parking_strategy.get_parking_struct(self.parking_type, self.side))
            self.car = Car(car_loc, heading_angle, self.config)

        self.car.loc_old = self.car.car_loc
        self.static_cars_vertices, self.static_parking_lot_vertices = self.parking_strategy.generate_static_obstacles(
            self.parking_lot, self.side)
        self.state = self.get_normalized_state()

        self.terminated = False
        self.truncated = False
        self.run_steps = 0

        self.window = None
        self.surf = None
        self.surf_car = None
        self.surf_parkinglot = None
        self.surf_text = None
        self.clock = None

        return self.state, {}

    def get_normalized_state(self):
        """
        Prepare and normalize the state vector for the environment by flattening and combining
        the car's velocity with the distances from parking lot vertices to the car's current location.

        Returns:
            np.ndarray: The normalized and flattened state vector consisting of the car's velocity
                        and the distances to each parking lot vertex, clipped in between -1 and 1.
        """

        # calculate the distance between the car and the parking lot vertices for the coordinate of the car
        distances = []
        for vertex in self.parking_lot_vertices:
            distance = self.transform_point(vertex[0], vertex[1],
                                            self.car.car_loc[0], self.car.car_loc[1], self.car.psi)
            distances.append(distance)
        distances = np.array(distances).flatten()

        # normalization
        normalized_distances = distances / self.config.velocity_limit

        # type1 state (default state)
        if self.config.state_type == 'type1':
            state = normalized_distances  # 8 elements

        # type2 state (guidance reward)
        if self.config.state_type == 'type2':
            guidance = self.transform_point(self.parking_lot[0], self.parking_lot[1],
                                            self.car.car_loc[0], self.car.car_loc[1], self.car.psi)
            normalized_guidance = guidance / self.config.max_distance
            state = np.concatenate((normalized_distances, normalized_guidance))  # 10 elements

        # type3 state (velocity)
        if self.config.state_type == 'type3':
            normalized_velocity = self.car.v / self.config.velocity_limit
            state = np.concatenate((normalized_velocity, normalized_distances))  # 9 elements

        # clip the state value
        state = np.clip(state, a_min=-1, a_max=1)

        return state

    @staticmethod
    def transform_point(x, y, car_x, car_y, heading) -> np.array(['x', 'y']):
        """
        Transform the global coordinate system to the local(car) coordinate system

        Return:
            np.array: x,y coordinate system of the car
        """
        # Translate the point to the new origin
        x -= car_x
        y -= car_y

        # Rotate the point based on the heading
        angle = heading - PI / 2
        new_x = x * math.cos(-angle) - y * math.sin(-angle)
        new_y = x * math.sin(-angle) + y * math.cos(-angle)

        return np.array([new_x, new_y])

    def _reward(self) -> int:
        self.run_steps += 1
        reward = 0

        # check the number of the step
        if self.run_steps == self.config.max_steps:
            reward -= 1
            self.truncated = True
            self.terminated = True
            print("The maximum step reaches")
            return reward

        # check the location
        if self.check_cross_border(self.parking_lot_vertices, self.side, self.car.car_vertices):
            reward -= 1
            self.terminated = True
            print("The car crossed the parking lot vertically/horizontally.")
            return reward

        if self.check_max_distance(self.parking_lot_vertices, self.car.car_loc, self.config.max_distance):
            reward -= 1
            self.terminated = True
            print(f"The distance between the car and the parking is more than {self.config.max_distance} meters")
            return reward

        # check a collision
        if self.check_collision():
            reward -= 1
            self.terminated = True
            print("The car has a collision")
            return reward

        # type1 (default reward)
        if self.config.reward_type == 'type1':
            if self.is_car_in_parking_lot():
                reward += 1
                self.terminated = True
                print("successful parking")
                return reward

        # type2 (guidance reward)
        if self.config.reward_type == 'type2':
            if self.is_car_in_parking_lot():
                if self.is_parking_successful(self.parking_lot, self.car.car_loc, self.config.center_threshold):
                    reward += 1
                    self.terminated = True
                    print("successful parking")

                    parking_angle = self.get_parking_angle(self.parking_type, self.side)
                    angle_penalty = self.calc_angle_dif(self.car.psi, parking_angle, self.config.max_angle_error)

                    # Adjust reward
                    reward -= angle_penalty
                    return reward

        return reward

    @staticmethod
    def is_parking_successful(parking_lot, car_loc, center_threshold):
        distance = abs(parking_lot - car_loc)
        if distance[0] <= center_threshold and distance[1] <= center_threshold:
            return True
        return False

    @staticmethod
    def get_parking_angle(parking_type, side):
        if parking_type == "perpendicular":
            if side == 1:
                return PI / 2
            elif side == 2:
                return -PI / 2
            elif side == 3:
                return 0
            elif side == 4:
                return PI
        elif parking_type == "parallel":
            if side in [1, 2]:
                return [0, PI]  # Car can face either 0 or pi
            elif side in [3, 4]:
                return [PI / 2, -PI / 2]  # Car can face either pi/2 or -pi/2

    @staticmethod
    def calc_angle_dif(psi, parking_angle, max_angle_error):
        # calculate the angle error
        if isinstance(parking_angle, list):
            angle_errors = [np.abs((psi - angle + PI) % (2 * PI) - PI) for angle in parking_angle]
            angle_error = min(angle_errors)
        else:
            angle_error = np.abs((psi - parking_angle + PI) % (2 * PI) - PI)
        angle_penalty = min(0.5 * (angle_error / max_angle_error), 0.5)
        return angle_penalty

    @staticmethod
    def check_cross_border(parking_lot_vertices, side, car_vertices) -> bool:
        """
        check if the car doesn't cross the horizontal/vertical parking border

        Return True if the car cross the horizontal/vertical parking border
        """
        # get each parking lot and car vertices
        pa_top_right, pa_bottom_right, pa_bottom_left, pa_top_left = parking_lot_vertices

        # Define the edges of the parking lot and car
        pa_left_edge = pa_top_left[0]
        pa_right_edge = pa_top_right[0]
        pa_top_edge = pa_top_left[1]
        pa_bottom_edge = pa_bottom_left[1]

        if side == 1:
            return np.any(car_vertices[:, 1] < pa_bottom_edge)
        elif side == 2:
            return np.any(car_vertices[:, 1] > pa_top_edge)
        elif side == 3:
            return np.any(car_vertices[:, 0] < pa_left_edge)
        else:
            return np.any(car_vertices[:, 0] > pa_right_edge)

    def is_car_in_parking_lot(self) -> bool:
        xy1, xy2, xy3, xy4 = self.parking_lot_vertices
        # Check if all car corners are within the parking area
        for corner in self.car.car_vertices:
            if not self.check_boundary(xy1, xy3, corner):
                return False
        return True

    def check_collision(self) -> bool:
        for static_car_vertex in self.static_cars_vertices:
            xy1, xy2, xy3, xy4 = static_car_vertex
            for car_vertex in self.car.car_vertices:
                if self.check_boundary(xy1, xy3, car_vertex):
                    return True
        return False

    @staticmethod
    def check_max_distance(parking_lot_vertices, car_loc, max_distance) -> bool:
        """
        check the distance between the car and the parking lot

        Return: True if it is more than 25 meters
        """
        for parking_lot in parking_lot_vertices:
            if (abs(parking_lot[0] - car_loc[0]) >= max_distance or
                    abs(parking_lot[1] - car_loc[1]) >= max_distance):
                return True
        return False

    @staticmethod
    def check_boundary(xy1, xy2, obj) -> bool:
        """
        check if obj is in between xy1 and xy2

        Parameter
            xy1: top right (x,y) position
            xy2: bottom left (x,y) position
            obj: targeted object (x,y) position

        Return:
            bool
        """
        if xy2[0] <= obj[0] <= xy1[0] and xy2[1] <= obj[1] <= xy1[1]:
            return True
        return False

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
