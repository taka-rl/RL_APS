import numpy as np
import random
import pygame
import gymnasium as gym
from typing import Optional
from sim_env.car import Car
from sim_env.com_fcn import meters_to_pixels, draw_object
from sim_env.parameters import *


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
        "render_fps": FPS,
        "action_types": ["continuous", "discrete"],
        "parking_types": ["parallel", "perpendicular"],
        "training_modes": [True, False]
    }

    def __init__(self, env_config) -> None:
        """
        Initializes a parking instance.

        Parameters:
            env_config: contains the action type, render mode and parking type
        """
        super().__init__()
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

        self.render_mode = env_config["render_mode"]
        self.parking_type = env_config["parking_type"]
        self.action_type = env_config["action_type"]
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(9,), dtype=np.float32)

        if self.action_type == "continuous":
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        elif self.action_type == "discrete":
            self.action_space = gym.spaces.Discrete(6)

        self.window = None
        self.surf = None
        self.surf_car = None
        self.surf_parkinglot = None
        self.surf_text = None
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
            if self.action_type == "discrete":
                if action == 0:  # move forward
                    action = np.array([1, 0])
                elif action == 1:  # move right forward
                    action = np.array([1, -np.pi/6])
                elif action == 2:  # move left forward
                    action = np.array([1, np.pi/6])
                elif action == 3:  # move backward
                    action = np.array([-1, 0])
                elif action == 4:  # move right backward
                    action = np.array([-1, -np.pi/6])
                elif action == 5:  # move left backward
                    action = np.array([-1, np.pi/6])
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
            return self._render(self.render_mode, WINDOW_W, WINDOW_H)

    def _render(self, mode: str, window_w, window_h):
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
                    self.surf_text = pygame.Surface((WINDOW_W, WINDOW_H), flags=pygame.SRCALPHA)
            font = pygame.font.SysFont('Times New Roman', 15)
            self.surf_text.fill((0, 0, 0, 0))

            # Initialize the parking lot surface
            if self.surf_parkinglot is None:
                self.surf_parkinglot = self._create_parking_surface()
                # draw the static obstacles
                self._draw_static_obstacles()
                # Draw the targeted parking space
                draw_object(self.surf_parkinglot, "RED", self.parking_lot_vertices)

            # Initialize the car(agent)
            if self.surf_car is None:
                self.surf_car = pygame.Surface((WINDOW_W, WINDOW_H), flags=pygame.SRCALPHA)
            self.surf_car.fill((0, 0, 0, 0))

            # draw the car(agent) movement
            self.car.draw_car(self.surf_car)

            # draw the car path
            car_loc_old = meters_to_pixels(self.car.loc_old)
            car_loc = meters_to_pixels(self.car.car_loc)
            pygame.draw.line(self.surf_parkinglot, COLORS["BLACK"], car_loc_old, car_loc)

            # display Multi-line text
            text_str = (f"Car location: {self.car.car_loc}\nVelocity: {self.car.v}\n"
                        f"Heading angle: {self.car.psi}\n")
            text_rect = pygame.Rect(400, 500, 100, 100)  # Define the rectangle area for text
            self.draw_multiline_text(self.surf_text, text_str, COLORS["BLACK"], text_rect, font)

            # Compose the final surface
            surf = self.surf_parkinglot.copy()
            surf.blit(self.surf_car, (0, 0))
            surf = pygame.transform.flip(surf, False, True)
            surf.blit(self.surf_text, (0, 0))

            # Update the display
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            # assert self.window is not None
            self.window.fill(COLORS["BLACK"])
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
    def _create_parking_surface():
        surf_parkinglot = pygame.Surface((WINDOW_W, WINDOW_H), flags=pygame.SRCALPHA)
        surf_parkinglot.fill(COLORS["WHITE"])
        for x in range(0, WINDOW_W, GRID_SIZE):
            pygame.draw.line(surf_parkinglot, COLORS["GRID_COLOR"], (x, 0), (x, WINDOW_H))
        for y in range(0, WINDOW_H, GRID_SIZE):
            pygame.draw.line(surf_parkinglot, COLORS["GRID_COLOR"], (0, y), (WINDOW_W, y))
        return surf_parkinglot

    def _draw_static_obstacles(self):
        for parking_lot_vertex in self.static_parking_lot_vertices:
            draw_object(self.surf_parkinglot, "YELLOW", parking_lot_vertex)
        for car_vertex in self.static_cars_vertices:
            draw_object(self.surf_parkinglot, "GREY", car_vertex)

    def generate_static_obstacles(self):
        static_cars_vertices = []
        static_parking_vertices = []

        offset = OFFSET_PARALLEL if self.parking_type == "parallel" else OFFSET_PERPENDICULAR

        # center locations for the static cars
        if self.side in [1, 2]:
            static_cars_loc = np.array([[self.parking_lot[0] + offset, self.parking_lot[1]],
                                        [self.parking_lot[0] - offset, self.parking_lot[1]]])

        else:
            static_cars_loc = np.array([[self.parking_lot[0], self.parking_lot[1] + offset],
                                        [self.parking_lot[0], self.parking_lot[1] - offset]])

        # calculate the obstacle cars vertices and parking lots vertices
        parking_struct = self.get_parking_struct(self.parking_type, self.side)
        car_struct = self.get_car_struct(self.parking_type, self.side)
        for loc in static_cars_loc:
            static_cars_vertices.append(car_struct + loc)
            static_parking_vertices.append(parking_struct + loc)
        return static_cars_vertices, static_parking_vertices

    @staticmethod
    def get_parking_struct(parking_type: str, side: int):
        """
        Get the parking structure based on the parking type.

        Parameters:
            parking_type (str): The type of parking arrangement.
            side (int): The side of the parking lot

        Returns:
            np.ndarray: The vertices for parking space structure.
        """
        if parking_type == "parallel":
            return PARALLEL_HORIZONTAL if side in [1, 2] else PARALLEL_VERTICAL

        else:  # perpendicular
            return PERPENDICULAR_HORIZONTAL if side in [1, 2] else PERPENDICULAR_VERTICAL

    @staticmethod
    def get_car_struct(parking_type: str, side: int):
        """
        Get the car structure based on the parking type.

        Parameters:
            parking_type (str): The type of parking arrangement.
            side (int): The side of the parking lot

        Returns:
            np.ndarray: The vertices for parking space structure.
        """
        if parking_type == "parallel":
            return CAR_STRUCT if side in [1, 2] else np.array([[+CAR_W / 2, +CAR_L / 2],
                                                               [+CAR_W / 2, -CAR_L / 2],
                                                               [-CAR_W / 2, -CAR_L / 2],
                                                               [-CAR_W / 2, +CAR_L / 2]],
                                                              dtype=np.float32)  # Coordinates adjusted for meters
        else:  # perpendicular
            return CAR_STRUCT if side in [3, 4] else np.array([[+CAR_W / 2, +CAR_L / 2],
                                                               [+CAR_W / 2, -CAR_L / 2],
                                                               [-CAR_W / 2, -CAR_L / 2],
                                                               [-CAR_W / 2, +CAR_L / 2]],
                                                              dtype=np.float32)  # Coordinates adjusted for meters

    def reset(
            self,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.side = self.set_initial_loc()

        if not self.training_mode:
            # to make sure the initial distance between the car and the parking lot is within 25 meters
            self.parking_lot = self.set_initial_parking_loc(self.side)
            self.parking_lot_vertices = self.parking_lot + self.get_parking_struct(self.parking_type, self.side)
            while True:
                car_loc = self.set_initial_car_loc(self.side, self.parking_lot)
                if not self.check_max_distance(self.parking_lot_vertices, car_loc):
                    break
            self.car = Car(car_loc, self.set_initial_heading())
        else:  # for training
            from sim_env.init_state import set_init_position
            car_loc, self.parking_lot, heading_angle = set_init_position(self.side)
            self.parking_lot_vertices = self.parking_lot + self.get_parking_struct(self.parking_type, self.side)
            self.car = Car(car_loc, heading_angle)

        self.car.loc_old = self.car.car_loc
        self.static_cars_vertices, self.static_parking_lot_vertices = self.generate_static_obstacles()
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

    @staticmethod
    def set_initial_heading() -> float:
        """
        Set the initial heading of the car between 0 and 2π radians
        """
        return np.random.uniform(0, 2 * np.pi)  # Full circle

    def get_normalized_state(self):
        """
        Prepare and normalize the state vector for the environment by flattening and combining
        the car's velocity with the distances from parking lot vertices to the car's current location.

        This function calculates the normalized distances between the car and each vertex of the
        parking lot, then concatenates these distances with the car's normalized velocity to form
        a comprehensive state vector suitable for use in reinforcement learning algorithms.

        Returns:
            np.ndarray: The normalized and flattened state vector consisting of the car's velocity
                        and the distances to each parking lot vertex, clipped in between -1 and 1.
        """

        # normalize distances between car's current location and parking lot vertices
        distances = ((self.parking_lot_vertices - self.car.car_loc) / MAX_DISTANCE).flatten()

        # combine normalized velocity with normalized distances to form the state vector
        state = np.concatenate(([self.car.v / VELOCITY_LIMIT], distances))

        # clip the state value
        state = np.clip(state, a_min=-1, a_max=1)

        return state

    @staticmethod
    def set_initial_loc() -> int:
        """
        Set the initial car and parking lot location

        Return:
            int value between 1 and 4
        """
        return random.randint(1, 4)

    @staticmethod
    def set_initial_car_loc(side, parking_loc) -> np.array(['x', 'y']):
        """
        Set the initial car location

        ini_dist (float): the initial distance between the car and the parking lot,
                        randomly setting between 10 and 20 meters.

        parking_loc (np.array): The [x, y] location of the parking lot in meters.

        side (int): determines on which side of the map the parking lot will be placed
                - 1: the car is placed on the bottom side of the parking area.
                    x is randomly set between 100 and 700 pixels (before scaling),
                    and y is plus ini_dist from parking_loc[1].
                - 2: the car is placed on the top side of the parking area.
                    x is randomly set between 100 and 700 pixels (before scaling),
                    and y is minus ini_dist from parking_loc[1].
                - 3: the car is placed on the left side of the parking area.
                    x is plus ini_dist from parking_loc[0].
                    and y is randomly set between 100 and 500 pixels (before scaling).
                - 4: the car is placed on the right side of the parking area.
                    x is minus ini_dist from parking_loc[0].
                    and y is randomly set between 100 and 500 pixels (before scaling).
        Return:
            np.array: the initial center of the car location [x,y] in meters,
                    adjusted for an appropriate distance from the parking lot.

        """
        init_dist = random.uniform(10, 20)

        if side == 1:
            x_car = random.uniform(100, WINDOW_W - 100) * PIXEL_TO_METER_SCALE
            y_car = parking_loc[1] + init_dist
        elif side == 2:
            x_car = random.uniform(100, WINDOW_W - 100) * PIXEL_TO_METER_SCALE
            y_car = parking_loc[1] - init_dist
        elif side == 3:
            x_car = parking_loc[0] + init_dist
            y_car = random.uniform(100, WINDOW_H - 100) * PIXEL_TO_METER_SCALE
        else:
            x_car = parking_loc[0] - init_dist
            y_car = random.uniform(100, WINDOW_H - 100) * PIXEL_TO_METER_SCALE

        return np.array([x_car, y_car])

    @staticmethod
    def set_initial_parking_loc(side) -> np.array(['x', 'y']):
        """
        Set the initial parking lot location

        side (int): determines on which side of the map the parking lot will be placed.
                - 1: the parking lot is placed on the bottom side,
                    x is randomly set between 100 and 700 pixels (before scaling),
                    and y is set to 50 pixels (before scaling).
                - 2: the parking lot is placed on the top side,
                    x is randomly set between 100 and 700 pixels (before scaling),
                    and y is set to 550 pixels (before scaling).
                - 3: the parking lot is placed on the left side, x is set to 50 pixels (before scaling),
                    and y is randomly set between 100 and 500 pixels (before scaling).
                - 4: the parking lot is placed on the right side, x is set to 750 pixels (before scaling),
                    and y is randomly set between 100 and 500 pixels (before scaling).

        Return:
            np.array:the center of the parking lot location [x,y]
        """
        if side == 1:
            x_parking = random.uniform(100, WINDOW_W - 100) * PIXEL_TO_METER_SCALE
            y_parking = 50 * PIXEL_TO_METER_SCALE
        elif side == 2:
            x_parking = random.uniform(100, WINDOW_W - 100) * PIXEL_TO_METER_SCALE
            y_parking = 550 * PIXEL_TO_METER_SCALE
        elif side == 3:
            x_parking = 50 * PIXEL_TO_METER_SCALE
            y_parking = random.uniform(100, WINDOW_H - 100) * PIXEL_TO_METER_SCALE
        else:
            x_parking = 750 * PIXEL_TO_METER_SCALE
            y_parking = random.uniform(100, WINDOW_H - 100) * PIXEL_TO_METER_SCALE

        return np.array([x_parking, y_parking])

    def _reward(self) -> int:
        self.run_steps += 1
        reward = 0

        # check the number of the step
        if self.run_steps == MAX_STEPS:
            reward -= 1
            self.truncated = True
            self.terminated = True
            print("The maximum step reaches")
            return reward

        # check the location
        if self.check_cross_border():
            reward -= 1
            self.terminated = True
            print("The car crossed the parking lot vertically/horizontally.")
            return reward

        if self.check_max_distance(self.parking_lot_vertices, self.car.car_loc):
            reward -= 1
            self.terminated = True
            print("The distance between the car and the parking is more than 25 meters")
            return reward

        # check a collision
        if self.check_collision():
            reward -= 1
            self.terminated = True
            print("The car has a collision")
            return reward

        # check the parking
        if self.is_parking_successful():
            reward += 1
            self.terminated = True
            print("successful parking")
            return reward
        return reward

    def check_cross_border(self) -> bool:
        """
        check if the car doesn't cross the horizontal/vertical parking border

        Return True if the car cross the horizontal/vertical parking border
        """
        # get each parking lot and car vertices
        pa_top_right, pa_bottom_right, pa_bottom_left, pa_top_left = self.parking_lot_vertices

        # Define the edges of the parking lot and car
        pa_left_edge = pa_top_left[0]
        pa_right_edge = pa_top_right[0]
        pa_top_edge = pa_top_left[1]
        pa_bottom_edge = pa_bottom_left[1]

        if self.side == 1:
            return np.any(self.car.car_vertices[:, 1] < pa_bottom_edge)
        elif self.side == 2:
            return np.any(self.car.car_vertices[:, 1] > pa_top_edge)
        elif self.side == 3:
            return np.any(self.car.car_vertices[:, 0] < pa_left_edge)
        else:
            return np.any(self.car.car_vertices[:, 0] > pa_right_edge)

    def is_parking_successful(self) -> bool:
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
    def check_max_distance(parking_lot_vertices, car_loc) -> bool:
        """
        check the distance between the car and the parking lot

        Return: True if it is more than 25 meters
        """
        for parking_lot in parking_lot_vertices:
            if (abs(parking_lot[0] - car_loc[0]) >= MAX_DISTANCE or
                    abs(parking_lot[1] - car_loc[1]) >= MAX_DISTANCE):
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
