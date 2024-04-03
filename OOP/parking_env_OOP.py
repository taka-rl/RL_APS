import numpy as np
import random
import pygame
import gymnasium as gym
from typing import Optional

'''
TODO: consider OOP or structure of this code -> Parking/Car/Render class?
TODO: consider making a function which can be used in def is_parking_successful(self): and def check_collision(self): and def is_valid_loc(self, width, height): 
as they have same logic
'''

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
PIXEL_TO_METER_SCALE = 0.05  # Define the scale as 1 pixel = 0.05 meters

# parameters for cars in the parking environment
CAR_L, CAR_W = 80 * PIXEL_TO_METER_SCALE, 40 * PIXEL_TO_METER_SCALE  # Car length and width in meters
CAR_STRUCT = np.array([[+CAR_L / 2, +CAR_W / 2],
                       [+CAR_L / 2, -CAR_W / 2],
                       [-CAR_L / 2, -CAR_W / 2],
                       [-CAR_L / 2, +CAR_W / 2]],
                      dtype=np.float32)  # Coordinates adjusted for meters

WHEEL_L, WHEEL_W = 15 * PIXEL_TO_METER_SCALE, 7 * PIXEL_TO_METER_SCALE  # Wheel length and width in meters
WHEEL_STRUCT = np.array([[+WHEEL_L / 2, +WHEEL_W / 2],
                         [+WHEEL_L / 2, -WHEEL_W / 2],
                         [-WHEEL_L / 2, -WHEEL_W / 2],
                         [-WHEEL_L / 2, +WHEEL_W / 2]],
                        dtype=np.float32)  # Coordinates adjusted for meters

WHEEL_POS = np.array([[25 * PIXEL_TO_METER_SCALE, 15 * PIXEL_TO_METER_SCALE],
                      [25 * PIXEL_TO_METER_SCALE, -15 * PIXEL_TO_METER_SCALE],
                      [-25 * PIXEL_TO_METER_SCALE, 15 * PIXEL_TO_METER_SCALE],
                      [-25 * PIXEL_TO_METER_SCALE, -15 * PIXEL_TO_METER_SCALE]],
                     dtype=np.float32)  # Position adjusted for meters

PARALLEL = np.array([
    [+CAR_W / 2 + 5 * PIXEL_TO_METER_SCALE, +CAR_L / 2 + 10 * PIXEL_TO_METER_SCALE],
    [+CAR_W / 2 + 5 * PIXEL_TO_METER_SCALE, -CAR_L / 2 - 10 * PIXEL_TO_METER_SCALE],
    [-CAR_W / 2 - 5 * PIXEL_TO_METER_SCALE, -CAR_L / 2 - 10 * PIXEL_TO_METER_SCALE],
    [-CAR_W / 2 - 5 * PIXEL_TO_METER_SCALE, +CAR_L / 2 + 10 * PIXEL_TO_METER_SCALE]],
    dtype=np.float32)  # Adjusted for meters

PERPENDICULAR = np.array([
    [+CAR_L / 2 + 5 * PIXEL_TO_METER_SCALE, +CAR_W / 2 + 10 * PIXEL_TO_METER_SCALE],
    [+CAR_L / 2 + 5 * PIXEL_TO_METER_SCALE, -CAR_W / 2 - 10 * PIXEL_TO_METER_SCALE],
    [-CAR_L / 2 - 5 * PIXEL_TO_METER_SCALE, -CAR_W / 2 - 10 * PIXEL_TO_METER_SCALE],
    [-CAR_L / 2 - 5 * PIXEL_TO_METER_SCALE, +CAR_W / 2 + 10 * PIXEL_TO_METER_SCALE]],
    dtype=np.float32)  # Adjusted for meters

OFFSET_PARALLEL = 100 * PIXEL_TO_METER_SCALE
OFFSET_PERPENDICULAR = 60 * PIXEL_TO_METER_SCALE

DT = 0.1

MAX_STEPS = 300
MAX_DISTANCE = 25.0


def meters_to_pixels(meters):
    """
    Convert meters to pixels based on the defined scale.

    Parameters:
        meters: The value in meters to convert.

    Returns:
        float: The equivalent value in pixels.
    """
    return meters / PIXEL_TO_METER_SCALE


class Car:
    def __init__(self, car_loc, psi=0.0, v=0.0):
        self.car_loc = car_loc
        self.psi = psi
        self.v = v
        self.delta = 0
        self.car_vertices = np.array([[0, 0], [0, 0], [0, 0], [0, 0]], dtype=np.float32)

    def kinematic_act(self, action):
        """
        Calculate the car(agent) movement

        Parameters:
            action(list): [a, δ]: a is acceleration, δ(delta) is steering angle.
            self.v : velocity
            self.psi(ψ): the heading angle of the car

        Kinematic bicycle model:
        x_dot = v * np.cos(psi)
        y_dot = v * np.sin(psi)
        v_dot = a
        psi_dot = v * np.tan(delta) / CAR_L
        """
        x_dot = self.v * np.cos(self.psi)
        y_dot = self.v * np.sin(self.psi)
        v_dot = action[0]
        psi_dot = self.v * np.tan(action[1]) / CAR_L
        car_loc = np.array([x_dot, y_dot])
        self.update_state(car_loc, v_dot, psi_dot, DT)
        self.delta = action[1]

    def update_state(self, car_loc, v_dot, psi_dot, dt):
        self.car_loc += dt * car_loc
        self.v += v_dot
        self.v = np.clip(self.v, -VELOCITY_LIMIT, VELOCITY_LIMIT)
        self.psi += dt * psi_dot

    @staticmethod
    def rotate_car(car_loc, angle=0.0):
        r = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ])
        return (r @ car_loc.T).T

    def get_car_vertices(self):
        # calculate the rotation of the car itself
        # add the center of the car x,y location
        self.car_vertices = self.rotate_car(CAR_STRUCT, angle=self.psi) + self.car_loc

    def draw_car(self, screen):
        """
        Draw the car(agent)

        Parameters:
            screen: pygame.Surface
        """
        # update the car(agent) vertices
        self.get_car_vertices()
        # convert meters to pixels
        car_vertices = meters_to_pixels(self.car_vertices)
        # draw the car(agent)
        pygame.draw.polygon(screen, COLORS["GREEN"], car_vertices)

        # wheels
        # calculate the rotation of the wheels
        wheel_points = self.rotate_car(WHEEL_POS, angle=self.psi)
        # draw each wheel
        for i, wheel_point in enumerate(wheel_points):
            if i < 2:
                wheel_vertices = self.rotate_car(WHEEL_STRUCT, angle=self.psi + self.delta)
            else:
                wheel_vertices = self.rotate_car(WHEEL_STRUCT, angle=self.psi)
            wheel_vertices += wheel_point + self.car_loc
            wheel_vertices = meters_to_pixels(wheel_vertices)
            pygame.draw.polygon(screen, COLORS["RED"], wheel_vertices)


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
        "render_mode": ["human", "no_render"],
        "render_fps": FPS,
        "action_type": ["continuous"],
        "parking_type": ["parallel", "perpendicular"]
    }

    def __init__(self, env_config) -> None:
        """
        Initializes a parking instance.

        Parameters:
            render_mode: the drawing mode for visualization
            action_type: the type of action for the agent
        """
        super().__init__()
        #assert render_mode in self.metadata["render_mode"]
        #assert action_type in self.metadata["action_type"]
        #assert parking_type in self.metadata["parking_type"]
        self.render_mode = env_config["render_mode"]
        self.parking_type = env_config["parking_type"]
        self.action_type = env_config["action_type"]
        self.observation_space = gym.spaces.Box(-1, 1, dtype=np.float32)

        if self.action_type == "continuous":
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
        self.car = Car(self.set_random_loc())

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
                self.car.loc_old = self.car.car_loc
                self.car.kinematic_act(action)
                self.car.get_car_vertices()

            self._reward()

            self.state = [self.car.v / VELOCITY_LIMIT, (self.parking_lot_vertices - self.car.car_loc) / MAX_DISTANCE]

        if self.render_mode == "human":
            self.render()

        return self.state, self.reward, self.terminated, self.truncated, {}

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
        if mode == "human" and self.window is None:
            # Initialize the parking environment window
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((window_w, window_h))
            pygame.display.set_caption("Parking Environment")
            if self.clock is None:
                self.clock = pygame.time.Clock()

        # for the parking lot
        if mode == "human" or mode == "rgb_array":
            if self.surf_parkinglot is None:
                self.surf_parkinglot = self._create_parking_surface()
                # draw the static obstacles
                self._draw_static_obstacles()

                # Draw the targeted parking space
                self._draw_parking_space("RED", self.parking_lot_vertices)

            # for the car(agent)
            if self.surf_car is None:
                self.surf_car = pygame.Surface(
                    (WINDOW_W, WINDOW_H), flags=pygame.SRCALPHA
                )
            self.surf_car.fill((0, 0, 0, 0))

            # draw the car(agent) movement
            self.car.draw_car(self.surf_car)

            # convert to pixels
            car_loc_old = meters_to_pixels(self.car.loc_old)
            car_loc = meters_to_pixels(self.car.car_loc)

            # draw the car path
            pygame.draw.line(self.surf_parkinglot, COLORS["BLACK"], car_loc_old, car_loc)

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

    @staticmethod
    def _create_parking_surface():
        surf_parkinglot = pygame.Surface((WINDOW_W, WINDOW_H), flags=pygame.SRCALPHA)
        surf_parkinglot.fill(COLORS["WHITE"])
        for x in range(0, WINDOW_W, GRID_SIZE):
            pygame.draw.line(surf_parkinglot, COLORS["GRID_COLOR"], (x, 0), (x, WINDOW_H))
        for y in range(0, WINDOW_H, GRID_SIZE):
            pygame.draw.line(surf_parkinglot, COLORS["GRID_COLOR"], (0, y), (WINDOW_W, y))
        return surf_parkinglot

    def _draw_parking_space(self, color, parking_lot_vertex):
        # convert to pixels and draw
        parking_lot_vertex = meters_to_pixels(parking_lot_vertex)
        pygame.draw.polygon(self.surf_parkinglot, COLORS[color], parking_lot_vertex)

    def _draw_static_cars(self, color, car_vertex):
        # convert to pixels and draw
        car_vertex = meters_to_pixels(car_vertex)
        pygame.draw.polygon(self.surf_parkinglot, COLORS[color], car_vertex)

    def _draw_static_obstacles(self):
        for parking_lot_vertex in self.static_parking_lot_vertices:
            self._draw_parking_space("YELLOW", parking_lot_vertex)
        for car_vertex in self.static_cars_vertices:
            self._draw_static_cars("GREY", car_vertex)

    def generate_static_obstacles(self):
        static_cars_vertices = []
        static_parking_vertices = []

        offset = OFFSET_PARALLEL if self.parking_type == "parallel" else OFFSET_PERPENDICULAR
        car_struct = self.get_car_struct_for_parallel() if self.parking_type == "parallel" else CAR_STRUCT

        # Center locations for the static cars
        static_cars_loc = np.array([[self.parking_lot[0], self.parking_lot[1] + offset],
                                    [self.parking_lot[0], self.parking_lot[1] - offset]])

        # calculate the obstacle cars vertices and parking lots vertices
        parking_type = self.get_parking_struct(self.parking_type)
        for loc in static_cars_loc:
            static_cars_vertices.append(car_struct + loc)
            static_parking_vertices.append(parking_type + loc)
        return static_cars_vertices, static_parking_vertices

    @staticmethod
    def get_car_struct_for_parallel():
        """
        Get the car structure for parallel parking.

        Returns:
            np.ndarray: The car structure vertices adjusted for parallel parking.
        """
        return np.array([
            [+CAR_W / 2, +CAR_L / 2],
            [+CAR_W / 2, -CAR_L / 2],
            [-CAR_W / 2, -CAR_L / 2],
            [-CAR_W / 2, +CAR_L / 2]],
            dtype=np.float32)

    @staticmethod
    def get_parking_struct(parking_type: str):
        """
        Get the parking structure based on the parking type.

        Parameters:
            parking_type (str): The type of parking arrangement.

        Returns:
            np.ndarray: The vertices for parking space structure.
        """
        return PARALLEL if parking_type == "parallel" else PERPENDICULAR

    def reset(
            self,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):
        super().reset(seed=seed)

        self.car.car_loc = self.set_random_loc()
        self.car.v = 0.0
        self.car.psi = 0.0
        self.car.loc_old = self.car.car_loc
        self.car.delta = 0.0
        self.parking_lot = self.set_random_loc()
        if self.parking_type == "parallel":
            self.parking_lot_vertices = self.parking_lot + PARALLEL
            self.static_cars_vertices, self.static_parking_lot_vertices = self.generate_static_obstacles()
        if self.parking_type == "perpendicular":
            self.parking_lot_vertices = self.parking_lot + PERPENDICULAR
            self.static_cars_vertices, self.static_parking_lot_vertices = self.generate_static_obstacles()

        self.state = [self.car.v / VELOCITY_LIMIT, (self.parking_lot_vertices - self.car.car_loc) / MAX_DISTANCE]
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

        # check the number of the step
        if self.run_steps == MAX_STEPS:
            self.truncated = True

        # check the location
        if self.is_valid_loc():
            self.reward -= 1
            self.terminated = True
            print("The car is not a valid location")

        # check a collision
        if self.check_collision():
            self.reward -= 1
            self.terminated = True
            print("The car has a collision")

        # check the parking
        if self.is_parking_successful():
            self.reward += 1
            self.truncated = True
            print("successful parking")

    def is_valid_loc(self):
        for car_vertex in self.car.car_vertices:
            if (car_vertex[0] < 0 or car_vertex[0] > WINDOW_W * PIXEL_TO_METER_SCALE or
                    car_vertex[1] < 0 or car_vertex[1] > WINDOW_H * PIXEL_TO_METER_SCALE):
                return True
        return False

    def is_parking_successful(self):
        pa_top_right, pa_bottom_right, pa_bottom_left, pa_top_left = self.parking_lot_vertices

        # Define the edges of the parking area
        pa_left_edge = pa_top_left[0]
        pa_right_edge = pa_top_right[0]
        pa_top_edge = pa_top_left[1]
        pa_bottom_edge = pa_bottom_left[1]

        # Check if all car corners are within the parking area
        for corner in self.car.car_vertices:
            if not (pa_left_edge <= corner[0] <= pa_right_edge and
                    pa_bottom_edge <= corner[1] <= pa_top_edge):
                return False
        return True

    def check_collision(self):
        for static_car_vertex in self.static_cars_vertices:
            xy1, xy2, xy3, xy4 = static_car_vertex
            for car_vertex in self.car.car_vertices:
                if xy4[0] <= car_vertex[0] <= xy1[0] and xy2[1] <= car_vertex[1] <= xy1[1]:
                    return True
        return False

    @staticmethod
    def set_random_loc():
        """
        Give random values for the center x,y location from 100 to maximum value - 100

        Return:
             np.array: x, y location values
        """
        x = random.uniform(100, WINDOW_W - 100) * PIXEL_TO_METER_SCALE
        y = random.uniform(100, WINDOW_H - 100) * PIXEL_TO_METER_SCALE
        return np.array([x, y])

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
