import numpy as np
import random
import pygame
import gymnasium as gym
from typing import Optional


'''
## todo
recheck check_collision function
recheck the size of the prallel/perpendicular parking
ongoing: consider OOP or structure of this code
consider making a function which can be used in def is_parking_successful(self): and def check_collision(self): and def is_valid_loc(self, width, height): 
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

# parameters for cars in the parking environment
CAR_L, CAR_W = 80, 40
CAR_STRUCT = np.array([[+CAR_L / 2, +CAR_W / 2],
                       [+CAR_L / 2, -CAR_W / 2],
                       [-CAR_L / 2, -CAR_W / 2],
                       [-CAR_L / 2, +CAR_W / 2]],
                      np.int32)
WHEEL_L, WHEEL_W = 15, 7
WHEEL_STRUCT = np.array([[+WHEEL_L / 2, +WHEEL_W / 2],
                         [+WHEEL_L / 2, -WHEEL_W / 2],
                         [-WHEEL_L / 2, -WHEEL_W / 2],
                         [-WHEEL_L / 2, +WHEEL_W / 2]],
                        np.int32)
WHEEL_POS = np.array([[25, 15], [25, -15], [-25, 15], [-25, -15]])
PARALLEL = np.array([
                [+CAR_W / 2 + 5, +CAR_L / 2 + 5],
                [+CAR_W / 2 + 5, -CAR_L / 2 - 5],
                [-CAR_W / 2 - 5, -CAR_L / 2 - 5],
                [-CAR_W / 2 - 5, +CAR_L / 2 + 5]],
                np.int32)
PERPENDICULAR = np.array([
                [+CAR_L / 2 + 5, +CAR_W / 2 + 5],
                [+CAR_L / 2 + 5, -CAR_W / 2 - 5],
                [-CAR_L / 2 - 5, -CAR_W / 2 - 5],
                [-CAR_L / 2 - 5, +CAR_W / 2 + 5]],
                np.int32)
DT = 1

MAX_STEPS = 300


class Car:
    def __init__(self, car_loc, psi=0, v=0):
        self.car_loc = car_loc
        self.psi = psi
        self.v = v
        self.delta = 0
        self.car_vertices = []

    def kinematic_act(self, action):
        """
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
    def rotate_car(car_loc, angle=0):
        r = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ])
        rotated_car_loc = (r @ car_loc.T).T
        return rotated_car_loc

    def get_car_vertices(self):
        # calculate the rotation of the car itself
        # add the center of the car x,y location
        self.car_vertices = self.rotate_car(CAR_STRUCT, angle=self.psi) + self.car_loc

    def draw_car(self, screen):
        """
        Parameters:
            screen: pygame.Surface

        """
        # the car(agent)
        self.get_car_vertices()
        # draw the car(agent)
        pygame.draw.polygon(screen, COLORS["GREEN"], self.car_vertices)
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

    def __init__(
            self,
            render_mode: Optional[str] = None,
            action_type: Optional[str] = None,
            parking_type: Optional[str] = None

    ) -> None:
        """
        Initializes a parking instance.

        Parameters:
            render_mode: the drawing mode for visualization
            action_type: the type of action for the agent
        """
        super().__init__()
        assert render_mode in self.metadata["render_mode"]
        assert action_type in self.metadata["action_type"]
        assert parking_type in self.metadata["parking_type"]
        self.render_mode = render_mode
        self.parking_type = parking_type
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

            reward = self._reward()
            self.state = [self.car.v, self.parking_lot_vertices - self.car.car_loc]

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
                self._draw_parking_space(self.parking_lot_vertices)

            # for the car(agent)
            if self.surf_car is None:
                self.surf_car = pygame.Surface(
                    (WINDOW_W, WINDOW_H), flags=pygame.SRCALPHA
                )
            self.surf_car.fill((0, 0, 0, 0))

            # draw the car(agent) movement
            self.car.draw_car(self.surf_car)

            # draw the car path
            pygame.draw.line(self.surf_parkinglot, COLORS["BLACK"], self.car.loc_old, self.car.car_loc)

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

    def _draw_parking_space(self, parking_lot_vertex):
        pygame.draw.polygon(self.surf_parkinglot, COLORS["YELLOW"], parking_lot_vertex)

    def _draw_static_obstacles(self):
        for parking_lot_vertex in self.static_parking_lot_vertices:
            self._draw_parking_space(parking_lot_vertex)
        for car_vertex in self.static_cars_vertices:
            pygame.draw.polygon(self.surf_parkinglot, COLORS["GREY"], car_vertex)

    def generate_static_obstacles(self, type):
        static_cars_vertices = []
        static_parking_vertices = []
        if self.parking_type == "parallel":
            # calculate the obstacle cars center location
            static_cars_loc = np.array([
                [self.parking_lot[0] + 0, self.parking_lot[1] + 50],
                [self.parking_lot[0] + 0, self.parking_lot[1] - 50],
            ])
            car_struct = np.array([
                [+CAR_W / 2, +CAR_L / 2],
                [+CAR_W / 2, -CAR_L / 2],
                [-CAR_W / 2, -CAR_L / 2],
                [-CAR_W / 2, +CAR_L / 2]],
                np.int32)

        if self.parking_type == "perpendicular":
            # calculate the obstacle cars center location
            static_cars_loc = np.array([
                [self.parking_lot[0] + 0, self.parking_lot[1] + 50],
                [self.parking_lot[0] + 0, self.parking_lot[1] - 30],
            ])
            car_struct = CAR_STRUCT
        # calculate the obstacle cars vertices
        for loc in static_cars_loc:
            car_vertices = car_struct + loc
            parking_vertices = type + loc
            static_cars_vertices.append(car_vertices)
            static_parking_vertices.append(parking_vertices)
        return static_cars_vertices, static_parking_vertices

    def reset(
            self,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):
        super().reset(seed=seed)

        self.car.car_loc = self.set_random_loc()
        self.car.v = 0
        self.car.psi = 0
        self.car.loc_old = self.car.car_loc
        self.car.delta = 0
        self.parking_lot = self.set_random_loc()
        if self.parking_type == "parallel":
            self.parking_lot_vertices = self.parking_lot + PARALLEL
            self.static_cars_vertices, self.static_parking_lot_vertices = self.generate_static_obstacles(PARALLEL)
        if self.parking_type == "perpendicular":
            self.parking_lot_vertices = self.parking_lot + PERPENDICULAR
            self.static_cars_vertices, self.static_parking_lot_vertices = self.generate_static_obstacles(PERPENDICULAR)

        self.state = [self.car.v, self.parking_lot_vertices - self.car.car_loc]
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

        return self.reward

    def is_valid_loc(self):
        for car_vertex in self.car.car_vertices:
            if (car_vertex[0] < 0 or car_vertex[0] > WINDOW_W or
                    car_vertex[1] < 0 or car_vertex[1] > WINDOW_H):
                return True
            else:
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
            if (xy4[0] <= self.car.car_loc[0] <= xy1[0] and
            xy2[1] <= self.car.car_loc[1] <= xy1[1]):
                return True
            else:
                return False

    @staticmethod
    def set_random_loc():
        x = random.uniform(100, WINDOW_W - 100)
        y = random.uniform(100, WINDOW_H - 100)
        loc = [x, y]
        return loc

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None

