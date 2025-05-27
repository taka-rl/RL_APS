import numpy as np
import math
import gymnasium as gym
from typing import Optional, Union, List
from sim_env.car import Car
from sim_env.renderer import Renderer
from sim_env.parameters import PI
from sim_env.parking import ParallelParking, PerpendicularParking


class Parking(gym.Env):
    """
    A Gymnasium environment for the parking simulation.

    This environment simulates a parking scenario where an autonomous vehicle (agent)
    attempts to park in a designated space while navigating obstacles and constraints.

    Attributes:
        metadata (dict): Defines available modes and configurations, including:
            - 'render_modes': ['human', 'no_render']
            - 'action_types': ['continuous', 'discrete']
            - 'parking_types': ['parallel', 'perpendicular']
            - 'training_modes': ['on', 'off']

        render_mode (str): The selected rendering mode for visualization ('human' or 'no_render').
        action_type (str): Defines the type of actions ('continuous' or 'discrete').
        parking_type (str): Specifies the parking type ('parallel' or 'perpendicular').
        training_mode (str): Defines whether the environment is in training mode ('on' or 'off').
        config (Config): Configuration object storing environment parameters.
        renderer (Renderer): The rendering system used to visualize the parking environment.

        observation_space (gym.spaces.Box): Defines the state representation of the environment.
        action_space (gym.spaces.Discrete or gym.spaces.Box): Defines the action space for the agent.

        parking_strategy (ParallelParking or PerpendicularParking):
            Determines the parking lot structure and rules based on the selected parking type.

        state (np.ndarray or None): The current state of the environment.
        terminated (bool or None): Indicates if the episode has ended.
        truncated (bool or None): Indicates if the episode ended due to reaching max steps.
        run_steps (int or None): Tracks the number of steps taken in an episode.
        side (int or None): Indicates the side of the environment where the parking lot is located.
        parking_lot (np.ndarray or None): The coordinates defining the parking lot position.
        parking_lot_vertices (np.ndarray or None): The corner vertices of the parking lot.
        car (Car or None): The autonomous vehicle (agent) in the environment.
        static_cars_vertices (np.ndarray or None): The vertices of static parked cars in the environment.
        static_parking_lot_vertices (np.ndarray or None): The vertices of additional obstacles in the parking lot.

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
        elif self.config.state_type == 'type4':
            self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(11,), dtype=np.float32)
        else:
            raise ValueError('State type shall be either type1, type2 or type3')

        # Action type
        if self.action_type == "continuous":
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        elif self.action_type == "discrete":
            self.action_space = gym.spaces.Discrete(6)

        # Rendering settings
        self.renderer = Renderer(self.config.render_config, self.config.wheel_size)

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
        self.v_penalty = self.config.penalty_ratio['velocity']
        self.angle_penalty = self.config.penalty_ratio['angle']

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
            return self._render(self.render_mode)

    def _render(self, mode: str):
        if mode == "human":
            self.renderer.initialize_window()
            self.renderer.draw_static_elements(self.parking_lot_vertices, self.static_parking_lot_vertices, self.static_cars_vertices)
            self.renderer.render(self.car, self.car.loc_old)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        # set the side and initial positions
        self.side = self.parking_strategy.set_initial_loc(self.config.side)
        if self.training_mode == 'off':
            self.parking_lot = self.parking_strategy.set_initial_parking_loc(self.side, self.renderer.window_width,
                                                                             self.renderer.window_height,
                                                                             self.config.window_width_offset,
                                                                             self.config.window_height_offset)
        else:  # 'on'
            self.parking_lot = self.config.default_parking_locations[self.side]

        self.parking_lot_vertices = (self.parking_lot +
                                     self.parking_strategy.get_parking_struct(self.parking_type, self.side))
        while True:
            car_loc = self.parking_strategy.set_initial_car_loc(self.side, self.parking_lot,
                                                                self.config.initial_distance_range,
                                                                self.config.car_loc_randomize_range)
            if not self.check_max_distance(self.parking_lot_vertices, car_loc, self.config.max_distance):
                break
        self.car = Car(car_loc, self.parking_strategy.set_initial_heading(self.parking_type, self.side), self.config)

        self.car.loc_old = self.car.car_loc
        self.static_cars_vertices, self.static_parking_lot_vertices = self.parking_strategy.generate_static_obstacles(
            self.parking_lot, self.side)
        self.state = self.get_normalized_state()

        self.terminated = False
        self.truncated = False
        self.run_steps = 0

        if self.render_mode == 'human':
            self.renderer.reset_render()

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
        elif self.config.state_type == 'type2':
            guidance = self.transform_point(self.parking_lot[0], self.parking_lot[1],
                                            self.car.car_loc[0], self.car.car_loc[1], self.car.psi)
            normalized_guidance = guidance / self.config.max_distance
            state = np.concatenate((normalized_distances, normalized_guidance))  # 10 elements

        # type3 state (velocity)
        elif self.config.state_type == 'type3':
            normalized_velocity = np.array([self.car.v / self.config.velocity_limit])
            state = np.concatenate((normalized_distances, normalized_velocity))  # 9 elements

        # type4 state (both guidance and velocity)
        elif self.config.state_type == 'type4':
            normalized_velocity = np.array([self.car.v / self.config.velocity_limit])
            guidance = self.transform_point(self.parking_lot[0], self.parking_lot[1],
                                            self.car.car_loc[0], self.car.car_loc[1], self.car.psi)
            normalized_guidance = guidance / self.config.max_distance
            state = np.concatenate((normalized_distances, normalized_guidance, normalized_velocity))  # 11 elements

        else:
            raise ValueError('State type shall be either type1, type2 or type3')

        # clip the state value
        state = np.clip(state, a_min=-1, a_max=1)

        return state

    @staticmethod
    def transform_point(x: float, y: float, car_x: float, car_y: float, heading: float) -> np.array(['x', 'y']):
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
        if self.run_steps >= self.config.max_steps:
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
                if self.is_car_in_threshold(self.parking_lot, self.car.car_loc, self.config.center_threshold):
                    reward += 1
                    self.terminated = True
                    print("successful parking")

                    parking_angle = self.get_parking_angle(self.parking_type, self.side)
                    angle_penalty = self.calc_angle_dif(self.car.psi, parking_angle, self.config.max_angle_error, self.angle_penalty)

                    # Adjust reward
                    reward -= angle_penalty
                    return reward

        # type3 (velocity)
        if self.config.reward_type == 'type3':
            if self.is_car_in_parking_lot():
                reward += 1
                self.terminated = True
                print("successful parking")

                # velocity check
                velocity_penalty = min(abs(self.v_penalty * (self.car.v / self.config.velocity_limit)), self.v_penalty)
                # Adjust reward
                reward -= velocity_penalty
                return reward

        # type4 (velocity and guidance reward)
        if self.config.reward_type == 'type4':
            if self.is_car_in_parking_lot():
                if self.is_car_in_threshold(self.parking_lot, self.car.car_loc, self.config.center_threshold):

                    reward += 1
                    self.terminated = True
                    print("successful parking")

                    # velocity check
                    velocity_penalty = min(abs(self.v_penalty * (self.car.v / self.config.velocity_limit)), self.v_penalty)
                    # Adjust reward
                    reward -= velocity_penalty

                    # angle check
                    parking_angle = self.get_parking_angle(self.parking_type, self.side)
                    angle_penalty = self.calc_angle_dif(self.car.psi, parking_angle, self.config.max_angle_error, self.angle_penalty)

                    # Adjust reward
                    reward -= angle_penalty
                return reward

        # type5 (velocity, guidance and distance reward)
        if self.config.reward_type == 'type5':
            if self.is_car_in_parking_lot():
                if self.is_car_in_threshold(self.parking_lot, self.car.car_loc, self.config.center_threshold):

                    reward += 1
                    self.terminated = True
                    print("successful parking")

                    # velocity check
                    velocity_penalty = min(abs(self.v_penalty * (self.car.v / self.config.velocity_limit)), self.v_penalty)
                    # Adjust reward
                    reward -= velocity_penalty

                    # angle check
                    parking_angle = self.get_parking_angle(self.parking_type, self.side)
                    angle_penalty = self.calc_angle_dif(self.car.psi, parking_angle, self.config.max_angle_error,
                                                        self.angle_penalty)

                    # Adjust reward
                    reward -= angle_penalty

                    # distance check
                    distance = self.parking_lot - self.car.car_loc
                    distance_penalty_x = min(abs(0.1 * (distance[0] / self.config.center_threshold)), 0.1)
                    distance_penatly_y = min(abs(0.1 * (distance[1] / self.config.center_threshold)), 0.1)

                    # Adjust reward
                    reward -= (distance_penalty_x + distance_penatly_y)

                    return reward

        return reward

    @staticmethod
    def is_car_in_threshold(parking_lot: np.ndarray, car_loc: np.ndarray, center_threshold: np.float32) -> bool:
        """
        Determines whether the car has successfully parked within the designated parking lot.
        The function checks if the car's center is within a defined threshold distance from the parking lot center.

        Parameters:
            parking_lot (np.ndarray): The [x, y] coordinates of the parking lot center.
            car_loc (np.ndarray): The [x, y] coordinates of the car's center.
            center_threshold (np.float32): The maximum allowable distance between the car's center
                                            and the parking lot center for a successful parking.

        Returns:
            bool: True if the car is within the parking lot and within the threshold distance, False otherwise.
        """
        distance = abs(parking_lot - car_loc)
        if distance[0] <= center_threshold and distance[1] <= center_threshold:
            return True
        return False

    @staticmethod
    def get_parking_angle(parking_type: str, side: int) -> Union[float, List[float]]:
        """
        Determines the expected parking angle based on parking type and side.

        Parameters:
            parking_type (str): The type of parking ("parallel" or "perpendicular").
            side (int): The side where the parking lot is located (1 to 4).

        Returns:
            Union[float, List[float]]: The expected parking angle.
                - A single float for perpendicular parking.
                - A list of floats for parallel parking.
        """
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
    def calc_angle_dif(psi: float, parking_angle: Union[float, List[float]], max_angle_error: np.float32,
                       angle_penalty: float) -> float:
        """
        Calculates the angle difference penalty.

        Parameters:
            psi (float): The current heading angle of the car.
            parking_angle (Union[float, List[float]]): The target parking angle,
                which can be a single float or a list of possible angles.
            max_angle_error (float): The maximum allowed angle error.
            angle_penalty (float): The maximum angle penalty value

        Returns:
            float: The calculated angle penalty.
        """
        # calculate the angle error
        if isinstance(parking_angle, list):
            angle_errors = [np.abs((psi - angle + PI) % (2 * PI) - PI) for angle in parking_angle]
            angle_error = min(angle_errors)
        else:
            angle_error = np.abs((psi - parking_angle + PI) % (2 * PI) - PI)
        angle_penalty = min(angle_penalty * (angle_error / max_angle_error), angle_penalty)
        return angle_penalty

    @staticmethod
    def check_cross_border(parking_lot_vertices: np.ndarray, side: int, car_vertices: np.ndarray) -> bool:
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
        """
        Checks if the car's vertices (corners) are entirely within the parking lot.
        This function verifies whether all four corners of the car remain within the defined parking lot boundaries.

        Returns:
            bool: True if the entire car is within the parking lot, False otherwise.
        """
        xy1, xy2, xy3, xy4 = self.parking_lot_vertices
        # Check if all car corners are within the parking area
        for corner in self.car.car_vertices:
            if not self.check_boundary(xy1, xy3, corner):
                return False
        return True

    def check_collision(self) -> bool:
        """
        Determines whether the car has collided with any static obstacles.

        The function iterates through all static parked cars in the environment and checks
        if any corner of the agent's car overlaps with the boundary of another car.

        Returns:
            bool: True if a collision is detected, False otherwise.
        """
        for static_car_vertex in self.static_cars_vertices:
            xy1, xy2, xy3, xy4 = static_car_vertex
            for car_vertex in self.car.car_vertices:
                if self.check_boundary(xy1, xy3, car_vertex):
                    return True
        return False

    @staticmethod
    def check_max_distance(parking_lot_vertices: np.ndarray, car_loc: np.ndarray, max_distance: np.float32) -> bool:
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
    def check_boundary(xy1: np.ndarray, xy2: np.ndarray, obj: np.ndarray) -> bool:
        """
        check if obj is in between xy1 and xy2

        Parameters:
            xy1: top right (x,y) position
            xy2: bottom left (x,y) position
            obj: targeted object (x,y) position

        Return:
            bool
        """
        if xy2[0] <= obj[0] <= xy1[0] and xy2[1] <= obj[1] <= xy1[1]:
            return True
        return False
