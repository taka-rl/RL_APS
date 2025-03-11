from typing import Union, Tuple
import numpy as np

PI = np.pi
PIXEL_TO_METER_SCALE = np.float32(0.05)


class CarSize:
    """
    Represents the dimensions of the car.

    Attributes:
        length (np.float32): Length of the car in meters (default: 4.0m).
        width (np.float32): Width of the car in meters (default: 2.0m).
        car_struct (np.ndarray): 2D array defining the car's four corner coordinates in meters.
        car_struct_2 (np.ndarray): Alternative 2D array representation for perpendicular parking.

    Default car size:
        - Length: 4 meters
        - Width: 2 meters
    """
    def __init__(self, length: float = 4.0, width: float = 2.0):
        self.length = np.float32(length)
        self.width = np.float32(width)
        self.car_struct = np.array([[+self.length / 2, +self.width / 2],
                                    [+self.length / 2, -self.width / 2],
                                    [-self.length / 2, -self.width / 2],
                                    [-self.length / 2, +self.width / 2]],
                                   dtype=np.float32)

        self.car_struct_2 = np.array([[+self.width / 2, +self.length / 2],
                                      [+self.width / 2, -self.length / 2],
                                      [-self.width / 2, -self.length / 2],
                                      [-self.width / 2, +self.length / 2]],
                                     dtype=np.float32)


class WheelSize:
    """
    Represents the dimensions and positions of the car's wheels.

    Attributes:
        length (np.float32): Length of a single wheel in meters (default: 0.75m).
        width (np.float32): Width of a single wheel in meters (default: 0.35m).
        wheel_struct (np.ndarray): 2D array defining the wheel's four corner coordinates in meters.
        wheel_pos (np.ndarray): Coordinates of the four wheels relative to the car.

    Default wheel size:
        - Length: 0.75 meters
        - Width: 0.35 meters

    Wheel positions:
        - Top right: (1.25, 0.75)
        - Bottom right: (1.25, -0.75)
        - Bottom left: (-1.25, -0.75)
        - Top left: (-1.25, 0.75)
    """

    def __init__(self, length: float = 0.75, width: float = 0.35):
        self.length = np.float32(length)
        self.width = np.float32(width)

        self.wheel_struct = np.array([[+self.length / 2, +self.width / 2],
                                      [+self.length / 2, -self.width / 2],
                                      [-self.length / 2, -self.width / 2],
                                      [-self.length / 2, +self.width / 2]],
                                     dtype=np.float32)
        self.wheel_pos = np.array([[1.25, 0.75],
                                   [1.25, -0.75],
                                   [-1.25, -0.75],
                                   [-1.25, 0.75]],
                                  dtype=np.float32)


class ParkingLotSize:
    """
    Represents different parking lot configurations.

    Attributes:
        length (np.float32): Length of the parking lot in meters (default: 6.0m).
        width (np.float32): Width of the parking lot in meters (default: 4.0m).
        parallel_horizontal (np.ndarray): Array defining the parking structure for horizontal parallel parking.
        parallel_vertical (np.ndarray): Array defining the parking structure for vertical parallel parking.
        perpendicular_horizontal (np.ndarray): Array defining the parking structure for
                                                horizontal perpendicular parking.
        perpendicular_vertical (np.ndarray): Array defining the parking structure for vertical perpendicular parking.
        offset_parallel (np.float32): Offset distance for static obstacles in parallel parking.
        offset_perpendicular (np.float32): Offset distance for static obstacles in perpendicular parking.

    Default parking lot size:
        - Length: 6 meters
        - Width: 4 meters
    """

    def __init__(self, length: float = 6.0, width: float = 4.0):
        self.length = np.float32(length)
        self.width = np.float32(width)

        self.parallel_horizontal = np.array([
            [+self.length / 2, +self.width / 2],
            [+self.length / 2, -self.width / 2],
            [-self.length / 2, -self.width / 2],
            [-self.length / 2, +self.width / 2]],
            dtype=np.float32)

        self.parallel_vertical = np.array([
            [+self.width / 2, +self.length / 2],
            [+self.width / 2, -self.length / 2],
            [-self.width / 2, -self.length / 2],
            [-self.width / 2, +self.length / 2]],
            dtype=np.float32)

        self.perpendicular_horizontal = np.array([
            [+self.width / 2, +self.length / 2],
            [+self.width / 2, -self.length / 2],
            [-self.width / 2, -self.length / 2],
            [-self.width / 2, +self.length / 2]],
            dtype=np.float32)

        self.perpendicular_vertical = np.array([
            [+self.length / 2, +self.width / 2],
            [+self.length / 2, -self.width / 2],
            [-self.length / 2, -self.width / 2],
            [-self.length / 2, +self.width / 2]],
            dtype=np.float32)

        self.offset_parallel = np.float32(8.0)
        self.offset_perpendicular = np.float32(4.0)


class Config:
    """
    Stores all configuration parameters for the parking environment.

    Attributes:
        car_size (CarSize): Car size settings.
        wheel_size (WheelSize): Wheel size settings.
        parking_lot_size (ParkingLotSize): Parking lot size settings.

        reward_type (str): Reward type used in reinforcement learning (default: 'type1').
            - type1: Default reward, primarily based on parking success.
            - type2: Includes a guidance reward, encouraging the agent to park parallel to the parking lot.
        state_type (str): State representation type used in reinforcement learning (default: 'type1').
            - type1: The agent receives relative coordinate data for each (x, y) vertex between the parking lot and
                        the agent location. It has 8 elements.
            - type2: The agent receives relative coordinate data for each (x, y) vertex between the parking lot and
                        the agent location, plus an additional guidance distance to help with alignment.
                        It has 10 elements.
            - type3: The agent receives relative coordinate data for each (x, y) vertex between the parking lot and
                        the agent location, along with its velocity. It has 9 elements.

        acceleration_limit (np.float32): Maximum acceleration limit in m/s².
        steering_limit (np.float32): Maximum steering angle limit in radians.
        velocity_limit (np.float32): Maximum velocity limit in m/s.
        dt (np.float32): Time step used in simulation.
        max_distance (np.float32): Maximum distance for parking in meters.
        max_steps (int): Maximum number of steps per episode.

        max_angle_error (np.float32): Maximum allowable angle error for guidance reward in radians.
        center_threshold (np.float32): Threshold for parking center alignment in meters.

        fps (int): Frames per second for rendering.
        window_width (int): Width of the simulation window in pixels.
        window_height (int): Height of the simulation window in pixels.
        grid_size (int): Grid size for rendering in pixels.
        window_width_offset (int): Offset to keep parking within the screen boundaries in pixels.
        window_height_offset (int): Offset to keep parking within the screen boundaries in pixels.

        colors (dict): Dictionary containing RGB tuples for different UI elements.

        default_parking_locations (dict): Dictionary mapping parking lot sides (1-4) to their respective locations.
        side (Union[int, Tuple[int, ...]]): Defines which side of the environment the parking lot is placed.
                                            It can be a single integer(fixed side) or a tuple(randomized side selection)
                                                1: Bottom side.
                                                2: Top side.
                                                3: Left side.
                                                4: Right side.
        car_loc_randomize_range (tuple): Range for randomizing car initial position in meters.
        initial_distance_range (tuple): Range for setting the initial distance between car and parking lot in meters.
        heading_angle_range (dict): Dictionary defining possible initial heading angles for the car in radians.

    """

    def __init__(self,
                 car_length: float = 4.0, car_width: float = 2.0,
                 wheel_length: float = 0.75, wheel_width: float = 0.35,
                 parking_length: float = 6.0, parking_width: float = 4.0,
                 max_distance: float = 25.0, max_steps: int = 80,
                 acceleration_limit: float = 1.0, steering_limit: float = PI/4, velocity_limit: float = 10.0,
                 max_angle_error: float = PI/12, center_threshold: float = 1.0,
                 reward_type: str = 'type1', state_type: str = 'type1',
                 side: Union[int, Tuple[int, ...]] = 1, default_parking_locations: dict = None,
                 car_loc_randomize_range: tuple = (-5, 5), initial_distance_range: tuple = (7.5, 15.0),
                 heading_angle_range: dict = None
                 ):
        self.car_size = CarSize(car_length, car_width)
        self.wheel_size = WheelSize(wheel_length, wheel_width)
        self.parking_lot_size = ParkingLotSize(parking_length, parking_width)

        # Reward and State settings
        self.reward_type = reward_type
        self.state_type = state_type

        # Action limits
        self.acceleration_limit = np.float32(acceleration_limit)
        self.steering_limit = np.float32(steering_limit)
        self.velocity_limit = np.float32(velocity_limit)
        self.dt = np.float32(0.1)
        self.max_distance = np.float32(max_distance)
        self.max_steps = max_steps

        # Guidance reward
        self.max_angle_error = np.float32(max_angle_error)
        self.center_threshold = np.float32(center_threshold)

        # Rendering settings
        self.fps = 30
        self.window_width = 800
        self.window_height = 600
        self.grid_size = 20
        self.window_width_offset = 100
        self.window_height_offset = 50

        self.colors = {
            "RED": (255, 100, 100),
            "GREEN": (0, 255, 0),
            "BLUE": (100, 200, 255),
            "YELLOW": (200, 200, 0),
            "BLACK": (0, 0, 0),
            "GREY": (100, 100, 100),
            "WHITE": (255, 255, 255),
            "GRID_COLOR": (200, 200, 200)
        }

        # Parking lot and Car location, and car's heading angle for training
        self.default_parking_locations = default_parking_locations or {1: np.array([15.0, 2.5]),
                                                                       2: np.array([15.0, 27.5]),
                                                                       3: np.array([2.5, 15.0]),
                                                                       4: np.array([37.5, 15.0])}

        if isinstance(side, int):
            if side not in (1, 2, 3, 4):
                raise ValueError("side must be an integer between 1 and 4.")
        elif isinstance(side, tuple):
            if not all(isinstance(s, int) and s in (1, 2, 3, 4) for s in side):
                raise ValueError("Each value in side tuple must be an integer between 1 and 4.")
        else:
            raise TypeError("side must be either an int (fixed side) or a tuple (randomized side selection).")
        self.side = side
        self.car_loc_randomize_range = car_loc_randomize_range
        self.initial_distance_range = initial_distance_range
        self.heading_angle_range = heading_angle_range or {
                    "perpendicular": {
                        1: (PI/12 * 5, PI/12 * 7),
                        2: (-PI/12 * 7, -PI/12 * 5),
                        3: (-PI/12, PI/12),
                        4: (PI - PI/12, PI + PI/12),
                    },
                    "parallel": {
                        1: (PI/6, PI/3),
                        2: (-PI/6, -PI/3),
                        3: (-PI/12, PI/12),
                        4: (-PI/12 * 11, PI/12 * 11),
                    }
                }
