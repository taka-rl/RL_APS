import numpy as np

PI = np.pi
PIXEL_TO_METER_SCALE = np.float32(0.05)


class CarSize:
    """
    Represents the car size.
    The default car size is as follows:
        Length: 4 meter
        Width: 2 meter
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
    Represents the car size.
    The default wheel size is as follows:
        Length: 0.75 meter
        Width: 0.35 meter

    The center of each wheel position is as follows:
        Top right: 1.25, 0.75
        Bottom right: 1.25, -0.75
        Bottom left: -1.25, -0.75
        Top left: -1.25, 0.75
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
    Presents a variety of parking lot types.
    The default parking lot size is as follows:
        Length: 6 meter
        Width: 4 meter
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
    """Stores all configuration parameters for the environment."""
    def __init__(self,
                 car_length: float = 4.0, car_width: float = 2.0,
                 wheel_length: float = 0.75, wheel_width: float = 0.35,
                 parking_length: float = 6.0, parking_width: float = 4.0,
                 max_distance: float = 25.0, max_steps: int = 80,
                 acceleration_limit: float = 1.0, steering_limit: float = PI / 4, velocity_limit: float = 10.0,
                 max_angle_error: float = PI / 12, center_threshold: float = 1.0,
                 reward_type: str = 'type1', state_type: str = 'type1',
                 side: int = 1, default_parking_locations: dict = None, car_loc_randomize_range: tuple = (-5, 5),
                 initial_distance_range: tuple = (7.5, 15.0), heading_angle_range: dict = None,
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
