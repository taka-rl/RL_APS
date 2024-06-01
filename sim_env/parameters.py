import numpy as np

# parameters for actions
PI = np.pi
ACCELERATION_LIMIT = 1.0
STEERING_LIMIT = PI / 4
VELOCITY_LIMIT = 10.0
DT = 0.1

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

# parameters for cars and parking lot in the parking environment
'''
Car: Length: 4m, Width: 2m
Perpendicular parking: Length: 6m, Width: 4m
Parallel parking: Length: 6m, Width: 4m
'''
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

PARALLEL_HORIZONTAL = np.array([
    [+CAR_L / 2 + 40 * PIXEL_TO_METER_SCALE, +CAR_W / 2 + 20 * PIXEL_TO_METER_SCALE],
    [+CAR_L / 2 + 40 * PIXEL_TO_METER_SCALE, -CAR_W / 2 - 20 * PIXEL_TO_METER_SCALE],
    [-CAR_L / 2 - 40 * PIXEL_TO_METER_SCALE, -CAR_W / 2 - 20 * PIXEL_TO_METER_SCALE],
    [-CAR_L / 2 - 40 * PIXEL_TO_METER_SCALE, +CAR_W / 2 + 20 * PIXEL_TO_METER_SCALE]],
    dtype=np.float32)  # Adjusted for meters

PARALLEL_VERTICAL = np.array([
    [+CAR_W / 2 + 20 * PIXEL_TO_METER_SCALE, +CAR_L / 2 + 20 * PIXEL_TO_METER_SCALE],
    [+CAR_W / 2 + 20 * PIXEL_TO_METER_SCALE, -CAR_L / 2 - 20 * PIXEL_TO_METER_SCALE],
    [-CAR_W / 2 - 20 * PIXEL_TO_METER_SCALE, -CAR_L / 2 - 20 * PIXEL_TO_METER_SCALE],
    [-CAR_W / 2 - 20 * PIXEL_TO_METER_SCALE, +CAR_L / 2 + 20 * PIXEL_TO_METER_SCALE]],
    dtype=np.float32)  # Adjusted for meters

PERPENDICULAR_HORIZONTAL = np.array([
    [+CAR_W / 2 + 20 * PIXEL_TO_METER_SCALE, +CAR_L / 2 + 20 * PIXEL_TO_METER_SCALE],
    [+CAR_W / 2 + 20 * PIXEL_TO_METER_SCALE, -CAR_L / 2 - 20 * PIXEL_TO_METER_SCALE],
    [-CAR_W / 2 - 20 * PIXEL_TO_METER_SCALE, -CAR_L / 2 - 20 * PIXEL_TO_METER_SCALE],
    [-CAR_W / 2 - 20 * PIXEL_TO_METER_SCALE, +CAR_L / 2 + 20 * PIXEL_TO_METER_SCALE]],
    dtype=np.float32)  # Adjusted for meters

PERPENDICULAR_VERTICAL = np.array([
    [+CAR_L / 2 + 20 * PIXEL_TO_METER_SCALE, +CAR_W / 2 + 20 * PIXEL_TO_METER_SCALE],
    [+CAR_L / 2 + 20 * PIXEL_TO_METER_SCALE, -CAR_W / 2 - 20 * PIXEL_TO_METER_SCALE],
    [-CAR_L / 2 - 20 * PIXEL_TO_METER_SCALE, -CAR_W / 2 - 20 * PIXEL_TO_METER_SCALE],
    [-CAR_L / 2 - 20 * PIXEL_TO_METER_SCALE, +CAR_W / 2 + 20 * PIXEL_TO_METER_SCALE]],
    dtype=np.float32)  # Adjusted for meters

OFFSET_PARALLEL = 160 * PIXEL_TO_METER_SCALE
OFFSET_PERPENDICULAR = 80 * PIXEL_TO_METER_SCALE
'''
OFFSET_PERPENDICULAR
60 + 10
70 + 15
80 + 20
OFFSET_PARALLEL
120 + 20
130 + 30
140 + 40
'''

MAX_DISTANCE = 25.0  # the maximum distance between the car and the parking lot
MAX_STEPS = 80  # the maximum step

# constants for the reward functions
MAX_ANGLE_ERROR = PI / 12
CENTER_THRESHOLD = 0.5
