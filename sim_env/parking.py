from typing import Union, Tuple
import numpy as np
import random
from sim_env.parameters import Config, PIXEL_TO_METER_SCALE


class BaseParking:
    def __init__(self, config: Config):
        self.config = config
        self.parking_lot_size = self.config.parking_lot_size
        self.car_size = self.config.car_size

    @staticmethod
    def set_initial_loc(side: Union[int, Tuple[int, ...]]) -> int:
        """
        Set the parking lot location randomly

        Parameters:
            side (Union[int, Tuple[int, ...]]):
                - If int, return as is (fixed side).
                - If tuple, randomly choose a side.

        Return:
            int: The selected parking lot side.
        """
        if isinstance(side, int):
            return side
        if isinstance(side, tuple):
            return random.choice(side)
        raise ValueError(f"Invalid side type: {type(side)}. Expected int or tuple.")

    def get_parking_struct(self, parking_type: str, side: int) -> np.ndarray:
        """
        Get the parking structure based on the parking type.

        Parameters:
            parking_type (str): The type of parking arrangement.
            side (int): The side of the parking lot

        Returns:
            np.ndarray: The vertices for parking space structure.
        """
        if parking_type == "parallel":
            if side in [1, 2]:
                return self.parking_lot_size.parallel_horizontal
            else:
                return self.parking_lot_size.parallel_vertical

        else:  # perpendicular
            if side in [1, 2]:
                return self.parking_lot_size.perpendicular_horizontal
            else:
                return self.parking_lot_size.perpendicular_vertical

    def get_car_struct(self, parking_type: str, side: int):
        """
        Get the car structure based on the parking type.

        Parameters:
            parking_type (str): The type of parking arrangement.
            side (int): The side of the parking lot

        Returns:
            np.ndarray: The vertices for parking space structure.
        """
        if parking_type == "parallel":
            return self.car_size.car_struct if side in [1, 2] else self.car_size.car_struct_2

        else:  # perpendicular
            return self.car_size.car_struct if side in [3, 4] else self.car_size.car_struct_2

    @staticmethod
    def set_initial_car_loc(side: int, parking_loc,
                            initial_distance_range: tuple, car_loc_randomized_range: tuple) -> np.array(['x', 'y']):
        """
        Determines the initial car location based on the parking side and randomized distance.

        This function sets the initial position of the car in relation to the parking lot, ensuring that
        it starts at a reasonable distance for a parking maneuver. The location is randomized within a given range.

        Parameters:
            side (int): Determines the parking lot side on the map:
                - `1`: Car is placed below the parking lot.
                    - `x` is randomly chosen from `car_loc_randomized_range` around `parking_loc[0]`.
                    - `y` is set to `parking_loc[1] + init_dist`.
                - `2`: Car is placed above the parking lot.
                    - `x` is randomly chosen from `car_loc_randomized_range` around `parking_loc[0]`.
                    - `y` is set to `parking_loc[1] - init_dist`.
                - `3`: Car is placed to the left of the parking lot.
                    - `x` is set to `parking_loc[0] + init_dist`.
                    - `y` is randomly chosen from `car_loc_randomized_range` around `parking_loc[1]`.
                - `4`: Car is placed to the right of the parking lot.
                    - `x` is set to `parking_loc[0] - init_dist`.
                    - `y` is randomly chosen from `car_loc_randomized_range` around `parking_loc[1]`.

            parking_loc (np.ndarray): The `[x, y]` coordinates of the parking lot in meters.

            initial_distance_range (tuple): The range `(min, max)` within which the initial distance
                between the car and the parking lot is randomly selected.

            car_loc_randomized_range (tuple): The range `(min, max)` within which the x or y position
                (depending on side) is randomly selected for additional randomness in placement.

        Returns:
            np.ndarray: The `[x, y]` coordinates representing the initial position of the car in meters.
        """

        init_dist = random.uniform(initial_distance_range[0], initial_distance_range[1])

        if side == 1:
            x_car = parking_loc[0] + random.uniform(car_loc_randomized_range[0], car_loc_randomized_range[1])
            y_car = parking_loc[1] + init_dist
        elif side == 2:
            x_car = parking_loc[0] + random.uniform(car_loc_randomized_range[0], car_loc_randomized_range[1])
            y_car = parking_loc[1] - init_dist
        elif side == 3:
            x_car = parking_loc[0] + init_dist
            y_car = parking_loc[1] + random.uniform(car_loc_randomized_range[0], car_loc_randomized_range[1])
        else:
            x_car = parking_loc[0] - init_dist
            y_car = parking_loc[1] + random.uniform(car_loc_randomized_range[0], car_loc_randomized_range[1])

        return np.array([x_car, y_car])

    @staticmethod
    def set_initial_parking_loc(side: int, window_w: int, window_h: int,
                                window_w_offset: int, window_h_offset: int) -> np.array(['x', 'y']):
        """
        Determines the initial location of the parking lot based on the given side.

        This function sets the parking lot's position on the map, ensuring proper placement while maintaining
        a margin from the screen edges. The position is adjusted based on predefined offsets and converted
        from pixels to meters using `PIXEL_TO_METER_SCALE`.

        Parameters:
            side (int): Specifies which side of the map the parking lot is placed:
                - `1`: Parking lot is at the **bottom** of the map.
                    - `x` is randomly chosen between `window_w_offset` and `window_w - window_w_offset`.
                    - `y` is set to `window_h_offset` (bottom).
                - `2`: Parking lot is at the **top** of the map.
                    - `x` is randomly chosen between `window_w_offset` and `window_w - window_w_offset`.
                    - `y` is set to `window_h - window_h_offset` (top).
                - `3`: Parking lot is on the **left** side of the map.
                    - `x` is set to `window_h_offset` (left).
                    - `y` is randomly chosen between `window_h_offset` and `window_h - window_h_offset`.
                - `4`: Parking lot is on the **right** side of the map.
                    - `x` is set to `window_w - window_w_offset` (right).
                    - `y` is randomly chosen between `window_h_offset` and `window_h - window_h_offset`.

            window_w (int): The width of the simulation window (in pixels).
            window_h (int): The height of the simulation window (in pixels).
            window_w_offset (int): The margin to avoid placing the parking lot too close to the left or right edges.
            window_h_offset (int): The margin to avoid placing the parking lot too close to the top or bottom edges.

        Returns:
            np.ndarray: A `[x, y]` array representing the center location of the parking lot in meters.
        """
        if side == 1:
            x_parking = random.uniform(window_w_offset, window_w - window_w_offset) * PIXEL_TO_METER_SCALE
            y_parking = window_h_offset * PIXEL_TO_METER_SCALE
        elif side == 2:
            x_parking = random.uniform(window_w_offset, window_w - window_w_offset) * PIXEL_TO_METER_SCALE
            y_parking = (window_h - window_h_offset) * PIXEL_TO_METER_SCALE
        elif side == 3:
            x_parking = window_h_offset * PIXEL_TO_METER_SCALE
            y_parking = random.uniform(window_h_offset, window_h - window_h_offset) * PIXEL_TO_METER_SCALE
        else:
            x_parking = (window_w - window_w_offset) * PIXEL_TO_METER_SCALE
            y_parking = random.uniform(window_h_offset, window_h - window_h_offset) * PIXEL_TO_METER_SCALE

        return np.array([x_parking, y_parking])

    def set_initial_heading(self, parking_type: str, side: int):
        """Set car's initial heading angle"""
        if parking_type not in ["perpendicular", "parallel"]:
            raise ValueError(f"Invalid parking type: {parking_type}. Must be 'perpendicular' or 'parallel'.")

        if side not in self.config.heading_angle_range[parking_type]:
            raise ValueError(f"Invalid side value: {side}. Must be 1, 2, 3, or 4.")

        heading_range = self.config.heading_angle_range[parking_type][side]
        return random.uniform(heading_range[0], heading_range[1])


class ParallelParking(BaseParking):
    def __init__(self, config: Config):
        super().__init__(config)

    def generate_static_obstacles(self, parking_lot, side: int):
        static_cars_vertices = []
        static_parking_vertices = []

        offset = self.parking_lot_size.offset_parallel

        if side in [1, 2]:
            static_cars_loc = np.array([[parking_lot[0] + offset, parking_lot[1]],
                                        [parking_lot[0] - offset, parking_lot[1]]])
        else:
            static_cars_loc = np.array([[parking_lot[0], parking_lot[1] + offset],
                                        [parking_lot[0], parking_lot[1] - offset]])

        parking_struct = self.get_parking_struct(parking_type="parallel", side=side)
        car_struct = self.get_car_struct(parking_type="parallel", side=side)
        for loc in static_cars_loc:
            static_cars_vertices.append(car_struct + loc)
            static_parking_vertices.append(parking_struct + loc)
        return static_cars_vertices, static_parking_vertices


class PerpendicularParking(BaseParking):
    def __init__(self, config: Config):
        super().__init__(config)

    def generate_static_obstacles(self, parking_lot, side: int):
        static_cars_vertices = []
        static_parking_vertices = []

        offset = self.parking_lot_size.offset_perpendicular

        if side in [1, 2]:
            static_cars_loc = np.array([[parking_lot[0] + offset, parking_lot[1]],
                                        [parking_lot[0] - offset, parking_lot[1]]])
        else:
            static_cars_loc = np.array([[parking_lot[0], parking_lot[1] + offset],
                                        [parking_lot[0], parking_lot[1] - offset]])

        parking_struct = self.get_parking_struct(parking_type="perpendicular", side=side)
        car_struct = self.get_car_struct(parking_type="perpendicular", side=side)
        for loc in static_cars_loc:
            static_cars_vertices.append(car_struct + loc)
            static_parking_vertices.append(parking_struct + loc)
        return static_cars_vertices, static_parking_vertices
