import numpy as np
import random
from sim_env.parameters import Config, PIXEL_TO_METER_SCALE


class BaseParking:
    def __init__(self, config: Config):
        self.config = config
        self.parking_lot_size = self.config.parking_lot_size
        self.car_size = self.config.car_size

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
    def set_initial_parking_loc(side: int, window_w: int, window_h: int) -> np.array(['x', 'y']):
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
            x_parking = random.uniform(100, window_w - 100) * PIXEL_TO_METER_SCALE
            y_parking = 50 * PIXEL_TO_METER_SCALE
        elif side == 2:
            x_parking = random.uniform(100, window_w - 100) * PIXEL_TO_METER_SCALE
            y_parking = 550 * PIXEL_TO_METER_SCALE
        elif side == 3:
            x_parking = 50 * PIXEL_TO_METER_SCALE
            y_parking = random.uniform(100, window_h - 100) * PIXEL_TO_METER_SCALE
        else:
            x_parking = 750 * PIXEL_TO_METER_SCALE
            y_parking = random.uniform(100, window_h - 100) * PIXEL_TO_METER_SCALE

        return np.array([x_parking, y_parking])

    def set_initial_heading(self, parking_type: str, side: int):
        """Set car's initial heading angle"""
        if side > 4:
            raise ValueError(f"Invalid side value: {side}. Valid values are from 1 to 4")

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
