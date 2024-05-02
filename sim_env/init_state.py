import numpy as np
import random


def set_init_position(side):
    """
    Set the initial car location, parking lot location and heading angle for the training

    Currently, there are 9 initial positions for the car and parking lot in order to encourage the agent to
    try to reverse into the parking lot.

    Parameters:
        side (int)
            - 1: the parking lot is placed on the bottom side
            - 2: the parking lot is placed on the top side
            - 3: the parking lot is placed on the left side
            - 4: the parking lot is placed on the right side

        num_dict (int): a number randomly selected for the initial position

        Returns:
            init_car_loc (np.array): initial car location
            init_parking_lot (np.array): initial parking lot location
            heading_angle (float): initial heading angle
    """
    # parameters for the training
    CAR_LOCATION = {
        1: {
            "1": np.array([18.0, 8.5]),
            "2": np.array([21.0, 6.5]),
            "3": np.array([12.0, 8.5]),
            "4": np.array([9.0, 6.5]),
            "5": np.array([18.0, 9.0]),
            "6": np.array([12.0, 9.0]),
            "7": np.array([16.5, 8.5]),
            "8": np.array([13.5, 8.5]),
            "9": np.array([15.0, 7.5]),
        },
        2: {
            "1": np.array([18.0, 8.5]),
            "2": np.array([21.0, 6.5]),
            "3": np.array([12.0, 8.5]),
            "4": np.array([9.0, 6.5]),
            "5": np.array([12.0, 21.0]),
            "6": np.array([18.0, 21.0]),
            "7": np.array([13.5, 21.5]),
            "8": np.array([16.5, 21.5]),
            "9": np.array([15.0, 22.5]),
        },
        3: {
            "1": np.array([18.0, 8.5]),
            "2": np.array([21.0, 6.5]),
            "3": np.array([12.0, 8.5]),
            "4": np.array([9.0, 6.5]),
            "5": np.array([9.0, 12.0]),
            "6": np.array([9.0, 18.0]),
            "7": np.array([8.5, 13.5]),
            "8": np.array([8.5, 16.5]),
            "9": np.array([7.5, 15.0]),
        },
        4: {
            "1": np.array([18.0, 8.5]),
            "2": np.array([21.0, 6.5]),
            "3": np.array([12.0, 8.5]),
            "4": np.array([9.0, 6.5]),
            "5": np.array([31.0, 18.0]),
            "6": np.array([31.0, 12.0]),
            "7": np.array([31.5, 16.5]),
            "8": np.array([31.5, 13.5]),
            "9": np.array([32.5, 15.0]),
        }
    }

    HEADING_ANGLE = {
        "1": np.pi / 4,
        "2": 0.0,
        "3": np.pi / 4 * 3,
        "4": np.pi,
        "5": np.pi / 3,
        "6": np.pi / 3 * 2,
        "7": np.pi / 12 * 5,
        "8": np.pi / 12 * 7,
        "9": np.pi / 2
    }

    PARKING_LOT = {1: np.array([15.0, 2.5]), 2: np.array([15.0, 27.5]),
                   3: np.array([2.5, 15.0]), 4: np.array([37.5, 15.0])}

    num_dict = str(random.randint(5, 9))
    if side == 1:
        heading_angle = HEADING_ANGLE[num_dict]
    elif side == 2:
        heading_angle = HEADING_ANGLE[num_dict] + np.pi
    elif side == 3:
        heading_angle = HEADING_ANGLE[num_dict] - np.pi / 2
    elif side == 4:
        heading_angle = HEADING_ANGLE[num_dict] + np.pi / 2
    else:
        raise ValueError(f"Invalid side: {side}. Valid options are 1 to 4")

    init_car_loc = CAR_LOCATION[side][num_dict]
    init_parking_lot = PARKING_LOT[side]

    return init_car_loc, init_parking_lot, heading_angle

