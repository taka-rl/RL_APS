import os
import platform
import tempfile
import numpy as np
import random
from ray.tune.logger import UnifiedLogger
from datetime import datetime

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


def set_init_position(side):
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


def custom_log_creator(custom_str: str, env_config: dict):
    """
    Set a folder for the training

    Parameter:
        custom_str: parking_type such as parallel or perpendicular
    Return:

    """
    tmp_path = create_folder_path(env_config=env_config, train_str="/training_result/")
    custom_path = get_current_path() + tmp_path
    timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    logdir_prefix = "{}_{}".format(custom_str, timestr)

    # check the folder existence
    create_training_folder(custom_path)

    def logger_creator(config):
        logdir = tempfile.mkdtemp(prefix=logdir_prefix, dir=custom_path)
        if not os.path.exists(custom_path):
            os.makedirs(custom_path)
        return UnifiedLogger(config, logdir, loggers=None)

    return logger_creator


def custom_log_checkpoint(custom_str: str, env_config: dict, algo):
    """
    Set a folder for the training result

    Parameter:
        env_name: environment name
        algo: type of the algorithm

    Return:
        str: folder path

    """
    timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    logdir_prefix = "{}_{}_{}".format(algo, custom_str, timestr)

    # check the folder existence
    tmp_path = create_folder_path(env_config=env_config, train_str="/trained_agent/")
    create_training_folder(get_current_path() + tmp_path + logdir_prefix)
    return get_current_path() + tmp_path + logdir_prefix


def set_path(env_config) -> str:
    """
    Set the folder path for the agent depending on the development environment(Win/Mac)

    Return:
        str: the folder/file path
    """
    # get the current folder path and OS info
    current_path = get_current_path().replace("sim_env", "training")
    os_name = get_os_info()

    # depending on OS(Win/Mac)
    # check the folder existence
    tmp_path = create_folder_path(env_config=env_config, train_str="/trained_agent/")

    if os_name == "Windows":
        checkpoint_path = current_path + tmp_path
    else:
        checkpoint_path = current_path + tmp_path
    return checkpoint_path


def create_folder_path(env_config: dict, train_str: str):
    return "/" + env_config.get("parking_type") + "/" + env_config.get("action_type") + train_str


def get_os_info() -> str:
    """
    Get OS information

    Return:
        str: OS (like Windows, Mac)
    """
    os_name = platform.system()
    return os_name.replace("\\", "/")


def get_current_path() -> str:
    """
    Get the current folder path

    Return:
         str: the current folder path
    """
    current_path = os.getcwd()
    return current_path.replace("\\", "/")


def is_folder(folder_path) -> bool:
    """
    check if folder_path folder exists or not

     Parameters:
        folder_path (str): The path to the folder to be checked and potentially created.

    Return:
        bool
    """
    if os.path.exists(folder_path):
        return True
    return False


def create_training_folder(folder_path):
    """
    Create a folder for the training.

    Parameters:
        folder_path (str): The path to the folder to be checked and potentially created.
    """
    if not is_folder(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")
