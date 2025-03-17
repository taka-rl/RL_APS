import os
import tempfile
from ray.tune.logger import UnifiedLogger
from datetime import datetime


def custom_log_creator(folder_path: str, custom_str: str):
    """
    Set a folder for the training

    Parameter:
        folder_path: folder path
        custom_str: parking_type such as parallel or perpendicular
    Return:

    """
    custom_path = get_current_path() + folder_path
    timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    logdir_prefix = "{}_{}".format(custom_str, timestr)

    # check the folder existence
    create_folder(custom_path)

    def logger_creator(config):
        logdir = tempfile.mkdtemp(prefix=logdir_prefix, dir=custom_path)
        if not os.path.exists(custom_path):
            os.makedirs(custom_path)
        return UnifiedLogger(config, logdir, loggers=None)

    return logger_creator


def custom_log_checkpoint(folder_path: str, custom_str: str, algo):
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
    create_folder(get_current_path() + folder_path + logdir_prefix)
    return get_current_path() + folder_path + logdir_prefix


def create_folder_path(env_config: dict, reward_type: str, state_type: str, is_training: bool) -> str:
    """Return a folder path depending on is_training value."""
    if is_training:
        return (f'/training_results/{env_config["parking_type"]}/'
                f'{env_config["action_type"]}/reward_{reward_type}/state_{state_type}/')
    else:
        return (f'/trained_agents/{env_config["parking_type"]}/'
                f'{env_config["action_type"]}/reward_{reward_type}/state_{state_type}/')


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


def create_folder(folder_path) -> None:
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
