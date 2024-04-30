import os
import platform
import tempfile
from ray.tune.logger import UnifiedLogger
from datetime import datetime


def custom_log_creator(custom_str: str, env_config: dict):
    """
    Set a folder for the training

    Parameter:
        custom_str: parking_type such as parallel or perpendicular
    Return:

    """
    tmp_path = "/" + env_config.get("parking_type") + "/" + env_config.get("action_type") + "/training_result/"
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
    tmp_path = "/" + env_config.get("parking_type") + "/" + env_config.get("action_type") + "/trained_agent/"
    create_training_folder(get_current_path() + tmp_path + logdir_prefix)
    return get_current_path() + "/trained_agent/" + logdir_prefix


def set_path() -> str:
    """
    Set the folder path for the agent depending on the development environment(Win/Mac)

    Return:
        str: the folder/file path
    """
    # get the current folder path and OS info
    current_path = get_current_path().replace("sim_env", "training")
    os_name = get_os_info()

    # depending on OS(Win/Mac)
    tmp_path = "/trained_agent/"

    if os_name == "Windows":
        checkpoint_path = current_path + tmp_path
    else:
        checkpoint_path = current_path + tmp_path
    return checkpoint_path


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
