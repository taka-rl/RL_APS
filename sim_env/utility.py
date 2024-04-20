import os
import platform
import tempfile
from ray.tune.logger import UnifiedLogger
from datetime import datetime


def custom_log_creator(custom_str: str):
    """
    Make a folder for the training

    Parameter:
        custom_str: parking_type such as parallel or perpendicular
    Return:

    """
    custom_path = get_current_path() + "/training/training_result/"
    timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    logdir_prefix = "{}_{}".format(custom_str, timestr)

    def logger_creator(config):
        logdir = tempfile.mkdtemp(prefix=logdir_prefix, dir=custom_path)
        if not os.path.exists(custom_path):
            os.makedirs(custom_path)
        return UnifiedLogger(config, logdir, loggers=None)

    return logger_creator


def custom_log_checkpoint(custom_str: str, algo):
    """
    Make a folder for the training result

    Parameter:
        env_name: environment name
        algo: type of the algorithm

    Return:
        str: folder path

    """
    timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    logdir_prefix = "{}_{}".format(custom_str, algo, timestr)
    return get_current_path() + "/training/trained_agent/" + logdir_prefix


def set_path(env_name: str) -> str:
    """
    Set the folder/file path depending on the development environment(Win/Mac)

    Return:
        str: the folder/file path
    """
    # get the current folder path and OS info
    current_path = get_current_path()
    os_name = get_os_info()

    # depending on OS(Win/Mac)
    tmp_path = "/trained_agent/" + env_name

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
