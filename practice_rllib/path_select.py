# to select the folder/file path depending on the development environment(Win/Mac)

import os
import platform


def select_path(env_name):
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

def get_os_info():
    os_name = platform.system()
    os_name = os_name.replace("\\", "/")
    return os_name

def get_current_path():
    current_path = os.getcwd()
    current_path = current_path.replace("\\", "/")
    return current_path
