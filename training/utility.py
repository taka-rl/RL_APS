import os
import tempfile
import glob
from ray.tune.logger import UnifiedLogger
from typing import Union, Tuple


def generate_unique_id(target_folder) -> int:
    """
    Generates a unique ID based on the exist folders.
    Returns:
        str: A unique identifier (e.g. 1, 2, 3 and so on).
    """
    # Check the target_folder if the same name exists
    matching_folders = glob.glob(target_folder)

    # Increment a count
    if matching_folders:
        count = len(matching_folders)
        return count
    else:
        return 0


def convert_side_to_abbr(side):
    """
    Convert side integer(s) to an abbreviation.

    Parameters:
        side: int (1-4) or tuple of ints.

    Returns:
        str: Abbreviated side name.
    """
    side_map = {1: 'b', 2: 't', 3: 'l', 4: 'r'}

    if isinstance(side, int):
        return side_map.get(side, "u")  # "u" for unknown cases
    elif isinstance(side, tuple) or isinstance(side, list):
        if side == (1, 2, 3, 4):
            return 'all'
        sorted_sides = sorted([side_map[s] for s in side if s in side_map])
        return "".join(sorted_sides) if sorted_sides else "u"
    else:
        return "u"


def custom_log_creator(folder_path: str, custom_str: str):
    """
    Set a folder for the training

    Parameter:
        folder_path: folder path
        custom_str: parking_type such as parallel or perpendicular
    Return:

    """
    custom_path = folder_path  # folder path
    logdir_prefix = custom_str  # folder name

    # check the folder existence
    create_folder(custom_path)

    def logger_creator(config):
        logdir = tempfile.mkdtemp(prefix=logdir_prefix, dir=custom_path)

        # Rename the created folder as it contains unnecessary characters
        os.rename(logdir, custom_path+logdir_prefix)
        # if not os.path.exists(custom_path):
            # os.makedirs(custom_path)
        return UnifiedLogger(config, custom_path+logdir_prefix, loggers=None)

    return logger_creator


def custom_log_checkpoint(folder_path: str, folder_name: str):
    """
    Set a folder for the training result

    Parameter:
        folder_path: Path for training results
        folder_name: Structured folder name

    Return:
        str: Full folder path
    """
    logdir_prefix = folder_name

    # check the folder existence
    checkpoint_path = os.path.join(folder_path, logdir_prefix)
    create_folder(checkpoint_path)
    return checkpoint_path


def create_folder_path(env_config: dict, is_training: bool) -> str:
    """
    Return a structured folder path based on environment config.

    Parameters:
        env_config: Dictionary containing environment configurations
        is_training: Boolean flag (True: Training, False: Evaluating)
    Returns:
        str: Structured folder path
    """
    base_folder = "/training_results" if is_training else "/trained_agents"

    folder_path = get_current_path() + f"{base_folder}/{env_config['parking_type']}/{env_config['action_type']}/"

    return folder_path


def create_folder_name(algo: str, env_config: dict, reward_type: str, state_type: str, num_train: int,
                       side: Union[int, Tuple[int]], folder_path: str, threshold: float = None,
                       angle_ratio: float = None, v_ratio: float = None) -> str:
    """
    Return a structured folder path based on environment config.

    Parameters:
        algo: Algorithms for training
        env_config: Dictionary containing environment configurations
        reward_type: Reward function type (e.g., type1, type2)
        state_type: State representation type (e.g., type1, type2)
        num_train: Number of training
        side: int or tuple representing the parking side(s)
        threshold: Threshold value for guidance reward
        angle_ratio: Ratio for guidance reward
        v_ratio: Ratio for velocity
        folder_path: Structured folder path

    Returns:
        str: Structured folder name

    """
    side_str = convert_side_to_abbr(side)
    folder_name = (f"{algo}_{env_config['parking_type']}_"
                   f"{env_config['action_type']}_{num_train}_r{reward_type}_s{state_type}_{side_str}")

    folder_name = folder_name.replace('type', '')

    # for Guidance and velocity rewards
    if reward_type == 'type2':
        threshold = f"{threshold:.1f}".replace(".", "")
        angle_ratio = f"{angle_ratio:.1f}".replace(".", "")
        folder_name = folder_name + f'_th{threshold}_ar{angle_ratio}'

    if reward_type == 'type3':
        v_ratio = f"{v_ratio:.1f}".replace(".", "")
        folder_name = folder_name + f'_vr{v_ratio}'

    if reward_type == 'type4':
        threshold = f"{threshold:.1f}".replace(".", "")
        angle_ratio = f"{angle_ratio:.1f}".replace(".", "")
        v_ratio = f"{v_ratio:.1f}".replace(".", "")
        folder_name = folder_name + f'_th{threshold}_ar{angle_ratio}_vr{v_ratio}'

    # Add id number
    id_num = generate_unique_id(folder_path + folder_name)
    if id_num:
        folder_name = folder_name + f'_{id_num}'

    return folder_name


def get_current_path() -> str:
    """
    Get the current folder path

    Return:
         str: the current folder path
    """
    return os.getcwd().replace("\\", "/")


def is_folder(folder_path) -> bool:
    """
    check if folder_path folder exists or not

     Parameters:
        folder_path (str): The path to the folder to be checked and potentially created.

    Return:
        bool: True if the folder exists, else False.
    """
    return os.path.exists(folder_path)


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
