import glob
from pathlib import Path
from typing import Union, Tuple


# utility.py is expected to be: RL_APS/training/utility.py
PROJECT_ROOT = Path(__file__).resolve().parent


def generate_unique_id(target_folder) -> int:
    """
    Generates a unique ID based on the exist folders.
    Returns:
        str: A unique identifier (e.g. 1, 2, 3 and so on).
    """
    # Check the target_folder if the same name exists
    matching_folders = glob.glob(target_folder + '*')

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


def create_folder(folder_path: str | Path) -> None:
    """
    Create a folder, including missing parent folders.
    """
    path = Path(folder_path)
    path.mkdir(parents=True, exist_ok=True)


def create_folder_path(env_config: dict, is_training: bool) -> str:
    """
    Return the base output directory.

    Parameters:
        env_config: Dictionary containing environment configurations
        is_training: Boolean indicating whether or not to create the folder
                    True: training results
                    False: trained checkpoints

    Training results:
        training/training_results/<parking_type>/<action_type>

    Trained checkpoints:
        training/trained_agents/<parking_type>/<action_type>
    """
    base_folder = "training_results" if is_training else "trained_agents"

    folder_path = (
        PROJECT_ROOT
        / base_folder
        / env_config["parking_type"]
        / env_config["action_type"]
    )

    return str(folder_path)


def create_training_result_dir(folder_path: str, folder_name: str) -> str:
    """
    Create and return a directory for TensorBoard and other training logs.
    """
    result_dir = Path(folder_path) / folder_name
    create_folder(result_dir)

    return str(result_dir)


def create_checkpoint_dir(folder_path: str, folder_name: str) -> str:
    """
    Create and return a directory for an RLlib checkpoint.
    """
    checkpoint_dir = Path(folder_path) / folder_name
    create_folder(checkpoint_dir)

    return str(checkpoint_dir)
