# to make a path for checkpoint

from datetime import datetime
from path_select import get_current_path


def custom_log_checkpoint(env_name, algo):
    timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    logdir_prefix = "{}_{}".format(algo, timestr)
    checkpoint = get_current_path() +  "/trained_agent/" + env_name + "/" + logdir_prefix
    return checkpoint
