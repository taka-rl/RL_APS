import time
import torch
import numpy as np
from pathlib import Path
from ray.rllib.core.rl_module import RLModule

from sim_env.parameters import Config, PI
from sim_env.parking_env import Parking
from training.utility import create_folder_path


def compute_action(module: RLModule, observation: np.ndarray, action_type: str) -> np.ndarray | int:
    """
    Compute a deterministic action from an observation.

    The unbatched observation is converted to a batch of size one and passed
    to the trained RLModule using ``forward_inference()``.

    For a continuous action space, this implementation assumes two action
    dimensions [acceleration, steering]. The mean values of the action
    distribution are used as the deterministic action and clipped to the
    normalized action range [-1.0, 1.0].

    For a discrete action space, the action with the highest logit is selected.

    Parameters:
        module (RLModule): Restored trained RLModule used for inference.
        observation (np.ndarray): Unbatched observation with shape (obs_dim,).
        action_type (str): Type of action space. Must be either "continuous" or "discrete".

    Returns:
        np.ndarray | int:
            For a continuous action space, a deterministic action
            [acceleration, steering] with shape (2,), clipped to [-1.0, 1.0].
            For a discrete action space, the integer index of the action with the highest logit.

    Raises:
        ValueError:
            If ``action_type`` is neither "continuous" nor "discrete".

    Reference:
        https://docs.ray.io/en/releases-2.58.0/rllib/getting-started.html
        #deploy-a-trained-model-for-inference
    """

    # Compute the next action from a batch (B=1) of observations.
    obs_batch = torch.from_numpy(observation).unsqueeze(0)  # add batch B=1 dimension
    model_outputs = module.forward_inference({'obs': obs_batch})

    # Extract the action distribution parameters from the output and dissolve batch dim.
    action_dist_params = model_outputs['action_dist_inputs'][0].numpy()

    if action_type == 'continuous':
        # Continuous action space with two dimensions:
        # [mean_acceleration, mean_steering, log_std_acceleration, log_std_steering].
        # Use the two means as the deterministic (maximum-likelihood) action.
        greedy_action = np.clip(action_dist_params[0:2], a_min=-1.0, a_max=1.0)

    elif action_type == 'discrete':
        # For discrete actions, you should take the argmax over the logits:
        greedy_action = np.argmax(action_dist_params)

    else:
        raise ValueError(
            f"Unsupported action_type: {action_type}. "
            "Expected 'continuous' or 'discrete'."
        )

    return greedy_action


if __name__ == '__main__':

    config = Config(car_length=4.0, car_width=2.0,
                    wheel_length=0.75, wheel_width=0.35,
                    parking_length=6.0, parking_width=4.0,
                    max_distance=25.0, max_steps=80,
                    acceleration_limit=1.0, steering_limit=PI/4, velocity_limit=2.77778,
                    max_angle_error=PI/6, center_threshold=0.5, penalty_ratio={'angle': 0.25, 'velocity': 0.25},
                    reward_type='type4', state_type='type4',
                    side=1, car_loc_randomize_range=(5.5, 6.0), initial_distance_range=(4.0, 5.0),
                    heading_angle_range={"parallel": {1: (PI/6, PI/4)}}
                   )

    env_config = {"render_mode": "human",
                  "action_type": "continuous",
                  "parking_type": "parallel",
                  "training_mode": "off",
                  'config': config}

    env = Parking(env_config)

    folder_name = 'PPO_parallel_continuous_100_r4_s4_b_th05_ar025_vr025'  # trained_agent folder
    folder_path = create_folder_path(env_config, is_training=False)
    checkpoint_path = Path(folder_path) / folder_name

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_path}")

    # Load RLModule from checkpoint
    rl_module = RLModule.from_checkpoint(
        checkpoint_path / "learner_group" / "learner" / "rl_module" / "default_policy"
    )

    for i in range(10):
        episode_reward = 0
        terminated = truncated = False
        obs, info = env.reset()

        while not terminated and not truncated:
            action = compute_action(rl_module, obs, env_config['action_type'])
            obs, reward, terminated, truncated, info = env.step(action)
            time.sleep(0.1)
            episode_reward += reward
        print(f'{i}: {episode_reward}')

    env.close()
