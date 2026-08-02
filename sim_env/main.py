import time
import torch
import numpy as np
from pathlib import Path
from ray.rllib.core.rl_module import RLModule

from parameters import Config, PI
from parking_env import Parking
from training.utility import create_folder_path


def compute_action(rl_module: RLModule, action_dist_class, obs: np.ndarray) -> np.ndarray:
    """
    Computes a deterministic action given observation.

    Parameters:
        rl_module: Restored trained RLModule.
        action_dist_class: Distribution class used during inference.
        obs: One unbatched observation.

    Returns:
        action (np.ndarray): Deterministic action
    """
    fwd_ins = {"obs": torch.Tensor([obs])}
    fwd_outputs = rl_module.forward_inference(fwd_ins)
    action_dist = action_dist_class.from_logits(fwd_outputs["action_dist_inputs"]).to_deterministic().sample()
    return action_dist[0].detach().cpu().numpy()


if __name__ == '__main__':

    config = Config(car_length=4.0, car_width=2.0,
                    wheel_length=0.75, wheel_width=0.35,
                    parking_length=6.0, parking_width=4.0,
                    max_distance=25.0, max_steps=80,
                    acceleration_limit=1.0, steering_limit=PI/4, velocity_limit=2.77778,
                    max_angle_error=PI/12, center_threshold=0.5, penalty_ratio={'angle': 0.25, 'velocity': 0.25},
                    reward_type='type1', state_type='type1',
                    side=(1, 2, 3, 4), car_loc_randomize_range=(-5, 5), initial_distance_range=(7.5, 15.0)
                    )
    env_config = {"render_mode": "human",
                  "action_type": "continuous",
                  "parking_type": "perpendicular",
                  "training_mode": "off",
                  'config': config}

    env = Parking(env_config)

    folder_name = 'PPO_perpendicular_continuous_100_r1_s1_all'  # trained_agent folder
    folder_path = create_folder_path(env_config, is_training=False)
    checkpoint_path = Path(folder_path) / folder_name

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_path}")

    # Load RLModule from checkpoint
    rl_module = RLModule.from_checkpoint(
        checkpoint_path / "learner_group" / "learner" / "rl_module" / "default_policy"
    )

    action_dist_class = rl_module.get_inference_action_dist_cls()

    episode_reward = 0
    for i in range(10):
        episode_reward = 0
        terminated = truncated = False
        obs, info = env.reset()
        actions = []

        while not terminated and not truncated:
            action = compute_action(rl_module, action_dist_class, obs)
            obs, reward, terminated, truncated, info = env.step(action)
            time.sleep(0.1)
            episode_reward += reward
        print(f'{i}: {episode_reward}')

    env.close()
