# Training result
This document includes the training results.

## Perpendicular Parking
This section presents the training results trained in the Perpendicular Parking environment.

### Default Setting (Reward/State: type1)

- Parameter settings for trainings

```
config = Config(car_length=4.0, car_width=2.0,
                wheel_length=0.75, wheel_width=0.35,
                parking_length=6.0, parking_width=4.0,
                max_distance=25.0, max_steps=80,
                acceleration_limit=1.0, steering_limit=PI/4, velocity_limit=10.0,
                max_angle_error=PI/12, center_threshold=0.5, penalty_ratio={'angle': 0.25, 'velocity': 0.25},
                reward_type='type1', state_type='type1',
                side=(1, 2, 3, 4), car_loc_randomize_range=(-5, 5), initial_distance_range=(7.5, 15.0)
                )
env_config = {"render_mode": "no_render",
              "action_type": "continuous",  # or "discrete"
              "parking_type": "perpendicular",
              "training_mode": "on",
              'config': config}

num_train = 100
```
- Trained agent (Continuous action)


https://github.com/user-attachments/assets/c788057a-c4f1-496e-ba9c-fbe45c5fc233



- Tensorboard captures


- Difference between Discrete and Continuous action spaces


### Guidance Reward (Reward/State: type2)
- Parameter settings for trainings

```
config = Config(car_length=4.0, car_width=2.0,
                wheel_length=0.75, wheel_width=0.35,
                parking_length=6.0, parking_width=4.0,
                max_distance=25.0, max_steps=80,
                acceleration_limit=1.0, steering_limit=PI/4, velocity_limit=10.0,
                max_angle_error=PI/12, center_threshold=0.5, penalty_ratio={'angle': 0.5, 'velocity': 0.0},
                reward_type='type2', state_type='type2',
                side=(1, 2, 3, 4), car_loc_randomize_range=(-5, 5), initial_distance_range=(7.5, 15.0)
                )
env_config = {"render_mode": "no_render",
              "action_type": "continuous",  # or "discrete"
              "parking_type": "perpendicular",
              "training_mode": "on",
              'config': config}

num_train = 300
```

- Trained agent (Continuous action)


https://github.com/user-attachments/assets/fde361ce-a9a1-4d40-a517-730138c3c8d9


- Tensorboard captures (only continuous action)


- How the agent behaviour improved with this reward

### Velocity Penalty (Reward/State: type3)
- Parameter settings for trainings

```
config = Config(car_length=4.0, car_width=2.0,
                wheel_length=0.75, wheel_width=0.35,
                parking_length=6.0, parking_width=4.0,
                max_distance=25.0, max_steps=80,
                acceleration_limit=1.0, steering_limit=PI/4, velocity_limit=10.0,
                max_angle_error=PI/12, center_threshold=0.5, penalty_ratio={'angle': 0.0, 'velocity': 0.5},
                reward_type='type3', state_type='type3',
                side=(1, 2, 3, 4), car_loc_randomize_range=(-5, 5), initial_distance_range=(7.5, 15.0)
                )
env_config = {"render_mode": "no_render",
              "action_type": "continuous",  # or "discrete"
              "parking_type": "perpendicular",
              "training_mode": "on",
              'config': config}

num_train = 100
```

- Trained agent (Continuous action)


https://github.com/user-attachments/assets/6f751090-fd88-45b8-804a-fdcff530bf10


- Tensorboard captures (only continuous action)


- How the agent behaviour improved with this reward


### Both Guidance Reward and Velocity Penalty (Reward/State: type4)
- Parameter settings for trainings

```
config = Config(car_length=4.0, car_width=2.0,
                wheel_length=0.75, wheel_width=0.35,
                parking_length=6.0, parking_width=4.0,
                max_distance=25.0, max_steps=80,
                acceleration_limit=1.0, steering_limit=PI/4, velocity_limit=10.0,
                max_angle_error=PI/12, center_threshold=0.5, penalty_ratio={'angle': 0.25, 'velocity': 0.25},
                reward_type='type4', state_type='type4',
                side=(1, 2, 3, 4), car_loc_randomize_range=(-5, 5), initial_distance_range=(7.5, 15.0)
                )
env_config = {"render_mode": "no_render",
              "action_type": "continuous",  # or "discrete"
              "parking_type": "perpendicular",
              "training_mode": "on",
              'config': config}

num_train = 300
```
- Trained agent (Continuous action)
  

https://github.com/user-attachments/assets/9fbfc305-6577-42bf-bd06-6ae67add61e9



- Tensorboard captures (only continuous action)


- How the agent behaviour improved with this reward




## Parallel Parking
### Default Setting
### Guidance Reward
### Velocity Penalty
### Both Guidance Reward and Velocity Penalty


