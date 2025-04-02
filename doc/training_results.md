# Training Results
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

- Training results  

![image](https://github.com/user-attachments/assets/51bc1042-b0c6-41a8-bd4f-8afedf11e121)

![image](https://github.com/user-attachments/assets/04d5966d-bfad-41b3-8302-a06ac9ebad9e)

![image](https://github.com/user-attachments/assets/ea305173-b705-4831-94f2-a13433c856d4)


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


- Training results  (only continuous action)

![image](https://github.com/user-attachments/assets/2a7b9d43-1071-4591-8a45-5df34d2e75e9)

![image](https://github.com/user-attachments/assets/296622de-8688-41ba-9cb3-b5657ccc9d4c)

![image](https://github.com/user-attachments/assets/7b6b8095-fb9c-4ac5-88ef-3175bb92fa51)



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

num_train = 200
```

- Trained agent (Continuous action)




https://github.com/user-attachments/assets/5023f03c-1565-462c-8f3f-370c0904c578



- Training results   (only continuous action)

![image](https://github.com/user-attachments/assets/01ae2476-291e-47c5-a3ab-6f1e2fb75e4a)

![image](https://github.com/user-attachments/assets/7148baee-f41c-495c-8bca-5f194152c265)

![image](https://github.com/user-attachments/assets/761db4fc-8993-490e-b2a9-7431e82a1ae7)


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



- Training results   (only continuous action)

![image](https://github.com/user-attachments/assets/7231898d-9e96-4f02-b3fa-d8925ed3c680)

![image](https://github.com/user-attachments/assets/4fc835e2-1818-4db8-998b-8a253b695d24)

![image](https://github.com/user-attachments/assets/3918acae-05aa-457c-aa6c-6c0328c96d79)

- How the agent behaviour improved with this reward


## Parallel Parking
### Default Setting
### Guidance Reward
### Velocity Penalty
### Both Guidance Reward and Velocity Penalty


