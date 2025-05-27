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


## Parallel Parking
### Default Setting

- Parameter settings for trainings

```
config = Config(car_length=4.0, car_width=2.0,
                wheel_length=0.75, wheel_width=0.35,
                parking_length=6.0, parking_width=4.0,
                max_distance=25.0, max_steps=80,
                acceleration_limit=1.0, steering_limit=PI/4, velocity_limit=10.0,
                max_angle_error=PI/12, center_threshold=0.5, penalty_ratio={'angle': 0, 'velocity': 0.5},
                reward_type='type1', state_type='type1',
                side=1, car_loc_randomize_range=(6.0, 7.0), initial_distance_range=(5.0, 7.0)
                )
env_config = {"render_mode": "no_render",
              "action_type": "continuous",
              "parking_type": "parallel",
              "training_mode": "on",
              'config': config}

# for folder names
num_train = 100

```
- Trained agent (Continuous action)



https://github.com/user-attachments/assets/ff5b1ee3-9173-4959-b12e-5466beb40c07



- Training results   (only continuous action)
![image](https://github.com/user-attachments/assets/df1d85e3-6b20-4980-804f-27f796fceed4)
<img width="1296" alt="image" src="https://github.com/user-attachments/assets/b4773ec4-0bf8-4bf7-8609-23038563de82" />
<img width="1297" alt="image" src="https://github.com/user-attachments/assets/5994c669-c7af-4302-bf25-ffb661ebeaf8" />


### Guidance Reward

- Parameter settings for trainings

```
config = Config(car_length=4.0, car_width=2.0,
                wheel_length=0.75, wheel_width=0.35,
                parking_length=6.0, parking_width=4.0,
                max_distance=25.0, max_steps=80,
                acceleration_limit=1.0, steering_limit=PI/4, velocity_limit=10.0,
                max_angle_error=PI/12, center_threshold=1.0, penalty_ratio={'angle': 0.5, 'velocity': 0.0},
                reward_type='type2', state_type='type2',
                side=1, car_loc_randomize_range=(6.0, 7.0), initial_distance_range=(5.0, 7.0)
                )

env_config = {"render_mode": "no_render",
              "action_type": "continuous",
              "parking_type": "parallel",
              "training_mode": "on",
              'config': config}

# for folder names
num_train = 100

```
- Trained agent (Continuous action)


https://github.com/user-attachments/assets/1e0da9e2-f4c6-4ec6-91be-acd11e22d492


- Training results   (only continuous action)
<img width="1279" alt="image" src="https://github.com/user-attachments/assets/42a1febc-3f4c-4aef-aa90-e9056cfc6b9f" />
<img width="1284" alt="image" src="https://github.com/user-attachments/assets/e2279e64-ee55-440d-a044-b32b652bf743" />
<img width="1284" alt="image" src="https://github.com/user-attachments/assets/d704b3a7-fd06-44b8-9fb9-2fd5f8fe8c0d" />


### Velocity Penalty
- Parameter settings for trainings

```
config = Config(car_length=4.0, car_width=2.0,
                wheel_length=0.75, wheel_width=0.35,
                parking_length=6.0, parking_width=4.0,
                max_distance=25.0, max_steps=80,
                acceleration_limit=1.0, steering_limit=PI/4, velocity_limit=10.0,
                max_angle_error=PI/12, center_threshold=1.0, penalty_ratio={'angle': 0.0, 'velocity': 0.5},
                reward_type='type3', state_type='type3',
                side=1, car_loc_randomize_range=(6.0, 7.0), initial_distance_range=(5.0, 7.0)
                )

env_config = {"render_mode": "no_render",
              "action_type": "continuous",
              "parking_type": "parallel",
              "training_mode": "on",
              'config': config}

# for folder names
num_train = 100

```
- Trained agent (Continuous action)  



https://github.com/user-attachments/assets/9b11b7ea-dbe1-47ed-aade-f8b9aa061676




- Training results   (only continuous action)
<img width="1285" alt="image" src="https://github.com/user-attachments/assets/b029a70c-70f7-4e96-90b9-749dbf5a2bd2" />
<img width="1285" alt="image" src="https://github.com/user-attachments/assets/ea9a392b-ecb8-4f28-9c0c-67878eb1347a" />
<img width="1285" alt="image" src="https://github.com/user-attachments/assets/d1690544-87d2-4cc9-9fbf-1285c2178da6" />


### Both Guidance Reward and Velocity Penalty


## Result analysis
This section describes the result plots in detail. 
