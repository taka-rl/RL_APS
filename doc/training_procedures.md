# Training procedures
This file describes about folder structures, necessary steps and parameter settings for trainings.

## Folder structure for both trained agents and training results


    RL_APS/training
    │── trained_agents
    │   │── perpendicular
    │   │   │── discrete
    │   │   │   │── folders for trained agent referred to the 'Naming order' below.
    │   │   │── continuous
    │   │   │   │── (same structure as discrete)
    │   │── parallel
    │   │   │── (same structure as perpendicular)
    │── training_results
    │   │── perpendicular
    │   │   │── discrete
    │   │   │   │── folders for training results folders referred to the 'Naming order' below.
    │   │   │── continuous
    │   │   │   │── (same structure as discrete)
    │   │── parallel
    │   │   │── (same structure as perpendicular)


## Folder naming rules
The folder for both trained agents and training results is created based on the following rules. 

### Element	Description	Example
| Elements      | Description                                  | Examples                | Abbreviation for folder nameing rules              |
|---------------|----------------------------------------------|-------------------------|----------------------------------------------------|
| algorithm     | The RL algorithm used in training            | PPO                     | -                                                  |
| parkingType   | The type of parking environment              | perpendicular, parallel | -                                                  |
| actionType    | The type of action space                     | discrete, continuous    | -                                                  |
| num_train     | The number of training                       | Integer (ex. 100)       | -                                                  |
| rewardType    | The reward function type                     | type1, type2, type3     | r1, r2, r3                                         |
| stateType     | The type of state representation             | type1, type2, type3     | s1, s2, s3                                         |
| side          | The side used for training                   | String (ex. b, t, l, r) | b: bottom, t: top, l:left, r:right, all: all sides |
| threshold     | The threshold value used in training         | 0.5, 1.0                | th05, th1                                          | 
| guidanceRatio | The ratio of guidance reward used            | 0.25, 0.5               | gr025, gr05                                        | 
| id            | A unique identifier for the training session | 1, 2, 3 and so on       | -  If id is 0, there is no id in folder name.      |

### Naming order
```
reward_type is type1: 
    {algorithm}_{parkingType}_{actionType}_{num_train}_{rewardType}_{stateType}_{side}_{id}
    Examples:
        PPO_perpendicular_continuous_100_r1_s1_all
        PPO_parallel_discrete_50_r1_s1_all_3

reward_type is either type2 or type3: 
    {algorithm}_{parkingType}_{actionType}_{rewardType}_{stateType}_{side}_{threshold}_{guidanceRatio}_{id}
    Examples:
        PPO_parallel_continuous_100_r2_s2_all_th05_gr05_2
```


## About parameters for training
### Parameters:
The parameters for trainings are defined in `parameters.py`.


## Training setups
1. Import Config class from `parameters.py` into `training.py`.
2. Make sure each parameter settings (`config` and `env_config`) in `training.py`.

   ```
   config = Config(car_length=4.0, car_width=2.0,
                wheel_length=0.75, wheel_width=0.35,
                parking_length=6.0, parking_width=4.0,
                max_distance=25.0, max_steps=80,
                acceleration_limit=1.0, steering_limit=PI/4, velocity_limit=10.0,
                max_angle_error=PI/12, center_threshold=0.25, penalty_ratio={'angle': 0.25, 'velocity': 0.25},
                reward_type='type1', state_type='type1',
                side=(1,2,3,4), car_loc_randomize_range=(-5, 5), initial_distance_range=(10.0, 15.0)
                )

   env_config = {"render_mode": "no_render",
                 "action_type": "continuous",
                 "parking_type": "perpendicular",
                 "training_mode": "on",
                 "config": config}
   ```

- Parameter descriptions for config.  
   Although each parameter is explained in `parameters.py`, the following parameters are often used.

   | Parameters              | Description                                                                                                                                                                                                     |
   |-------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
   | reward_type             | Reward type for trainings. There are 4 types and the fault is 'type1'.                                                                                                                                          |
   | state_type              | State type for trainings. There are 4 types and the fault is 'type1'.                                                                                                                                           |
   | penalty_ratio           | Reward weights for guidance and velocity. You need to pay attention to this if you set type2, 3, 4 in reward_type.                                                                                              |
   | car_loc_randomize_range | Range (in meters) for randomizing the car's initial position. The first and second elements represent the minimum and maximum values, respectively. The default value is (-5, 5).                               |
   | initial_distance_range  | Range (in meters) for setting the initial distance between the car and the parking lot. The first and second elements represent the minimum and maximum values, respectively. The default value is (7.5, 15.0). |
   | side                    | Defines which side of the environment the parking lot is placed. It can be a single integer(fixed side) or a tuple(randomized side selection). The default value is 1 (Bottom side).                            |


- Parameter descriptions for env_config

   | Parameters    | Description                                                                                                                                                                                                 |
   |---------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
   | render_mode   | Set either 'human' or 'no_render'. 'no_render' is recommended for training. 'human' is useful for evaluation.                                                                                               |
   | action_type   | Set either 'continuous' or 'discrete'.                                                                                                                                                                      |
   | parking_type  | Set either 'perpendicular' or 'parallel'.                                                                                                                                                                   |
   | training_mode | Set either 'on' or 'off'. If the training_mode is 'on', the parking lot location is fixed during the training. If the training_mode is 'off', the parking lot location is randomly set during the training. |
   | config        | Config class object                                                                                                                                                                                         |

   **Tips of Parameter settings for Trainings**  
   It is vital to encourage the agent to approach to the parking lot by setting both the agent and the parking lot closely.  
   It is also essential to set the heading angle for the agent to move to the parking lot easily.

   ```
   - Example for Perpendicular Parking
      config = Cofig(side=(1,2,3,4), car_loc_randomize_range=(-5, 5), 
               initial_distance_range=(7.5, 15.0)
               )  # Heading angles are defined in `parameters.py` and use it.
               
      env_config = {"render_mode": "no_render",
                     "action_type": "continuous or discrete",
                     "parking_type": "perpendicular or parallel",
                     "training_mode": "on",
                     "config": config}
      
   - Example for Parallel Parking
      **Update later**
   ```

3. Sets the number of training.  
   ![image](https://github.com/user-attachments/assets/19aa8211-67ed-4b18-820c-c31bcecf050d)

4. After settings, run `training.py` to execute a training.
   
## Evaluate the trained agent
After the training, you can see the agent behaviour in `main.py`.  
All you need to do is to follow the following steps.
1. Set the folder name that you want to evaluate.  
   ![image](https://github.com/user-attachments/assets/a998d659-0ce4-4948-878f-ec1ef9dd67f2)

2. Make sure each parameter setting in `main.py`, referring to the chapter 2 on `Training setups`. 

3. Run `main.py`.

## Display the training results  
Use the following command so that you can see data in the training_agent folder.  
```
tensorboard --logdir=folder path  
tensorboard --logdir=C:\Users\-------\----
```


