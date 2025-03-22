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
2. Make sure each parameter setting.  
   ![image](https://github.com/user-attachments/assets/e88a2af1-1d96-4a92-9a04-6d1d30d91430)

3. Also sets env_config.  
   ![image](https://github.com/user-attachments/assets/a03c5891-d558-4cac-a3aa-b2e3ff308d61)

4. Sets the number of training.  
   ![image](https://github.com/user-attachments/assets/19aa8211-67ed-4b18-820c-c31bcecf050d)

5. After settings, run `training.py` to execute a training.
   
## Evaluate the training result
After the training, you can see the agent behaviour in `main.py`.  
All you need to do is to follow the following steps.
1. Set the folder name that you want to evaluate.  
   ![image](https://github.com/user-attachments/assets/a998d659-0ce4-4948-878f-ec1ef9dd67f2)

2. Run `main.py`.

