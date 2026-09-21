# Training Results
This document shows the training results of both perpendicular parking and parallel parking.

## Perpendicular Parking

### Default Setting (Reward/State: type1)

- Parameter settings for the training

```
config = Config(car_length=4.0, car_width=2.0,
                wheel_length=0.75, wheel_width=0.35,
                parking_length=6.0, parking_width=4.0,
                max_distance=25.0, max_steps=80,
                acceleration_limit=1.0, steering_limit=PI/4, velocity_limit=2.77778,
                max_angle_error=PI/12, center_threshold=0.5, penalty_ratio={'angle': 0.25, 'velocity': 0.25},
                reward_type='type1', state_type='type1',
                side=(1, 2, 3, 4), car_loc_randomize_range=(-2.5, 2.5), initial_distance_range=(5.0, 7.5)
                )
                
env_config = {"render_mode": "no_render",
              "action_type": "continuous",  # or "discrete"
              "parking_type": "perpendicular",
              "training_mode": "on",
              'config': config}

num_train = 100
```

- Trained agent (Continuous action)

https://github.com/user-attachments/assets/3037c1a2-86c6-4039-9258-bbe9ae767bd4

- Training results (Continuous action)
![Figure 1: The mean length plot for the type1 of the perpendicular parking](../training/assets/perpendicular_parking/perpendicular_type1_len_mean.png)
*Figure 1: The mean episode length of the continuous action space in the type1 of the perpendicular parking for the 100-iteration training*
![Figure 2: The max reward plot for the type1 of the perpendicular parking](../training/assets/perpendicular_parking/perpendicular_type1_reward_max.png)
*Figure 2: The maximum reward in each iteration of the continuous action space in the type1 of the perpendicular parking for the 100-iteration training*
![Figure 3: The mean reward plot for the type1 of the perpendicular parking](../training/assets/perpendicular_parking/perpendicular_type1_reward_mean.png)
*Figure 3: The mean reward in each episode of the continuous action space in the type1 of the perpendicular parking for the 100-iteration training*


- Result analysis  
Figure 2 shows that the agent achieved the maximum reward early in training, indicating that it was able to successfully complete the parking task at an early stage. Moreover, as shown in Figure 1, the mean episode length gradually decreased as training progressed. 
This trend indicates that the agent learned a more efficient parking policy throughout training, enabling it to reverse into the parking slot properly.
In addition, the mean reward shown in Figure 3 increased sharply from approximately -1 to 0.7 between 50K and 150K episodes, and eventually exceeded 0.9 by the end of training.


### Guidance Reward (Reward/State: type2)

- Parameter settings for the training

```
config = Config(car_length=4.0, car_width=2.0,
                wheel_length=0.75, wheel_width=0.35,
                parking_length=6.0, parking_width=4.0,
                max_distance=25.0, max_steps=100,
                acceleration_limit=1.0, steering_limit=PI/4, velocity_limit=2.77778,
                max_angle_error=PI/4.5, center_threshold=0.5, penalty_ratio={'angle': 0.5, 'velocity': 0.0},
                reward_type='type2', state_type='type2',
                side=1, car_loc_randomize_range=(-2.5, 2.5), initial_distance_range=(5.5, 7.5)
                )
                
env_config = {"render_mode": "no_render",
              "action_type": "continuous",  # or "discrete"
              "parking_type": "perpendicular",
              "training_mode": "on",
              'config': config}

num_train = 100
```

- Trained agent (Continuous action)

https://github.com/user-attachments/assets/7ef0b4ae-326c-462b-8ce0-f7ffcbbf6a26


- Training results (Continuous action)
![Figure 4: The mean length plot for the type2 of the perpendicular parking](../training/assets/perpendicular_parking/perpendicular_type2_len_mean.png)
*Figure 4: The mean episode length of the continuous action space in the type2 of the perpendicular parking for the 100-iteration training*
![Figure 5: The max reward plot for the type2 of the perpendicular parking](../training/assets/perpendicular_parking/perpendicular_type2_reward_max.png)
*Figure 5: The maximum reward in each iteration of the continuous action space in the type2 of the perpendicular parking for the 100-iteration training*
![Figure 6: The mean reward plot for the type2 of the perpendicular parking](../training/assets/perpendicular_parking/perpendicular_type2_reward_mean.png)
*Figure 6: The mean reward in each episode of the continuous action space in the type2 of the perpendicular parking for the 100-iteration training*


- Result analysis  
Figure 5 shows that the agent successfully completed the parking task, reaching more than 0.98 reward at 50K episodes. 
In addition, Figure 4 illustrates the mean episode length decreased significantly during the first 150K episodes, indicating that the agent was able to improve its policy. 
At the same time, the mean reward reached 0.7 at 150 episodes and continued to increase, eventually reaching approximately 0.95 by the end of training. 
These results suggest that the angle penalty successfully encouraged the agent to reduce its heading error.

- Training history  
At the beginning of training, the agent failed to learn the importance of reducing its heading error despite the use of an angle penalty.
This occurred because the agent's heading angle often exceeded `max_angle_error` by the time each episode finished.  
The `max_angle_error` parameter defines the heading-error range over which the angle penalty increases proportionally. If the heading error exceeds `max_angle_error`, the maximum angle penalty is applied regardless of how much larger the error becomes.  
Initially, `max_angle_error` value was set to PI/12 (15 degrees) during training. However, this threshold did not work effectively because the agent could not differentiate between a heading error of 15 degrees and an error greater than 15 degrees, as both resulted in the same maximum penalty. This lack of differentiation prevented the agent from learning to reduce larger heading errors.  
Increasing `max_angle_error` to PI/4.5 (40 degrees) allowed the agent to distinguish  between a wider range of heading errors, enabling it to reduce its heading error relative to the parking space.


### Velocity Penalty (Reward/State: type3)
- Parameter settings for the training

```
config = Config(car_length=4.0, car_width=2.0,
                wheel_length=0.75, wheel_width=0.35,
                parking_length=6.0, parking_width=4.0,
                max_distance=25.0, max_steps=80,
                acceleration_limit=1.0, steering_limit=PI/4, velocity_limit=2.77778,
                max_angle_error=PI/12, center_threshold=0.5, penalty_ratio={'angle': 0.0, 'velocity': 0.5},
                reward_type='type3', state_type='type3',
                side=1, car_loc_randomize_range=(-2.5, 2.5), initial_distance_range=(5.5, 7.5)
                )
                
env_config = {"render_mode": "no_render",
              "action_type": "continuous",  # or "discrete"
              "parking_type": "perpendicular",
              "training_mode": "on",
              'config': config}

num_train = 200
```

- Trained agent (Continuous action)

https://github.com/user-attachments/assets/95bb08ec-4963-47ad-ae02-76217edc01cf

- Training results (Continuous action)
![Figure 7: The mean length plot for the type3 of the perpendicular parking](../training/assets/perpendicular_parking/perpendicular_type3_len_mean.png)
*Figure 7: The mean episode length of the continuous action space in the type3 of the perpendicular parking for the 200-iteration training*
![Figure 8: The max reward plot for the type3 of the perpendicular parking](../training/assets/perpendicular_parking/perpendicular_type3_reward_max.png)
*Figure 8: The maximum reward in each iteration of the continuous action space in the type3 of the perpendicular parking for the 200-iteration training*
![Figure 9: The mean reward plot for the type3 of the perpendicular parking](../training/assets/perpendicular_parking/perpendicular_type3_reward_mean.png)
*Figure 9: The mean reward in each episode of the continuous action space in the type3 of the perpendicular parking for the 200-iteration training*


- Training analysis  
Figure 8 illustrates that the agent was able to complete the parking task early in training. 
Additionally, the mean episode length shown in Figure 7 decreased substantially to approximately 50 by around 150K episodes, reaching about 46 by the end of training. 
This trend suggests that the agent learnt a more effective parking policy. The mean reward shown in Figure 9 increased sharply, exceeding 0.5 at 150K episodes, and then slowly increased to approximately 0.9 by the end of training. These results indicate that the velocity penalty successfully encouraged the agent to reduce its velocity while approaching the parking space.


### Both Guidance Reward and Velocity Penalty (Reward/State: type4)
- Parameter settings for the training

```
config = Config(car_length=4.0, car_width=2.0,
                wheel_length=0.75, wheel_width=0.35,
                parking_length=6.0, parking_width=4.0,
                max_distance=25.0, max_steps=80,
                acceleration_limit=1.0, steering_limit=PI/4, velocity_limit=2.77778,
                max_angle_error=PI/4.5, center_threshold=0.5, penalty_ratio={'angle': 0.25, 'velocity': 0.25},
                reward_type='type4', state_type='type4',
                side=1, car_loc_randomize_range=(-2.5, 2.5), initial_distance_range=(5.5, 7.5)
                )
                
env_config = {"render_mode": "no_render",
              "action_type": "continuous",  # or "discrete"
              "parking_type": "perpendicular",
              "training_mode": "on",
              'config': config}

num_train = 200
```

- Trained agent (Continuous action)

https://github.com/user-attachments/assets/aef287df-88b4-4672-b855-cbd782413d07

- Training results (Continuous action)  
![Figure 10: The mean length plot for the type4 of the perpendicular parking](../training/assets/perpendicular_parking/perpendicular_type4_len_mean.png)
*Figure 10: The mean episode length of the continuous action space in the type4 of the perpendicular parking for the 200-iteration training*
![Figure 11: The max reward plot for the type4 of the perpendicular parking](../training/assets/perpendicular_parking/perpendicular_type4_reward_max.png)
*Figure 11: The maximum reward in each iteration of the continuous action space in the type4 of the perpendicular parking for the 200-iteration training*
![Figure 12: The mean reward plot for the type4 of the perpendicular parking](../training/assets/perpendicular_parking/perpendicular_type4_reward_mean.png)
*Figure 12: The mean reward in each episode of the continuous action space in the type4 of the perpendicular parking for the 200-iteration training*


- Training analysis  
Figure 11 illustrates that the agent was able to complete the parking task early in training, with the maximum reward eventually exceeding 0.95. The mean episode length shown in Figure 10 plunged to approximately 48 at 100K episodes, stabilizing around 45 by the end of training. Moreover, Figure 12 shows that the agent achieved a mean reward of more than 0.5 at 100K, and the mean reward gradually increased to 0.89 by the end of training. These results indicate that the agent learned a more effective policy under a reward function incorporating both the angle and velocity penalties. 


## Parallel Parking
### Default Setting (Reward/State: type1)

- Parameter settings for the training

```
config = Config(car_length=4.0, car_width=2.0,
                wheel_length=0.75, wheel_width=0.35,
                parking_length=6.0, parking_width=4.0,
                max_distance=25.0, max_steps=80,
                acceleration_limit=1.0, steering_limit=PI/4, velocity_limit=2.77778,
                max_angle_error=PI/12, center_threshold=0.5, penalty_ratio={'angle': 0, 'velocity': 0.5},
                reward_type='type1', state_type='type1',
                side=1, car_loc_randomize_range=(5.5, 6.0), initial_distance_range=(5.0, 6.0),
                heading_angle_range={"parallel": {1: (PI/6, PI/4)}}
                )
                
env_config = {"render_mode": "no_render",
              "action_type": "continuous",
              "parking_type": "parallel",
              "training_mode": "on",
              'config': config}

num_train = 100

```

- Trained agent (Continuous action)

https://github.com/user-attachments/assets/42b0bc5d-c2a8-4428-bd06-a62967ce3cf1

- Training results (Continuous action)
![Figure 13: The mean length plot for the type1 of the parallel parking](../training/assets/parallel_parking/parallel_type1_len_mean.png)
*Figure 13: The mean episode length of the continuous action space in the type1 of the parallel parking for the 100-iteration training*
![Figure 14: The max reward plot for the type1 of the parallel parking](../training/assets/parallel_parking/parallel_type1_reward_max.png)
*Figure 14: The maximum reward in each iteration of the continuous action space in the type1 of the parallel parking for the 100-iteration training*
![Figure 15: The mean reward plot for the type1 of the parallel parking](../training/assets/parallel_parking/parallel_type1_reward_mean.png)
*Figure 15: The mean reward in each episode of the continuous action space in the type1 of the parallel parking for the 100-iteration training*


- Training analysis  
Figure 14 depicts the maximum reward trend, showing that the agent successfully completed the parking task early in training, reaching the maximum reward at 60K episodes. 
In addition, the mean reward shown in Figure 15 surged from approximately -0.75 to nearly 1.0 between 50K and 100K episodes, indicating that the agent learned a more effective parking policy.
Moreover, the mean episode length in Figure 13 decreased significantly to around 45 by 100K episodes and eventually stabilized around 38.


### Guidance Reward (Reward/State: type2)

- Parameter settings for the training

```
config = Config(car_length=4.0, car_width=2.0,
                wheel_length=0.75, wheel_width=0.35,
                parking_length=6.0, parking_width=4.0,
                max_distance=25.0, max_steps=80,
                acceleration_limit=1.0, steering_limit=PI/4, velocity_limit=2.77778,
                max_angle_error=PI/6, center_threshold=0.5, penalty_ratio={'angle': 0.5, 'velocity': 0.0},
                reward_type='type2', state_type='type2',
                side=1, car_loc_randomize_range=(5.5, 6.0), initial_distance_range=(4.0, 5.0),
                heading_angle_range={"parallel": {1: (PI/6, PI/4)}}
                )

env_config = {"render_mode": "no_render",
              "action_type": "continuous",
              "parking_type": "parallel",
              "training_mode": "on",
              'config': config}

num_train = 100

```

- Trained agent (Continuous action)

https://github.com/user-attachments/assets/890e23aa-7657-4e0b-ac80-7550117164ab

- Training results (Continuous action)
![Figure 16: The mean length plot for the type2 of the parallel parking](../training/assets/parallel_parking/parallel_type2_len_mean.png)
*Figure 16: The mean episode length of the continuous action space in the type2 of the parallel parking for the 100-iteration training*
![Figure 17: The max reward plot for the type2 of the parallel parking](../training/assets/parallel_parking/parallel_type2_reward_max.png)
*Figure 17: The maximum reward in each iteration of the continuous action space in the type2 of the parallel parking for the 100-iteration training*
![Figure 18: The mean reward plot for the type2 of the parallel parking](../training/assets/parallel_parking/parallel_type2_reward_mean.png)
*Figure 18: The mean reward in each episode of the continuous action space in the type2 of the parallel parking for the 100-iteration training*


- Training analysis  
Although the mean episode length shown in Figure 16 remained nearly unchanged until 100K episodes, it decreased sharply to around 52 at 150K episodes, and continued to decline gradually, reaching approximately 41 by the end of training.
Similarly, the mean reward illustrated in Figure 18 was 0.0 until 75K episodes, then exceeded 0.5 at 150K episodes, eventually reached approximately 0.8.
Figure 17 shows that the maximum reward was around 0 at 30K episodes before decreasing to approximately -1.0 by 75K episodes. Then, it increased dramatically from -1.0 to nearly 0.6 between 75K and 100K and continued to increase gradually, eventually reaching approximately 0.92 by the end of training. These results suggest that the angle penalty successfully encouraged the agent to reduce its heading error, as observed in perpendicular parking type2.


- Training history  
Although the agent successfully completed the parking task, the agent failed to sufficiently reduce its heading error. The reason was similar to the issue observed in perpendicular parking type2.  
When training was executed with `max_angle_error` set to PI/12, which was exactly the same phenomenon in perpendicular parking type2, the mean reward was around 0.5.  
Increasing `max_angle_error` to PI/6 (30 degrees) enabled the agent to align its heading parallel to the parking space during training. However, the agent did not approach the center of the parking slot because `center_threshold` was set to 1.0 (meter). Even though the mean reward increased from 0.5 to nearly 1.0 with the configuration, it was not a typical parking scenario, in which a vehicle is usually positioned closer to the center of the parking space.  
Therefore, training was conducted with `center_threshold` set to 0.5 and `max_angle_error`  kept at PI/6. Under these conditions, the agent appropriately reversed into the center of the parking space, considering its heading parallel to the parking slot.
However, a possible next step would be to revise the termination logic so that the agent is allowed several additional steps to further refine its position and heading after entering the parking space.


### Velocity Penalty (Reward/State: type3)
- Parameter settings for the training

```
config = Config(car_length=4.0, car_width=2.0,
                wheel_length=0.75, wheel_width=0.35,
                parking_length=6.0, parking_width=4.0,
                max_distance=25.0, max_steps=80,
                acceleration_limit=1.0, steering_limit=PI/4, velocity_limit=2.77778,
                max_angle_error=PI/12, center_threshold=1.0, penalty_ratio={'angle': 0.0, 'velocity': 0.5},
                reward_type='type3', state_type='type3',
                side=1, car_loc_randomize_range=(5.5, 6.0), initial_distance_range=(4.0, 5.0),
                heading_angle_range={"parallel": {1: (PI/6, PI/4)}}
                )

env_config = {"render_mode": "no_render",
              "action_type": "continuous",
              "parking_type": "parallel",
              "training_mode": "on",
              'config': config}

num_train = 100

```

- Trained agent (Continuous action)  

https://github.com/user-attachments/assets/4f534256-7ab9-4c98-ad53-5f2ef09b3f31

- Training results (Continuous action)
![Figure 19: The mean length plot for the type3 of the parallel parking](../training/assets/parallel_parking/parallel_type3_len_mean.png)
*Figure 19: The mean episode length of the continuous action space in the type3 of the parallel parking for the 100-iteration training*
![Figure 20: The max reward plot for the type3 of the parallel parking](../training/assets/parallel_parking/parallel_type3_reward_max.png)
*Figure 20: The maximum reward in each iteration of the continuous action space in the type3 of the parallel parking for the 100-iteration training*
![Figure 21: The mean reward plot for the type3 of the parallel parking](../training/assets/parallel_parking/parallel_type3_reward_mean.png)
*Figure 21: The mean reward in each episode of the continuous action space in the type3 of the parallel parking for the 100-iteration training*


- Training analysis  
The mean reward shown in Figure 21 increased significantly to approximately 0.5 by 80K while Figure 19 depicts the mean episode length, illustrating it decreased dramatically to around 52 during the same period. 
Although the mean episode length bounced back to nearly 60 at around 110K episodes, it continued to decline, eventually reaching approximately 50 by the end of training. Meanwhile, the mean reward continued to increase and eventually exceeded 0.9.
Figure 20 shows that the max reward reached nearly 0.7 at 50K episodes. After the dramatic rise from approximately 0.7 to 0.9 between 50K and 100K episodes, the maximum reward remained relatively stable and eventually approached closely 1.0.


### Both Guidance Reward and Velocity Penalty (Reward/State: type4)
- Parameter settings for the training

```
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

env_config = {"render_mode": "no_render",
              "action_type": "continuous",
              "parking_type": "parallel",
              "training_mode": "on",
              'config': config}

num_train = 100

```

- Trained agent (Continuous action)

https://github.com/user-attachments/assets/95793e46-3394-4f5d-9e24-232f20a247fb

- Training results (Continuous action)
![Figure 22: The mean length plot for the type4 of the parallel parking](../training/assets/parallel_parking/parallel_type4_len_mean.png)
*Figure 22: The mean episode length of the continuous action space in the type4 of the parallel parking for the 100-iteration training*
![Figure 23: The max reward plot for the type4 of the parallel parking](../training/assets/parallel_parking/parallel_type4_reward_max.png)
*Figure 23: The maximum reward in each iteration of the continuous action space in the type4 of the parallel parking for the 100-iteration training*
![Figure 24: The mean reward plot for the type4 of the parallel parking](../training/assets/parallel_parking/parallel_type4_reward_mean.png)
*Figure 24: The mean reward in each episode of the continuous action space in the type4 of the parallel parking for the 100-iteration training*

- Training analysis  
Figure 22 shows that the mean episode length dropped from approximately 80 to around 53 between 40K and 100K episodes, with the mean reward surging to nearly 0.5 over the same period as shown in Figure 24.
After 100K episodes, the mean episode length continued to fall to approximately 47, while the mean reward rose gradually to around 0.8.
Additionally, as shown in Figure 23, the agent achieved the max reward of 0.5 at 50K episodes. The maximum reward then continued to increase throughout training, reaching approximately 0.9. These results indicate that the agent learned a more effective policy under a reward function incorporating both the angle and velocity penalties as observed in perpendicular parking type4. 


- Training history
Under the current parameter setting, the agent successfully reversed toward the center of the parking space and completed the parking task. However, implementing the adjustment logic in the Parallel Parking Type2 would allow the agent to adjust its position and heading alongside the parking space, potentially enhancing both its performance and reward.  
The agent was also able to approach the center of the parking space when `parking_length` was set to 7.0. With this setting, roughly 0.5 meters of additional space was available in front of the vehicle compared with `parking_length` set to 6.0, making it easier for the agent to reverse towards the parking slot from the front.  
However, this configuration is not a real parallel parking scenario. Therefore, setting `parking_length` to 6.0 is more appropriate for training.


## Conclusion
These training results above describes that the agent successfully learned more effective parking policies across all reward and state configurations. Through exploration during training, the agent was able to improve its action while reducing the penalties. 


### Future developments
The next challenge will be to implement one additional feature.
The new logic should allow the agent several additional steps to adjust its position, velocity and heading within the parking space as such adjustments are common especially in parallel parking. 
