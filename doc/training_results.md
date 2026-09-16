<style>
    /* initialise the counter */
    body { counter-reset: figureCounter; }
    /* increment the counter for every instance of a figure even if it doesn't have a caption */
    figure { counter-increment: figureCounter; }
    /* prepend the counter to the figcaption content */
    figure figcaption:before {
        content: "Figure " counter(figureCounter) ": "
    }
</style>



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


- Training results (Continuous action)

<figure>
    <img src="../training/assets/perpendicular_parking/perpendicular_type1_len_mean.png" alt="The mean length plot for the type1 of the perpendicular parking" width="1000">
    <figcaption>The mean episode length of the continuous action space in the type1 of the perpendicular parking for the 100-iteration training</figcaption>
</figure>

<figure>
    <img src="../training/assets/perpendicular_parking/perpendicular_type1_reward_max.png" alt="The max reward plot for the type1 of the perpendicular parking" width="1000">
    <figcaption>The maximum reward in each iteration of the continuous action space in the type1 of the perpendicular parking for the 100-iteration training</figcaption>
</figure>

<figure>
    <img src="../training/assets/perpendicular_parking/perpendicular_type1_reward_mean.png" alt="The mean reward plot for the type1 of the perpendicular parking" width="1000">
    <figcaption>The mean reward in each episode of the continuous action space in the type1 of the perpendicular parking for the 100-iteration training</figcaption>
</figure>

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



- Training results (Continuous action)

<figure>
    <img src="../training/assets/perpendicular_parking/perpendicular_type2_len_mean.png" alt="The mean length plot for the type2 of the perpendicular parking" width="1000">
    <figcaption>The mean episode length of the continuous action space in the type2 of the perpendicular parking for the 100-iteration training</figcaption>
</figure>

<figure>
    <img src="../training/assets/perpendicular_parking/perpendicular_type2_reward_max.png" alt="The max reward plot for the type2 of the perpendicular parking" width="1000">
    <figcaption>The maximum reward in each iteration of the continuous action space in the type2 of the perpendicular parking for the 100-iteration training</figcaption>
</figure>

<figure>
    <img src="../training/assets/perpendicular_parking/perpendicular_type2_reward_mean.png" alt="The mean reward plot for the type2 of the perpendicular parking" width="1000">
    <figcaption>The mean reward in each episode of the continuous action space in the type2 of the perpendicular parking for the 100-iteration training</figcaption>
</figure>

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


- Training results (Continuous action)

<figure>
    <img src="../training/assets/perpendicular_parking/perpendicular_type3_len_mean.png" alt="The mean length plot for the type3 of the perpendicular parking" width="1000">
    <figcaption>The mean episode length of the continuous action space in the type3 of the perpendicular parking for the 200-iteration training</figcaption>
</figure>

<figure>
    <img src="../training/assets/perpendicular_parking/perpendicular_type3_reward_max.png" alt="The max reward plot for the type3 of the perpendicular parking" width="1000">
    <figcaption>The maximum reward in each iteration of the continuous action space in the type3 of the perpendicular parking for the 200-iteration training</figcaption>
</figure>

<figure>
    <img src="../training/assets/perpendicular_parking/perpendicular_type3_reward_mean.png" alt="The mean reward plot for the type3 of the perpendicular parking" width="1000">
    <figcaption>The mean reward in each episode of the continuous action space in the type3 of the perpendicular parking for the 200-iteration training</figcaption>
</figure>

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


- Training results (Continuous action)

<figure>
    <img src="../training/assets/perpendicular_parking/perpendicular_type4_len_mean.png" alt="The mean length plot for the type4 of the perpendicular parking" width="1000">
    <figcaption>The mean episode length of the continuous action space in the type4 of the perpendicular parking for the 200-iteration training</figcaption>
</figure>

<figure>
    <img src="../training/assets/perpendicular_parking/perpendicular_type4_reward_max.png" alt="The max reward plot for the type4 of the perpendicular parking" width="1000">
    <figcaption>The maximum reward in each iteration of the continuous action space in the type4 of the perpendicular parking for the 200-iteration training</figcaption>
</figure>

<figure>
    <img src="../training/assets/perpendicular_parking/perpendicular_type4_reward_mean.png" alt="The mean reward plot for the type4 of the perpendicular parking" width="1000">
    <figcaption>The mean reward in each episode of the continuous action space in the type4 of the perpendicular parking for the 200-iteration training</figcaption>
</figure>

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


- Training results (Continuous action)

<figure>
    <img src="../training/assets/parallel_parking/parallel_type1_len_mean.png" alt="The mean length plot for the type1 of the parallel parking" width="1000">
    <figcaption>The mean episode length of the continuous action space in the type1 of the parallel parking for the 100-iteration training</figcaption>
</figure>

<figure>
    <img src="../training/assets/parallel_parking/parallel_type1_reward_max.png" alt="The max reward plot for the type1 of the parallel parking" width="1000">
    <figcaption>The maximum reward in each iteration of the continuous action space in the type1 of the parallel parking for the 100-iteration training</figcaption>
</figure>

<figure>
    <img src="../training/assets/parallel_parking/parallel_type1_reward_mean.png" alt="The mean reward plot for the type1 of the parallel parking" width="1000">
    <figcaption>The mean reward in each episode of the continuous action space in the type1 of the parallel parking for the 100-iteration training</figcaption>
</figure>

- Training analysis  


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


- Training results (Continuous action)

<figure>
    <img src="../training/assets/parallel_parking/parallel_type2_len_mean.png" alt="The mean length plot for the type2 of the parallel parking" width="1000">
    <figcaption>The mean episode length of the continuous action space in the type2 of the parallel parking for the 100-iteration training</figcaption>
</figure>

<figure>
    <img src="../training/assets/parallel_parking/parallel_type2_reward_max.png" alt="The max reward plot for the type2 of the parallel parking" width="1000">
    <figcaption>The maximum reward in each iteration of the continuous action space in the type2 of the parallel parking for the 100-iteration training</figcaption>
</figure>

<figure>
    <img src="../training/assets/parallel_parking/parallel_type2_reward_mean.png" alt="The mean reward plot for the type2 of the parallel parking" width="1000">
    <figcaption>The mean reward in each episode of the continuous action space in the type2 of the parallel parking for the 100-iteration training</figcaption>
</figure>

- Training analysis  



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


- Training results (Continuous action)

<figure>
    <img src="../training/assets/parallel_parking/parallel_type3_len_mean.png" alt="The mean length plot for the type3 of the parallel parking" width="1000">
    <figcaption>The mean episode length of the continuous action space in the type1 of the parallel parking for the 100-iteration training</figcaption>
</figure>

<figure>
    <img src="../training/assets/parallel_parking/parallel_type3_reward_max.png" alt="The max reward plot for the type3 of the parallel parking" width="1000">
    <figcaption>The maximum reward in each iteration of the continuous action space in the type3 of the parallel parking for the 100-iteration training</figcaption>
</figure>

<figure>
    <img src="../training/assets/parallel_parking/parallel_type3_reward_mean.png" alt="The mean reward plot for the type3 of the parallel parking" width="1000">
    <figcaption>The mean reward in each episode of the continuous action space in the type3 of the parallel parking for the 100-iteration training</figcaption>
</figure>

- Training analysis  


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



**This is a test code for video** 

<video width="720" height="360" controls>
  <source src="../training/assets/parallel_parking/parallel_type4.mov" type="video/quicktime">
  Your browser does not support the video tag.
</video>

https://github.com/taka-rl/RL_APS/blob/feature/train_evaluate_agent/training/assets/parallel_parking/parallel_type4.mov

- Training results (Continuous action)

<figure>
    <img src="../training/assets/parallel_parking/parallel_type4_len_mean.png" alt="The mean length plot for the type4 of the parallel parking" width="1000">
    <figcaption>The mean episode length of the continuous action space in the type4 of the parallel parking for the 100-iteration training</figcaption>
</figure>

<figure>
    <img src="../training/assets/parallel_parking/parallel_type4_reward_max.png" alt="The max reward plot for the type4 of the parallel parking" width="1000">
    <figcaption>The maximum reward in each iteration of the continuous action space in the type4 of the parallel parking for the 100-iteration training</figcaption>
</figure>

<figure>
    <img src="../training/assets/parallel_parking/parallel_type4_reward_mean.png" alt="The mean reward plot for the type4 of the parallel parking" width="1000">
    <figcaption>The mean reward in each episode of the continuous action space in the type4 of the parallel parking for the 100-iteration training</figcaption>
</figure>

- Training analysis  


## Conclusion


### Future developments

