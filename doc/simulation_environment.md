# Simulation Environment
This environment supports both parallel and perpendicular parking, using either discrete or continuous action spaces.  
As illustrated in the figure below, the custom environment can render a 2D environment with a top-down view and simulate parking movement using front-wheel steering through the Kinematic bicycle model. 
At each step, the car is rendered based on the input actions, which include acceleration and steering angle, and the next state of the vehicle is simulated.  

## Environment Description
The yellow rectangles represent the parking lot for two obstacles depicted as grey rectangles. 
The red rectangle indicates the parking lot for the agent, shown as the green rectangle.
The grey grid line is drawn at 1-meter intervals.
Although the visualization window size is 800 by 600 pixels, this corresponds to a physical size of 40 meters by 30 meters, defining 1 pixel as 0.05 meters and units are unified as meters. 
The parking lot size is 6 meters in length and 4 meters in width, while the car size for the agent is 4 meters in length and 2 meters in width. 
Furthermore, it can display some information such as the center of the car, the velocity, the heading angle and its radian. 
This is useful for debugging to make sure the car’s movement.

・Perpendicular parking image  
![image](https://github.com/taka-rl/RL_APS/assets/157423802/ae308d2d-1c6a-4a6c-a707-f5dae83b23db)

・Parallel parking image  
![image](https://github.com/taka-rl/RL_APS/assets/157423802/34acb574-9777-4ce0-91b5-6789214f6dda)

## Kinematic Bicycle Model
- Kinematic bicycle model equation  
  In order to simulate the car’s movement, the kinematic bicycle model was used in the environment. (x, y) are the coordinates of the center of the car. The car’s velocity is controlled within 10 km/m since the car is usually at a low speed during parking.
  This means if the velocity becomes over 10 km/m, it is clipped as 10 km/m. The following equation is the Kinematic bicycle model used in the simulation.
   
        x_dot = v * cos(ψ)
        y_dot = v * sin(ψ)
        v_dot = a
        ψ_dot = v * tan(delta) / CAR_L
  v: velocity, ψ(psi): heading angle of the car, a: acceleration, δ(delta): steering angle, CAR_L: car length  
  x: the center of the car position in x axle, y: the center of the car position in y axle
  
  The steering angle is limited between −𝜋/4 and 𝜋/4 to make sure the car's actions are realistic and to prevent the agent's exploration range from being too large.
  Therefore, in a given time, according to the equations above, the coordinate of the center of the car (𝑥′, 𝑦′) after moving can be obtained by the following equations.

      x_dash = dt ∗ x_dot + x  
      y_dash = dt ∗ y_dot + y  
      v = v_dot + v  
      ψ = dt ∗ ψ_dot  
  
## Reinforcement Learning
### Algorithm
Proximal Policy Optimization (PPO) was used for training the reinforcement learning agent. Experiment tests for PPO conducted by OpenAI prove that PPO outperforms other online policy gradient methods while keeping a favorable balance between sample complexity, simplicity, and wall-time. PPO is an actor-critic algorithm which takes a hybrid approach combining a value-based and policy-based approach, where the actor decides actions to take, and the critic evaluates the actions taken by the actor.   

### Environment
The overview of RL model for this project is illustrated in the following figure.  
![image](https://github.com/taka-rl/RL_APS/assets/157423802/192dc6cb-3ee7-4fec-8e92-db4cdd4a516c)

### Action type
There are two action types which are continuous and discrete.  
Actions as input values for the agent are [a, δ] in both type.  
a is acceleration, limited between -1 and 1 m/s^2  as the maximum value.
δ is set to between -𝜋/4〜𝜋/4 as the maximum value.
- continuous
  The agent can choose between -1 and 1 for both acceleration and steering angle, multiplying them by the maximum limit values.  

- discrete
  There are 6 different actions. 

  | number | action values[𝑎, δ] | description         |
  |--------|----------------------|---------------------|
  | 0      | [1, 0]               | move forward        |
  | 1      | [1, -𝜋/6]           | move right forward  |
  | 2      | [1, 𝜋/6]            | move left forward   |
  | 3      | [-1, 0]              | move backward       |
  | 4      | [-1, -𝜋/6]          | move right backward |
  | 5      | [-1, 𝜋/6]           | move left backward  |
 
### State value for the agent
This section describes the state values designed in this project.  

| State type                                        | Description                                                                                                                                                                                                                                |
|---------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| The coordinate of the parking space corner points | The transformed coordinate system from the global coordinate system to the local coordinate system and divided by the maximum distance defined as 'max_distance' in Config class in `parameters.py` for normalization.                     |
| Guidance point                                    | The center point of the parking lot in the coordinate system of the car, transformed into the car's relative coordinate system and by the maximum distance defined as 'max_distance' in Config class in `parameters.py` for normalization. | 
| Velocity                                          | The car's velocity value and is divided by the velocity limit defined as 'velocity_limit' in Config class in `parameters.py` for normalization.                                                                                            |

#### Description about relative coordinate system
In global coordinates, the car must account for its own position and orientation within the global frame, complicating calculations. 
Expressing a global coordinate system as a local coordinate system simplifies the representation, making it easier to manage and understand. 
The following figure illustrates the local coordinate system where the front side of the car is positive along the y-axis, and the right side of the car is positive along the x-axis. 
The distance between each parking lot vertex and the center of the car is transformed into the car’s relative coordinate system.

![image](https://github.com/taka-rl/RL_APS/assets/157423802/dc39da5f-558b-4350-ae86-c97411c1dfbf)


### Reward type
The following reward functions are designed in this project.
The reward is given at the end of each episode. Therefore, the current episode is terminated when one of these events happens.  
There are 4 types as reward types. Common is applied to all the types. 


| Reward name       | Type    | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
|-------------------|---------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Successful reward | Common  | When the agent reverses into the parking lot, then the agent obtains a +1 reward.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| Collision check   | Common  | When the agent enters the grey rectangle, the agent obtains a -1 reward.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | 
| Maximum step      | Common  | When the agent takes more than the set maximum step, it obtains a -1 reward.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| Line cross-check  | Common  | When the agent crosses the parking lot border vertically or horizontally, it obtains a -1 reward. This check is implemented to ensure realistic behavior, preventing the car from crossing to the opposite side of the parking lot to park. It is dependent on the placement of the parking lot. If the agent crosses the bottom border, it indicates a horizontal border crossing. Another example is if the parking lot is placed on the right side of the visualization window; when the agent crosses the right border of the parking lot, it indicates a vertical border crossing.                                  |
| Maximum distance  | Common  | When the agent is farther away from the parking lot than the set maximum distance, it obtains a -1 reward.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| Guidance reward   | type2/4 | When the car’s vertices are within the parking lot and the center of the car is within a set distance threshold from the center of the parking lot, the agent receives a +1 reward. A small value is then subtracted from this reward based on the angle error to ensure the car is parallel with the parking lot borders. The penalty for angle error is linearly related, where an angle error of 0 degrees results in no penalty, and larger errors reduce the reward accordingly. The purpose of this reward is to encourage the agent to park near the center of the parking lot and to be parallel to the borders. |
| Velocity penalty  | type3/4 | This velocity penalty is linearly related. If velocity is 0km/h, it is no penalty. Larger velocity value reduce the reward accordingly. The purpose of this penalty is to encourage the agent to approach the parking lot gently.                                                                                                                                                                                                                                                                                                                                                                                        |

