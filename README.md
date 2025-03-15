# RL_APS
This repository is a development environment for my thesis: "Reinforcement Learning-Based Automated Parking Systems." It provides the training environment for the agent in both parallel and perpendicular parking scenarios.

The following videos present the trained agent's behavior in the parking simulation.
The environment is a continuous action space in the perpendicular parking.
- without guidance
  
https://github.com/taka-rl/RL_APS/assets/157423802/d21c33ac-b5c0-4244-801e-7933abfa9792

- with guidance
  
https://github.com/taka-rl/RL_APS/assets/157423802/930700ff-9d21-4bcc-a0a7-bce26dfaa3a3


## Folder structure


    ├── sim_env                         # simulation environment
    │   ├── car.py                      # car class
    │   ├── main.py                     # visualize the trained agent
    │   ├── parameters.py               # parameter class
    │   ├── parking.py                  # parking class
    │   ├── parking_env.py              # parking environment class
    │   └── renderer.py                 # renderer class
    ├── tests                           # unit tests
    │   ├── conftest.py                 # 
    │   ├── test_car.py                 # test for Car class
    │   ├── test_parameters.py          # test for parameters
    │   ├── test_parking.py             # test for BaseParking, PerpendicularParking and ParallelParking class 
    │   ├── test_parking_env.py         # test for the parking environment 
    │   └── test_renderer.py            # test for Renderer class
    ├── training                        # training
    │   ├── parallel                    # training results for parallel parking
    │   │    ├── continuous             # continuous action
    │   │    │     ├── trained_agent    # trained agent
    │   │    │     └── training_result  # training result
    │   │    └── discrete               # discrete action
    │   │    │     ├── trained_agent    # trained agent
    │   │    │     └── training_result  # training result
    │   ├── perpendicular               # training results for perpendicular parking 
    │   │    ├── continuous             # continuous action
    │   │    │     ├── trained_agent    # trained agent
    │   │    │     └── training_result  # training result
    │   │    └── discrete               # discrete action
    │   │    │     ├── trained_agent    # trained agent
    │   │    │     └── training_result  # training result
    │   ├── training.py                 # for training
    │   └── utility.py                  # utility functions
    ├── old                              
    ├── practice_pygame                 # pygame practice
    ├── practice_rllib                  # Ray RLlib practice
    ├── requirements.txt                # Required dependencies
    └── README.md                       # Project documentation


## Simulation Environment
### used tools
The libraries and their versions are as follows.

| tool | version |
| ---- | ----|
| Python | 3.10.11 |
| Gymnasium | 0.28.1 | 
| Ray RLlib | 2.9.0 |
| Numpy | 1.26.3 |
| Pygame | 2.1.3 |

### environment description
This environment supports both parallel and perpendicular parking, using either discrete or continuous action spaces.  
As illustrated in the figure below, the custom environment can render a 2D environment with a top-down view and simulate parking movement using front-wheel steering through the Kinematic bicycle model. 
At each step, the car is rendered based on the input actions, which include acceleration and steering angle, and the next state of the vehicle is simulated.  
The yellow rectangles represent the parking lot for two obstacles depicted as grey rectangles. The red rectangle indicates the parking lot for the agent, shown as the green rectangle.
The grey grid line is drawn at 1-meter intervals.
Although the visualization window size is 800 by 600 pixels, this corresponds to a physical size of 40 meters by 30 meters, defining 1 pixel as 0.05 meters and units are unified as meters. The parking lot size is 6 meters in length and 4 meters in width, while the car size for the agent is 4 meters in length and 2 meters in width. Furthermore, it can display some information such as the center of the car, the velocity, the heading angle and its radian. This is useful for debugging to make sure the car’s movement.

・Perpendicular parking image  
![image](https://github.com/taka-rl/RL_APS/assets/157423802/ae308d2d-1c6a-4a6c-a707-f5dae83b23db)

・Parallel parking image  
![image](https://github.com/taka-rl/RL_APS/assets/157423802/34acb574-9777-4ce0-91b5-6789214f6dda)

## Kinematic Bicycle model
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
  
## Reinforcement learning
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
This section describes the state value designed in this project.  
The coordinate of the parking space corner points, which means the transformed coordinate system from the global coordinate system to the local coordinate system.
In global coordinates, the car must account for its own position and orientation within the global frame, complicating calculations. Expressing a global coordinate system as a local coordinate system simplifies the representation, making it easier to manage and understand. The following figure illustrates the local coordinate system where the front side of the car is positive along the y-axis, and the right side of the car is positive along the x-axis. The distance between each parking lot vertex and the center of the car is transformed into the car’s relative coordinate system.

![image](https://github.com/taka-rl/RL_APS/assets/157423802/dc39da5f-558b-4350-ae86-c97411c1dfbf)

The distance is divided by the maximum distance for normalization and the Maximum distance is 25m.

### Reward type
The following reward functions are designed in this project. The reward is given at the end of each episode. Therefore, the current episode is terminated when one of these events happens.  

| Reward type       | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
|-------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Successful reward | When the agent reverses into the parking lot, then the agent obtains a +1 reward.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| Collision check   | When the agent enters the grey rectangle, the agent obtains a -1 reward.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | 
| Maximum step      | When the agent takes more than the set maximum step, it obtains a -1 reward.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| Line cross-check  | When the agent crosses the parking lot border vertically or horizontally, it obtains a -1 reward. This check is implemented to ensure realistic behavior, preventing the car from crossing to the opposite side of the parking lot to park. It is dependent on the placement of the parking lot. If the agent crosses the bottom border, it indicates a horizontal border crossing. Another example is if the parking lot is placed on the right side of the visualization window; when the agent crosses the right border of the parking lot, it indicates a vertical border crossing. |
| Maximum distance  | When the agent is farther away from the parking lot than the set maximum distance, it obtains a -1 reward.                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
  

#### guidance point
The agent received a +1 reward when it parked in the parking lot. However, the reward system was modified with the introduction of the guidance point.  
If the car’s vertices are within the parking lot and the center of the car is within a set distance threshold from the center of the parking lot, the agent receives a +1 reward. A small value is then subtracted from this reward based on the angle error to ensure the car is parallel with the parking lot borders. The penalty for angle error is linearly related, where an angle error of 0 degrees results in no penalty, and larger errors reduce the reward accordingly, up to a maximum penalty of 0.5.  
The reason for this is to encourage the agent to park near the center of the parking lot and to be parallel to the borders.

## How to use
### install tools
The first is to install necessary libraries.  

ray rllib: `pip install "ray[rllib]" tensorflow`  
Gymnasium: `pip install "gymnasium[all]"`

### Settings for training
1. Set parameters for training in the Config object in `training.py`
   You can modify the maximum velocity, steps, acceleration, steering angle, car size, parking lot size, reward and state types and so on related to the simulation.
   The Config object is defined in `parameters.py`.
   ![image](https://github.com/user-attachments/assets/9f9eb8e3-5e1d-4183-8608-7100339cd6ec)  

2. Choose the parking type and action space type.
   It is recommended to set "no_render" as "render_mode" for the training in terms of efficiency and speed.  
   ![image](https://github.com/user-attachments/assets/d72b6a73-551b-4157-b046-9fdbe85fe309)


3. Set the number of iterations for the training at line 31, num_train = "the number of iterations".  
   ![image](https://github.com/user-attachments/assets/51581c24-2955-454f-a923-177ff5cce1b8)

4. After these settings, you can execute the training.py script. After the training, the result folder and the agent folder are saved in the training folder.

- folder structure for the training
  

    ├── training                         # training
         └── parking type                # parallel/perpendiuclar parking
               └── action space          # continuous/discrete action
                    ├── trained_agent    # the policy data is stored.
                    └── training_result  # the training result is stored.


## Visualize the agent
After the training, you can observe the trained agent's behaviour using main.py script.  
Line 12, set the folder name in the trained_agent folder saved after the training.  
Do not forget to set "human" as "render_mode" at line 6.
![image](https://github.com/taka-rl/RL_APS/assets/157423802/20be06ef-1e91-43f5-9275-8d805f5ac903)

Choose how the action is set among the following three options.
If you use the trained agent, Line 22 should be executed and others should be commented out.  
![image](https://github.com/taka-rl/RL_APS/assets/157423802/76e5b056-7324-49bf-978f-0b7237aa2674)  
Line 22: action is chosen by the trained agent.  
Line 23: The random action is taken.  
Line 24: The action is set manually.  

## Evaluation
Use the following command so that you can see data in the training_agent folder.  
```
tensorboard --logdir=folder path  
tensorboard --logdir=C:\Users\-------\----
```

For example:  
![image](https://github.com/taka-rl/RL_APS/assets/157423802/3a254879-2176-43c9-8a95-a777b348c38c)  
![image](https://github.com/taka-rl/RL_APS/assets/157423802/a51cb69d-472a-4370-942f-ccca2fecb4c4)  
![image](https://github.com/taka-rl/RL_APS/assets/157423802/d4e2cec0-def9-483c-b44f-dc4cf9b52397)  


