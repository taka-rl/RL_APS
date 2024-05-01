# RL_APS
This repository is a development environment for my thesis which is "Reinforcement learning based automated parking systems". It provides the training envrionment for the agent in both parallel and perpendicular parkings.

# Folder structure
- sim_env: contains the necessary scripts for the parking simulation
  - parking_env.py: environment class
  - car.py: car class
  - com_fcn.py: common functions
  - parameters.py: parameters for the simulation environment
  - init_state.py: initialize the car/parking lot location and heading angle for the training
- training: contains the script for the training
  - training.py: for the training
  - utility.py: useful functions for the training
- practice_rllib: to learn how to use Ray RLlib for my thesis
- practice_pygame: to learn how to draw the parking environment for my thesis

# Simulation Environment
- Python
- Gymnasium
- Ray RLlib

# Kinematic Bicycle model
- Kinematic bicycle model equation
 
        x_dot = v * np.cos(ψ)
        y_dot = v * np.sin(ψ)
        v_dot = a
        psi_dot = v * np.tan(delta) / CAR_L
  v: velocity, ψ(psi): heading angle of the car, a: acceleration, δ(delta): steering angle, CAR_L: car length  
  x: the center of the car position in x axle, y: the center of the car position in y axle  

# Reinforcement learning
## Environment
This environment provides Parallel and Perpendicular parking.  

## Action type
There are two action types which are continuous and discrete.  
Actions as input values for the agent are [a, δ] in both type.  
- continuous  
  a is acceleration, limited between -1 and 1 m/s^2
  δ is set between -pi/4〜pi/4  
- discrete
  Action space is 6.  
  0: [1, 0] # move forward  
  1: [1, -np.pi/6] # move right forward  
  2: [1, np.pi/6] # move left forward  
  3 :[-1, 0] # move backward  
  4 :[-1, -np.pi/6] # move right backward  
  5 :[-1, np.pi/6] # move left backward  
 
## State value for the agent
Output values of the algorithm are v, and the distance between the 4 parking corner points(x,y) and the center position of the car(x,y).  
(v is divided by the maximum velocity and the disance is divided by the maximaum distance for normalization.)
Maximum velocity is +/-10km/h and Maximum distance is 25m.

## Reward type
If the car parks in the parking lot succesfully, the reward +1 is given.  
The reward -1 is given if the following conditions are met.
  - the car get collision.
  - the distance between the car and the parking lot is more than 25meters.
  - the car crosses the parking lot border horizontally/vertically.

# How to use
## Training

## Visualize the agent

