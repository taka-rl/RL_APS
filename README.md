# RL_APS
This repository is a development environment for my thesis which is "Reinforcement learning based automated parking systems". It provides the training envrionment for the agent in both parallel and perpendicular parkings.

# Folder structure
- sim_env: contains the necessary scripts for the parking simulation
  - parking_env.py:
  - car.py:
  - com_fcn.py:
  - parameters.py:
- practice_rllib: to learn how to use Ray RLlib for my thesis
- practice_pygame: to learn how to draw the parking environment for my thesis

# Simulation Environment
- Python
- Gymnasium
- Ray RLlib

# Parking type
This environment provides Parallel and Perpendicular parking.  
# Kinematic Bicycle model
- Kinematic bicycle model equation
 
        x_dot = v * np.cos(ψ)
        y_dot = v * np.sin(ψ)
        v_dot = a
        psi_dot = v * np.tan(delta) / CAR_L
  v: velocity, ψ(psi): heading angle of the car, a: acceleration, δ(delta): steering angle, CAR_L: car length  
  x: the center of the car position in x axle, y: the center of the car position in y axle  
Maximum velocity is set +/-10km/h  
Acceleration limitation is -1〜1 m/s^2

# State value for the agent
Input values are [a, δ], setting -pi/4〜pi/4 for δ.  
Output values are v, and the distance between the 4 parking corner points(x,y) and the center position of the car(x,y).  
(v is divided by the maximum velocity and the disance is divided by the maximaum distance for normalization.)

# Reward type
If the car parks in the parking lot succesfully, the reward +1 is given.  
The reward -1 is given if the following conditions are met.
  - the car get collision.
  - the distance between the car and the parking lot is more than 25meters.
  - the car crosses the parking lot border horizontally/vertically.

