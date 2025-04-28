# RL_APS
This repository is a development environment for "Reinforcement Learning-Based Automated Parking Systems." It provides the training environment for the agent in both parallel and perpendicular parking scenarios.

The following videos present the trained agent's behavior in the parking simulation.

- Perpendicular Parking
  
https://github.com/user-attachments/assets/9fbfc305-6577-42bf-bd06-6ae67add61e9

- Parallel Parking  
Uploaded later under development.

For more results, Please refer to [doc/training_results.md.](https://github.com/taka-rl/RL_APS/blob/8-update-the-repository/doc/training_results.md)

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
    │   ├── traning_results             # training results
    │   ├── trained_agents              # trained agents
    │   ├── training.py                 # for training
    │   └── utility.py                  # utility functions
    ├── doc                             # documents for this repository
    ├── old                              
    ├── practice_pygame                 # pygame practice
    ├── practice_rllib                  # Ray RLlib practice
    ├── requirements.txt                # Required dependencies
    └── README.md                       # Project documentation



## Simulation Environment
Please refer to the [doc/simulation_environment.md](https://github.com/taka-rl/RL_APS/blob/8-update-the-repository/doc/simulation_environment.md) for the following information.
- Simulation Environment
- Environment Description
- Kinematic Bicycle Model
- Reinforcement Learning/Environment, Action, State, Reward

## How to use
### Used tools
The libraries and their versions are as follows.

| tool      | version |
|-----------|---------|
| Python    | 3.10.11 |
| Gymnasium | 0.28.1  | 
| Ray RLlib | 2.9.0   |
| Numpy     | 1.26.3  |
| Pygame    | 2.1.3   |

### install tools
The first is to install necessary libraries.  

ray rllib: `pip install "ray[rllib]" tensorflow`  
Gymnasium: `pip install "gymnasium[all]"`


### Settings for training, Visualizing the agent and Evaluation
Please refer to [doc/training_procedures.md](https://github.com/taka-rl/RL_APS/blob/8-update-the-repository/doc/training_procedures.md) for the training setting.  

## Training results
Please refer to [doc/training_results.md](https://github.com/taka-rl/RL_APS/blob/8-update-the-repository/doc/training_results.md).

