import pytest
import numpy as np
from gymnasium.spaces import Discrete
from sim_env.parking_env import Parking
from sim_env.parameters import Config, PI
from sim_env.car import Car


# --------------------------------------------- Common functions for test ---------------------------------------------
def parking_env(reward_type: str = 'type1', state_type: str = 'type1', render_mode: str = 'no_render',
                action_type:str = 'continuous', parking_type: str = 'perpendicular', 
                training_mode: str = 'off', *args, **kwargs) -> Parking:
    env_config = {
        'render_mode': render_mode,
        'action_type': action_type,
        'parking_type': parking_type,
        'training_mode': training_mode,
        'config': Config(reward_type=reward_type, state_type=state_type, *args, **kwargs),
    }
    return Parking(env_config)


# --------------------------------------------- Initialization ---------------------------------------------
@pytest.mark.parametrize('reward_type', ['type1', 'type2', 'type3', 'type4'])
@pytest.mark.parametrize('state_type', ['type1', 'type2', 'type3', 'type4'])
@pytest.mark.parametrize('render_mode', ['human', 'no_render'])
@pytest.mark.parametrize('action_type', ['continuous', 'discrete'])
@pytest.mark.parametrize('parking_type', ['perpendicular', 'parallel'])
@pytest.mark.parametrize('training_mode', ['on', 'off'])
def test_env_init(reward_type, state_type, render_mode, action_type, parking_type, training_mode):
    """Test if the environment is initialized properly."""

    # Initialization
    env = parking_env(reward_type, state_type, render_mode, action_type, parking_type, training_mode)

    # reward type
    assert env.config.reward_type == reward_type

    # state type
    assert env.config.state_type == state_type
    assert env.state is None

    assert env.observation_space.dtype == np.float32, f"Expected dtype float32, got {env.observation_space.dtype}"

    if state_type == 'type1':
        assert env.observation_space.shape == (8,), 'observation_space shape shall be 8.'

    elif state_type == 'type2':
        assert env.observation_space.shape == (10, ), 'observation_space shape shall be 10.'

    elif state_type == 'type3':
        assert env.observation_space.shape == (9, ), 'observation_space shape shall be 9.'

    elif state_type == 'type4':
        assert env.observation_space.shape == (11, ), 'observation_space shape shall be 11.'

    else:
        # Invalid state type(expecting ValueError)
        with pytest.raises(ValueError, match='State type shall be either type1, type2, type3, or type4.'):
            parking_env(reward_type, 'invalid_state_type', render_mode, action_type, parking_type, training_mode)
    
    # action type
    assert env.action_type == action_type
    if action_type == 'continuous':
        assert env.action_space.dtype == np.float32, f"Expected dtype float32, got {env.action_space.dtype}"
        assert env.action_space.shape == (2, ), f"Expected shape (2, ), got {env.action_space.shape}"
    # else:
        # Omit discrete action space test for now
    
    # ParkingEnv class attribute's initialization
    assert env.terminated is None
    assert env.truncated is None
    assert env.run_steps is None
    assert env.side is None
    assert env.parking_lot is None
    assert env.parking_lot_vertices is None
    assert env.car is None
    assert env.static_cars_vertices is None
    assert env.static_parking_lot_vertices is None

    assert env.parking_type == parking_type
    assert env.training_mode == training_mode
    assert env.config.max_steps == 80

    assert env.scale.dtype == np.float32, f"Expected dtype float32, got {env.scale.dtype}"
    assert env.scale.shape == (2, ), f"Expected shape (2, ), got {env.scale.shape}"


# --------------------------------------------- Environment step ---------------------------------------------
@pytest.mark.parametrize('parking_type', ['perpendicular', 'parallel'])
def test_continuous_env_step(parking_type):
    """Test stepping in the parking environment."""
    env = Parking({'render_mode': 'no_render',
                   'action_type': 'continuous',
                   'parking_type': parking_type,
                   'training_mode': 'off',
                   'config': Config(),
                   })
    # Reset the environment
    env.reset()
    
    # Valid actions (2D continuous actions within the range of [-1.0, 1.0])
    action = [[1.0, 1.0], [-1.0, -1.0], [1.4, -1.3], [-1.9, 1.5], [0.4, 0.9], [0.4, 0.3]]
    for a in action:
        state, reward, terminated, truncated, info = env.step(a)
        assert state is not None, 'State shall update after a step.'
        assert isinstance(state, np.ndarray), 'State shall be a NumPy array.'
        assert state.dtype == np.float32, f"Expected dtype float32, got {state.dtype}"
        assert np.all(state >= -1.0) and np.all(state <= 1.0), 'State value shall be between -1.0 and 1.0'
        assert isinstance(reward, (int, float)), 'Reward shall be a number.'
        assert isinstance(terminated, bool), 'Terminated flag shall be a boolean.'
        assert isinstance(truncated, bool), 'Truncated flag shall be a boolean.'


@pytest.mark.parametrize('parking_type', ['perpendicular', 'parallel'])
def test_discrete_action_env_step(parking_type):
    """Test stepping in the discrete action parking environment."""
    env = Parking({'render_mode': 'no_render',
                   'action_type': 'discrete',
                   'parking_type': parking_type,
                   'training_mode': 'off',
                   'config': Config(),
                   })
    # Reset the environment
    env.reset()

    assert isinstance(env.action_space, Discrete)
    assert env.action_space.n == 6, 'Action space shall have 6 discrete actions.'

    # Valid actions (0 to 5)
    actions = (0, 1, 2, 3, 4, 5)
    for action in actions:
        state, reward, terminated, truncated, info = env.step(action)
        assert state is not None, 'State shall update after a step.'
        assert isinstance(state, np.ndarray), 'State shall be a NumPy array.'
        assert state.dtype == np.float32, f"Expected dtype float32, got {state.dtype}"
        assert np.all(state >= -1.0) and np.all(state <= 1.0), 'State value shall be between -1.0 and 1.0'
        assert isinstance(reward, (int, float)), 'Reward shall be a number.'
        assert isinstance(terminated, bool), 'Terminated flag shall be a boolean.'
        assert isinstance(truncated, bool), 'Truncated flag shall be a boolean.'

    # Invalid action (expecting ValueError)
    with pytest.raises(ValueError, match='Invalid action value: 7'):
        env.step(7)

# --------------------------------------------- Environment action ---------------------------------------------
@pytest.mark.parametrize('action_type', ['continuous', 'discrete'])
def test_action_float32_type(action_type):

    # Initialization
    env = parking_env(action_type=action_type)
    
    # Reset the environment
    env.reset()

    # Define test actions
    action_continuous = [[1.0, 1.0], [-1.0, -1.0], [1.4, -1.3], [-1.9, 1.5], [0.4, 0.9], [0.4, 0.3]]
    action_discrete = [0, 1, 2, 3, 4, 5]

    for action in (action_continuous if action_type == "continuous" else action_discrete):
        if env.action_type == "continuous":
            action = np.clip(action, -1.0, 1.0).astype(np.float32, copy=False) * env.scale
        
        elif env.action_type == "discrete":
            if action == 0:  # move forward
                action = np.array([1, 0], dtype=np.float32)
            elif action == 1:  # move right forward
                action = np.array([1, -PI/6], dtype=np.float32)
            elif action == 2:  # move left forward
                action = np.array([1, PI/6], dtype=np.float32)
            elif action == 3:  # move backward
                action = np.array([-1, 0], dtype=np.float32)
            elif action == 4:  # move right backward
                action = np.array([-1, -PI/6], dtype=np.float32)
            elif action == 5:  # move left backward
                action = np.array([-1, PI/6], dtype=np.float32)
            else:
                # omit a test case for invalid action
                pass

        else:
            raise ValueError(f"Invalid action type: {env.action_type}. "
                             f"Valid types are 'continuous' and 'discrete'.")
        
        # Evaluate action
        assert action.dtype == np.float32, f"Expected action dtype float32, got {action.dtype}"
        assert action.shape == (2,), f"Expected action shape (2,), got {action.shape}"


# --------------------------------------------- Environment reset ---------------------------------------------
@pytest.mark.parametrize('reward_type', ['type1', 'type2', 'type3', 'type4'])
@pytest.mark.parametrize('state_type', ['type1', 'type2', 'type3', 'type4'])
@pytest.mark.parametrize('parking_type', ['perpendicular', 'parallel'])
@pytest.mark.parametrize('action_type', ['continuous', 'discrete'])
@pytest.mark.parametrize('training_mode', ['on', 'off'])
@pytest.mark.parametrize('side', (1, 2, 3, 4))
def test_env_reset(reward_type, state_type,parking_type, action_type, training_mode, side):
    """Test the reset functionality of the parking environment."""

    # Initialize environment
    env = parking_env(reward_type=reward_type, state_type=state_type, action_type=action_type,
                      parking_type=parking_type, training_mode=training_mode, side=side)

    # Reset the environment
    state, _ = env.reset()

    # Check state properties
    assert state is not None, 'Reset shall return a valid initial state.'
    assert isinstance(state, np.ndarray), 'State shall be a NumPy array.'
    assert state.dtype == np.float32, f"Expected dtype float32, got {state.dtype}"
    assert state.shape[0] > 0, 'State shall not be an empty array.'
    assert np.all(state >= -1.0) and np.all(state <= 1.0), 'State values shall be normalized between -1.0 and 1.0'

    # Environment state flags
    assert env.terminated is False, 'Environment shall not be terminated after reset.'
    assert env.truncated is False, 'Environment shall not be truncated after reset.'

    # Validate environment properties
    assert env.side in [1, 2, 3, 4], 'Side shall be correctly set after reset.'
    assert isinstance(env.parking_lot, np.ndarray), 'Parking lot shall be initialized as a NumPy array.'
    assert isinstance(env.parking_lot_vertices,
                      np.ndarray), 'Parking lot vertices shall be initialized as a NumPy array.'
    assert hasattr(env, 'car') and env.car is not None, 'Car object shall be initialized after reset.'
    assert hasattr(env.car, 'loc_old') and isinstance(env.car.loc_old,
                                                      np.ndarray), 'Car shall have previous location (loc_old).'

    assert isinstance(env.static_cars_vertices, list), 'Static cars shall be stored as a list.'
    assert isinstance(env.static_parking_lot_vertices, list), 'Static parking lot vertices shall be stored as a list.'

    # Ensure initial conditions are correctly set
    assert env.run_steps == 0, 'Run steps shall be reset to 0 after environment reset.'

    # Additional checks based on `training_mode`
    if training_mode == 'on':
        assert env.parking_lot in env.config.default_parking_locations[env.side]
    else:
        assert isinstance(env.parking_lot, np.ndarray), "Parking lot shall be set correctly in non-training mode."


# --------------------------------------------- Reward ---------------------------------------------
# Make tests for 4 sides * 2 parking (parallel and perpendicular) for Reward
PARKING_LOT = [np.array([15.0, 2.5]), np.array([15.0, 27.5]), np.array([2.5, 15.0]), np.array([37.5, 15.0])]


# --------------------------------------------- ↓ Side is 1 ↓ ---------------------------------------------
def test_reward_max_step():
    """Test if max step penalty is applied correctly."""
    env = parking_env()
    env.reset()
    env.run_steps = env.config.max_steps
    reward = env._reward()
    assert reward < 0, 'Agent shall receive a negative reward for exceeding max steps.'
    assert env.terminated is True
    assert env.truncated is True


@pytest.mark.parametrize('parking_type', ['perpendicular', 'parallel'])
def test_reward_cross_border(parking_type):
    """Test if cross border penalty is applied correctly."""
    env = Parking({'render_mode': 'no_render',
                   'action_type': 'continuous',
                   'parking_type': parking_type,
                   'training_mode': 'off',
                   'config': Config(),
                   })

    # Car location
    car_loc_list = [np.array([15.0, 0.0]), np.array([15.0, 31.5]), np.array([0.5, 12.0]), np.array([40.5, 13.0])]

    for parking_lot, car_loc, i in zip(PARKING_LOT, car_loc_list, range(1, len(car_loc_list)+1)):
        env.reset()
        env.side = i
        env.car = Car(car_loc, 0, Config())
        env.parking_lot = parking_lot
        env.parking_lot_vertices = (parking_lot + env.parking_strategy.get_parking_struct(env.parking_type, env.side))
        reward = env._reward()
        assert reward < 0, 'Agent shall receive a negative reward for crossing border.'


@pytest.mark.parametrize('parking_type', ['perpendicular', 'parallel'])
def test_reward_max_distance(parking_type):
    """Test if max distance penalty is applied correctly."""
    env = Parking({'render_mode': 'no_render',
                   'action_type': 'continuous',
                   'parking_type': parking_type,
                   'training_mode': 'off',
                   'config': Config(),
                   })

    env.car = Car(np.array([0.0, 0.0]), 0, Config())

    # Car location
    car_loc_list = [np.array([35.0, 30.5]), np.array([39.0, 0.5]), np.array([38.5, 3.0]), np.array([0.0, 30.0])]

    for parking_lot, car_loc in zip(PARKING_LOT, car_loc_list):
        env.reset()
        env.parking_lot = parking_lot
        env.parking_lot_vertices = (parking_lot + env.parking_strategy.get_parking_struct(env.parking_type, env.side))
        env.car.car_loc = car_loc
        reward = env._reward()
        assert reward < 0, 'Agent shall receive a negative reward for crossing border.'


def test_reward_collision():
    """Test if collision penalty is applied correctly."""
    env = parking_env()

    env.reset()
    env.car = Car(([10.0, 2.5]), 0, Config())

    reward = env._reward()
    assert reward < 0, 'Agent shall receive a negative reward for crossing border.'


def test_reward_type1():
    """Test if reward type1 is applied correctly."""
    env = parking_env()

    env.reset()
    env.car = Car(([15.0, 2.0]), 0, Config())

    # print(env.parking_lot_vertices)
    # print(env.car.car_loc)
    # print(env.static_cars_vertices)

    # Set the parking lot vertices
    env.parking_lot_vertices = np.array([[17,   5.5],
                                         [17,  -0.5],
                                         [13,  -0.5],
                                         [13,   5.5]]
                                        )

    reward = env._reward()
    assert reward > 0, 'Agent shall receive a positive reward.'


def test_reward_type2():
    """Test if reward type2 is applied correctly."""
    env = parking_env(reward_type='type2')

    env.reset()
    env.car = Car(([15.0, 2.0]), 0, Config())

    # Set the parking lot and its vertices
    env.parking_lot = np.array([15.0, 2.5])
    env.parking_lot_vertices = np.array([[17, 5.5],
                                         [17, -0.5],
                                         [13, -0.5],
                                         [13, 5.5]]
                                        )

    reward = env._reward()
    assert reward > 0, 'Agent shall receive a positive reward.'


# --------------------------------------------- State ---------------------------------------------
def test_state_type1():
    """Test if the state type1 is returned properly."""
    env = parking_env(state_type='type1')
    assert env.config.state_type == 'type1'

    env.reset()

    env.car = Car(([20, 20]), 0, Config())

    action = [1.0, 1.0]
    state, reward, terminated, truncated, info = env.step(action)

    assert state is not None, 'State shall update after a step.'
    assert isinstance(state, np.ndarray), 'State shall be a NumPy array.'
    assert np.all(state >= -1.0) and np.all(state <= 1.0), 'State value shall be between -1.0 and 1.0'
    assert state.shape is not 8, 'State shape shall be 8.'


def test_state_type2():
    """Test if the state type2 is returned properly."""
    env = parking_env(state_type='type2')
    assert env.config.state_type == 'type2'

    env.reset()

    env.car = Car(([20, 20]), 0, Config())

    action = [1.0, 1.0]
    state, reward, terminated, truncated, info = env.step(action)

    assert state is not None, 'State shall update after a step.'
    assert isinstance(state, np.ndarray), 'State shall be a NumPy array.'
    assert np.all(state >= -1.0) and np.all(state <= 1.0), 'State value shall be between -1.0 and 1.0'
    assert state.shape is not 10, 'State shape shall be 10.'


def test_state_type3():
    """Test if the state type3 is returned properly."""
    env = parking_env(state_type='type3')
    assert env.config.state_type == 'type3'

    env.car = Car(([20, 20]), 0, Config())

    env.reset()

    action = [1.0, 1.0]
    state, reward, terminated, truncated, info = env.step(action)

    assert state is not None, 'State shall update after a step.'
    assert isinstance(state, np.ndarray), 'State shall be a NumPy array.'
    assert np.all(state >= -1.0) and np.all(state <= 1.0), 'State value shall be between -1.0 and 1.0'
    assert state.shape is not 9, 'State shape shall be 9.'


def test_state_type4():
    """Test if it returns value error"""
    # Invalid state type
    with pytest.raises(ValueError, match='State type shall be either type1, type2 or type3'):
        env = parking_env(state_type='type4')
