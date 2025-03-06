import pytest
import numpy as np
from gymnasium.spaces import Discrete

from sim_env.parking_env import Parking
from sim_env.parameters import Config
from sim_env.car import Car


# --------------------------------------------- Common functions for test ---------------------------------------------
def parking_env(reward_type: str = 'type1', state_type: str = 'type1'):
    env_config = {
        'render_mode': 'no_render',
        'action_type': 'continuous',
        'parking_type': 'perpendicular',
        'training_mode': 'on',
        'config': Config(reward_type=reward_type, state_type=state_type)
    }
    return Parking(env_config)


# --------------------------------------------- Initialization ---------------------------------------------
@pytest.mark.parametrize('parking_type', ['perpendicular', 'parallel'])
@pytest.mark.parametrize('action_type', ['continuous', 'discrete'])
def test_env_init(parking_type, action_type):
    """Test if the environment is initialized properly."""

    env = Parking({'render_mode': 'no_render',
                   'action_type': action_type,
                   'parking_type': parking_type,
                   'training_mode': 'off',
                   'config': Config(),
                   })
    assert env.action_type == action_type
    assert env.parking_type == parking_type
    assert env.config.max_steps == 80


# --------------------------------------------- Environment step ---------------------------------------------
@pytest.mark.parametrize('parking_type', ['perpendicular', 'parallel'])
@pytest.mark.parametrize('action_type', ['continuous', 'discrete'])
def test_env_step(parking_type, action_type):
    """Test stepping in the parking environment."""
    env = Parking({'render_mode': 'no_render',
                   'action_type': action_type,
                   'parking_type': parking_type,
                   'training_mode': 'off',
                   'config': Config(),
                   })
    # Reset the environment
    env.reset()

    # Move forward
    if action_type == 'continuous':
        action = [1.0, 0.0]
    elif action_type == 'discrete':
        action = int(0)
    state, reward, terminated, truncated, info = env.step(action)

    assert state is not None, 'State shall update after a step.'
    assert isinstance(state, np.ndarray), 'State shall be a NumPy array.'
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
        assert np.all(state >= -1.0) and np.all(state <= 1.0), 'State value shall be between -1.0 and 1.0'
        assert isinstance(reward, (int, float)), 'Reward shall be a number.'
        assert isinstance(terminated, bool), 'Terminated flag shall be a boolean.'
        assert isinstance(truncated, bool), 'Truncated flag shall be a boolean.'

    # Invalid action (expecting ValueError)
    with pytest.raises(ValueError, match='Invalid action value: 7'):
        env.step(7)


# --------------------------------------------- Environment reset ---------------------------------------------
@pytest.mark.parametrize('parking_type', ['perpendicular', 'parallel'])
@pytest.mark.parametrize('action_type', ['continuous', 'discrete'])
def test_env_reset(parking_type, action_type):
    """Test reset functionality."""
    env = Parking({'render_mode': 'no_render',
                   'action_type': action_type,
                   'parking_type': parking_type,
                   'training_mode': 'off',
                   'config': Config(),
                   })
    # Reset the environment
    env.reset()

    state, _ = env.reset()
    assert state is not None, 'Reset shall return an initial state.'
    assert isinstance(state, np.ndarray), 'State shall be a NumPy array.'
    assert np.all(state >= -1.0) and np.all(state <= 1.0), 'State value shall be between -1.0 and 1.0'
    assert not env.terminated, 'Environment shall not be terminated after reset.'
    assert not env.truncated, 'Environment shall not be truncated after reset.'


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
    env.car = Car(([10.0, 4.0]), 0, Config())

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
