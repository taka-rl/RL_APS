import pytest
import numpy as np
from gymnasium.spaces import Discrete

from sim_env.parking_env import Parking
from sim_env.parameters import Config


# --------------------------------------------- Common functions for test ---------------------------------------------
@pytest.fixture
def continuous_perpendicular_env():
    env_config = {
        "render_mode": "no_render",
        "action_type": "continuous",
        "parking_type": "perpendicular",
        "training_mode": "off",
        "config": Config()
    }
    return Parking(env_config)


@pytest.fixture
def continue_parallel_env():
    env_config = {
        "render_mode": "no_render",
        "action_type": "continuous",
        "parking_type": "parallel",
        "training_mode": "off",
        "config": Config()
    }
    return Parking(env_config)


@pytest.fixture
def discrete_perpendicular_env():
    env_config = {
        "render_mode": "no_render",
        "action_type": "discrete",
        "parking_type": "perpendicular",
        "training_mode": "off",
        "config": Config()
    }
    return Parking(env_config)


@pytest.fixture
def discrete_parallel_env():
    env_config = {
        "render_mode": "no_render",
        "action_type": "discrete",
        "parking_type": "parallel",
        "training_mode": "off",
        "config": Config()
    }
    return Parking(env_config)


# --------------------------------------------- Initialization ---------------------------------------------
def test_continue_perpendicular_env_init(continuous_perpendicular_env):
    """Test if the continue and perpendicular parking environment is initialized properly."""
    assert continuous_perpendicular_env.action_type == "continuous"
    assert continuous_perpendicular_env.parking_type == "perpendicular"
    assert continuous_perpendicular_env.config.max_steps == 80


def test_continue_parallel_init(continue_parallel_env):
    """Test if the continue parallel parking environment is initialized properly."""
    assert continue_parallel_env.action_type == "continuous"
    assert continue_parallel_env.parking_type == "parallel"
    assert continue_parallel_env.config.max_steps == 80


def test_discrete_perpendicular_env_init(discrete_perpendicular_env):
    """Test if the discrete and perpendicular parking environment is initialized properly."""
    assert discrete_perpendicular_env.action_type == "discrete"
    assert discrete_perpendicular_env.parking_type == "perpendicular"
    assert discrete_perpendicular_env.config.max_steps == 80


def test_discrete_parallel_init(discrete_parallel_env):
    """Test if the discrete parallel parking environment is initialized properly."""
    assert discrete_parallel_env.action_type == "discrete"
    assert discrete_parallel_env.parking_type == "parallel"
    assert discrete_parallel_env.config.max_steps == 80


# --------------------------------------------- Environment step ---------------------------------------------
def test_continue_perpendicular_env_step(continuous_perpendicular_env):
    """Test stepping in the continue and perpendicular parking environment."""
    # Reset the environment
    continuous_perpendicular_env.reset()

    # Move forward
    action = [1.0, 0.0]
    state, reward, terminated, truncated, info = continuous_perpendicular_env.step(action)

    assert state is not None, "State shall update after a step."
    assert -1.0 <= state.all() <= 1.0, "State value shall be between -1.0 and 1.0"
    assert isinstance(reward, (int, float)), "Reward shall be a number."
    assert isinstance(terminated, bool), "Terminated flag shall be a boolean."
    assert isinstance(truncated, bool), "Truncated flag shall be a boolean."


def test_discrete_perpendicular_env_step(discrete_perpendicular_env):
    """Test stepping in the discrete and perpendicular parking environment."""
    # Reset the environment
    discrete_perpendicular_env.reset()
    assert isinstance(discrete_perpendicular_env.action_space, Discrete)
    assert discrete_perpendicular_env.action_space.n == 6, "Action space should have 6 discrete actions."

    # Valid actions (0 to 5)
    actions = (0, 1, 2, 3, 4, 5)
    for action in actions:
        state, reward, terminated, truncated, info = discrete_perpendicular_env.step(action)
        assert state is not None, "State shall update after a step."
        assert isinstance(state, np.ndarray), "State should be a NumPy array."
        assert np.all(state >= -1.0) and np.all(state <= 1.0), "State value shall be between -1.0 and 1.0"
        assert isinstance(reward, (int, float)), "Reward shall be a number."
        assert isinstance(terminated, bool), "Terminated flag shall be a boolean."
        assert isinstance(truncated, bool), "Truncated flag shall be a boolean."

    # Invalid action (expecting ValueError)
    with pytest.raises(ValueError, match="Invalid action value: 7"):
        discrete_perpendicular_env.step(7)


# --------------------------------------------- Environment reset ---------------------------------------------
def test_continue_perpendicular_env_reset(continuous_perpendicular_env):
    """Test reset functionality."""
    state, _ = continuous_perpendicular_env.reset()
    assert state is not None, "Reset shall return an initial state."
    assert not continuous_perpendicular_env.terminated, "Environment shall not be terminated after reset."
    assert not continuous_perpendicular_env.truncated, "Environment shall not be truncated after reset."


# --------------------------------------------- Reward ---------------------------------------------

# --------------------------------------------- State ---------------------------------------------
