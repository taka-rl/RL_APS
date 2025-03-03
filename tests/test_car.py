import pytest
import numpy as np
from sim_env.car import Car
from sim_env.parameters import Config, PI


@pytest.fixture
def env_car():
    config = Config()
    car_loc = np.array([0.0, 0.0])
    psi = 0.0
    return Car(car_loc, psi, config)


def test_car_initialization(env_car):
    """Test if the car is initialized properly."""
    assert np.all(env_car.car_loc == np.array([0.0, 0.0]))
    assert env_car.psi == 0.0
    assert env_car.v == 0.0
    assert env_car.delta == 0.0


def test_car_kinematic_act(env_car):
    """Test if the car is calculated following the kinematic equation."""
    action = np.array([1.0, 1.0])
    prev_loc = env_car.car_loc.copy()

    # Kinematic actions
    env_car.kinematic_act(action)
    env_car.kinematic_act(action)

    assert not np.array_equal(env_car.car_loc, prev_loc), "Car location shall change after action."


def test_rotate_car(env_car):
    """Test if the car is rotated by 90 degrees."""
    # Set psi to 90 degrees
    env_car.psi = PI / 2

    # Rotation
    rotated = env_car.calc_car_vertices()

    expected = np.array([[-1.0, 2.0],
                         [1.0, 2.0],
                         [1.0, -2.0],
                         [-1.0, -2.0]])

    assert np.allclose(rotated, expected, atol=1e-5), "Rotation failed!"

