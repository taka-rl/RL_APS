import pytest
import numpy as np
from sim_env.parameters import CarSize, WheelSize, ParkingLotSize, Config, PI


@pytest.mark.parametrize("length, width", [
    (4.0, 2.0),
    (4.5, 2.2)
])
def test_car_size(length, width):
    """Test CarSize dimensions and structures."""
    car = CarSize(length, width)
    assert car.length == np.float32(length)
    assert car.width == np.float32(width)
    assert car.car_struct.shape == (4, 2)  # Shall have 4 corner points


@pytest.mark.parametrize("length, width", [
    (0.75, 0.35),
    (0.8, 0.4)
])
def test_wheel_size(length, width):
    """Test WheelSize dimensions and structures."""
    wheel = WheelSize(length, width)
    assert wheel.length == np.float32(length)
    assert wheel.width == np.float32(width)
    assert wheel.wheel_struct.shape == (4, 2)  # Shall have 4 corner points
    assert wheel.wheel_pos.shape == (4, 2)  # Shall have 4 wheels


@pytest.mark.parametrize("length, width", [
    (6.0, 4.0),
    (7.0, 4.5)
])
def test_parking_lot_size(length, width):
    """Test ParkingLotSize dimensions and different parking lot structures."""
    parking = ParkingLotSize(length, width)
    assert parking.length == np.float32(length)
    assert parking.width == np.float32(width)
    assert parking.parallel_horizontal.shape == (4, 2)
    assert parking.perpendicular_vertical.shape == (4, 2)


def test_config_defaults():
    """Test default values in Config."""
    config = Config()
    assert config.car_size.length == np.float32(4.0)
    assert config.wheel_size.length == np.float32(0.75)
    assert config.parking_lot_size.length == np.float32(6.0)

    # Reward and State settings
    assert config.reward_type == 'type1'
    assert config.state_type == 'type1'

    # Action limits
    assert config.acceleration_limit == np.float32(1.0)
    assert config.steering_limit == np.float32(PI / 4)
    assert config.velocity_limit == np.float32(10.0)
    assert config.dt == np.float32(0.1)
    assert config.max_distance == np.float32(25.0)
    assert config.max_steps == 80

    # Guidance reward
    assert config.max_angle_error == np.float32(PI / 12)
    assert config.center_threshold == np.float32(1.0)

    # Rendering settings
    assert config.fps == 30
    assert config.window_width == 800
    assert config.window_height == 600
    assert config.grid_size == 20

    assert config.colors == {
        "RED": (255, 100, 100),
        "GREEN": (0, 255, 0),
        "BLUE": (100, 200, 255),
        "YELLOW": (200, 200, 0),
        "BLACK": (0, 0, 0),
        "GREY": (100, 100, 100),
        "WHITE": (255, 255, 255),
        "GRID_COLOR": (200, 200, 200)
    }


def test_config_custom():
    """Test custom configuration initialization."""
    custom_config = Config(car_length=4.8, car_width=2.1,
                           max_distance=30.0, max_steps=100,
                           acceleration_limit=1.5, steering_limit=PI/5,
                           reward_type='type2', state_type='type3'
                           )
    assert custom_config.car_size.length == np.float32(4.8)
    assert custom_config.car_size.width == np.float32(2.1)
    assert custom_config.max_distance == np.float32(30.0)
    assert custom_config.max_steps == 100

    # Reward and State settings
    assert custom_config.reward_type == 'type2'
    assert custom_config.state_type == 'type3'

    # Action limits
    assert custom_config.acceleration_limit == np.float32(1.5)
    assert custom_config.steering_limit == np.float32(PI / 5)
