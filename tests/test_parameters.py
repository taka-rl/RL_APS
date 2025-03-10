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

    # Parking lot and Car location, and car's heading angle for training
    expected_parking_locations = {1: np.array([15.0, 2.5]),
                                  2: np.array([15.0, 27.5]),
                                  3: np.array([2.5, 15.0]),
                                  4: np.array([37.5, 15.0])}

    for key in expected_parking_locations:
        assert np.array_equal(config.default_parking_locations[key], expected_parking_locations[key])

    assert config.side == 1
    assert type(config.side) is int
    assert config.car_loc_randomize_range == (-5, 5)
    assert config.initial_distance_range == (7.5, 15.0)
    assert config.heading_angle_range == {
        "perpendicular": {
            1: (PI / 12 * 5, PI / 12 * 7),
            2: (-PI / 12 * 7, -PI / 12 * 5),
            3: (-PI / 12, PI / 12),
            4: (PI - PI / 12, PI + PI / 12),
        },
        "parallel": {
            1: (PI / 6, PI / 3),
            2: (-PI / 6, -PI / 3),
            3: (-PI / 12, PI / 12),
            4: (-PI / 12 * 11, PI / 12 * 11),
        }
    }


def test_config_custom():
    """Test custom configuration initialization."""
    custom_config = Config(car_length=4.8, car_width=2.1,
                           max_distance=30.0, max_steps=100,
                           acceleration_limit=1.5, steering_limit=PI/5,
                           reward_type='type2', state_type='type3',
                           side=2, default_parking_locations={1: np.array([12.0, 5.5]), 2: np.array([18.0, 25.5]),
                                                              3: np.array([3.5, 16.0]), 4: np.array([36.5, 12.0])},
                           car_loc_randomize_range=(-7.0, 7.0), initial_distance_range=(10, 17.5),
                           heading_angle_range={"perpendicular": {1: (PI / 12 * 6, PI / 12 * 8),
                                                                  2: (-PI / 12 * 9, -PI / 12 * 6),
                                                                  3: (-PI / 6, PI / 6),
                                                                  4: (PI - PI / 8, PI + PI / 8),
                                                                  },
                                                "parallel": {1: (PI / 6, PI / 3),
                                                             2: (-PI / 6, -PI / 3),
                                                             3: (-PI / 12, PI / 12),
                                                             4: (-PI / 12 * 11, PI / 12 * 11)
                                                             }
                                                }

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

    # Parking lot and Car location, and car's heading angle for training
    expected_parking_locations = {1: np.array([12.0, 5.5]),
                                  2: np.array([18.0, 25.5]),
                                  3: np.array([3.5, 16.0]),
                                  4: np.array([36.5, 12.0])
                                  }
    for key in expected_parking_locations:
        assert np.array_equal(custom_config.default_parking_locations[key], expected_parking_locations[key])

    assert custom_config.side == 2
    assert type(custom_config.side) is int

    custom_config.side = (3, 4)
    assert type(custom_config.side) is tuple
    assert custom_config.side == (3, 4)

    assert custom_config.car_loc_randomize_range == (-7.0, 7.0)
    assert custom_config.initial_distance_range == (10, 17.5)
    assert custom_config.heading_angle_range == {"perpendicular": {1: (PI / 12 * 6, PI / 12 * 8),
                                                                   2: (-PI / 12 * 9, -PI / 12 * 6),
                                                                   3: (-PI / 6, PI / 6),
                                                                   4: (PI - PI / 8, PI + PI / 8),
                                                                   },
                                                 "parallel": {1: (PI / 6, PI / 3),
                                                              2: (-PI / 6, -PI / 3),
                                                              3: (-PI / 12, PI / 12),
                                                              4: (-PI / 12 * 11, PI / 12 * 11)
                                                              }
                                                 }

    assert set(custom_config.heading_angle_range["parallel"].keys()) == {1, 2, 3, 4}
    assert set(custom_config.heading_angle_range["perpendicular"].keys()) == {1, 2, 3, 4}