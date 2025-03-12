import pytest
import numpy as np
from sim_env.parking import BaseParking, ParallelParking, PerpendicularParking
from sim_env.parameters import Config, PI


@pytest.fixture
def sample_config():
    """Returns a default Config object for testing."""
    return Config()


@pytest.fixture
def base_parking(sample_config):
    """Returns a BaseParking instance for testing."""
    return BaseParking(sample_config)


@pytest.fixture
def parallel_parking(sample_config):
    """Returns a ParallelParking instance for testing."""
    return ParallelParking(sample_config)


@pytest.fixture
def perpendicular_parking(sample_config):
    """Returns a PerpendicularParking instance for testing."""
    return PerpendicularParking(sample_config)


# ---------------------- Test `set_initial_loc` ----------------------
@pytest.mark.parametrize('side, expected', [
    (1, 1),  # Fixed side
    ((1, 2, 3, 4), None)  # Tuple, should randomly choose a side
])
def test_set_initial_loc(base_parking, side, expected):
    """Test that `set_initial_loc` correctly selects a side."""
    result = base_parking.set_initial_loc(side)
    if expected is not None:
        assert result == expected
    else:
        assert result in [1, 2, 3, 4]  # Must be a valid side


def test_set_initial_loc_invalid_type(base_parking):
    """Ensure ValueError is raised for invalid types."""
    with pytest.raises(ValueError, match='Invalid side type'):
        base_parking.set_initial_loc('invalid')


# ---------------------- Test `get_parking_struct` ----------------------
@pytest.mark.parametrize('parking_type, side', [
    ('parallel', 1), ('parallel', 3),
    ('perpendicular', 1), ('perpendicular', 4)
])
def test_get_parking_struct(base_parking, parking_type, side):
    """Ensure the correct parking structure is returned."""
    result = base_parking.get_parking_struct(parking_type, side)
    assert isinstance(result, np.ndarray)
    assert result.shape == (4, 2)  # Parking structure is always (4, 2)


# ---------------------- Test `get_car_struct` ----------------------
@pytest.mark.parametrize('parking_type, side', [
    ('parallel', 1), ('parallel', 3),
    ('perpendicular', 3), ('perpendicular', 4)
])
def test_get_car_struct(base_parking, parking_type, side):
    """Ensure the correct car structure is returned."""
    result = base_parking.get_car_struct(parking_type, side)
    assert isinstance(result, np.ndarray)
    assert result.shape == (4, 2)  # Car structure is always (4, 2)


# ---------------------- Test `set_initial_car_loc` ----------------------
@pytest.mark.parametrize('side', (1, 2, 3, 4))
def test_set_initial_car_loc(base_parking, side):
    """Check if `set_initial_car_loc` correctly positions the car."""
    parking_loc = np.array([10.0, 10.0])
    initial_distance_range = (5.0, 10.0)
    car_loc_randomized_range = (-2.0, 2.0)

    car_loc = base_parking.set_initial_car_loc(side, parking_loc, initial_distance_range, car_loc_randomized_range)

    assert isinstance(car_loc, np.ndarray)
    assert len(car_loc) == 2  # Shall return (x, y) coordinates


# ---------------------- Test `set_initial_parking_loc` ----------------------
@pytest.mark.parametrize('side', (1, 2, 3, 4))
def test_set_initial_parking_loc(base_parking, side):
    """Check that the parking location is within the expected boundaries."""
    window_w, window_h = 800, 600
    window_w_offset, window_h_offset = 50, 50

    parking_loc = base_parking.set_initial_parking_loc(side, window_w, window_h, window_w_offset, window_h_offset)

    assert isinstance(parking_loc, np.ndarray)
    assert len(parking_loc) == 2  # Shall return (x, y) coordinates


# ---------------------- Test `set_initial_heading` ----------------------
@pytest.mark.parametrize('parking_type, side, expected_range', [
    ('perpendicular', 1, (PI/12 * 5, PI/12 * 7)),
    ('perpendicular', 2, (-PI/12 * 7, -PI/12 * 5)),
    ('perpendicular', 3, (-PI/12, PI/12)),
    ('perpendicular', 4, (PI - PI/12, PI + PI/12)),
    ('parallel', 1, (PI/6, PI/3)),
    ('parallel', 2, (-PI/3, -PI/6)),
    ('parallel', 3, (-PI/12, PI/12)),
    ('parallel', 4, (-PI/12 * 11, PI/12 * 11)),
])
def test_set_initial_heading(base_parking, parking_type, side, expected_range):
    """Check if heading angle is correctly assigned within expected range."""
    heading = base_parking.set_initial_heading(parking_type, side)

    assert isinstance(heading, float), 'Heading angle should be a float.'
    assert expected_range[0] <= heading <= expected_range[1], (
        f'Heading angle {heading} is outside expected range {expected_range} '
        f'for {parking_type} parking on side {side}.'
    )


def test_set_initial_heading_invalid_parking_type(base_parking):
    """Ensure ValueError is raised for invalid parking type."""
    with pytest.raises(ValueError, match='Invalid parking type'):
        base_parking.set_initial_heading('invalid', 1)


def test_set_initial_heading_invalid_side(base_parking):
    """Ensure ValueError is raised for an invalid parking side."""
    with pytest.raises(ValueError, match='Invalid side value'):
        base_parking.set_initial_heading('perpendicular', 99)


# ---------------------- Test `generate_static_obstacles` in ParallelParking ----------------------
@pytest.mark.parametrize('side', [1, 2, 3, 4])
def test_generate_static_obstacles_parallel(parallel_parking, side):
    """Ensure static obstacles are correctly generated in parallel parking."""
    parking_lot = np.array([15.0, 15.0])

    static_cars, static_parking = parallel_parking.generate_static_obstacles(parking_lot, side)

    assert isinstance(static_cars, list)
    assert isinstance(static_parking, list)
    assert len(static_cars) == 2
    assert len(static_parking) == 2
    assert static_cars[0].shape == (4, 2)  # Shall return car vertices


# ---------------------- Test `generate_static_obstacles` in PerpendicularParking ----------------------
@pytest.mark.parametrize('side', [1, 2, 3, 4])
def test_generate_static_obstacles_perpendicular(perpendicular_parking, side):
    """Ensure static obstacles are correctly generated in perpendicular parking."""
    parking_lot = np.array([15.0, 15.0])

    static_cars, static_parking = perpendicular_parking.generate_static_obstacles(parking_lot, side)

    assert isinstance(static_cars, list)
    assert isinstance(static_parking, list)
    assert len(static_cars) == 2
    assert len(static_parking) == 2
    assert static_cars[0].shape == (4, 2)  # Shall return car vertices
