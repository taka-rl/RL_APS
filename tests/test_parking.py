import numpy as np
import pytest
from sim_env.parameters import PI, PIXEL_TO_METER_SCALE


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
@pytest.mark.parametrize('parking_type', ('parallel', 'perpendicular'))
@pytest.mark.parametrize('side', (1, 2, 3, 4))
def test_get_parking_struct(base_parking, parking_type, side):
    """Ensure the correct parking structure is returned."""
    result = base_parking.get_parking_struct(parking_type, side)
    assert isinstance(result, np.ndarray)
    assert result.shape == (4, 2)  # Parking structure is always (4, 2)

    # Get expected parking structure
    expected_structure = None
    if parking_type == "parallel":
        expected_structure = (
            base_parking.parking_lot_size.parallel_horizontal if side in [1, 2]
            else base_parking.parking_lot_size.parallel_vertical
        )
    elif parking_type == "perpendicular":
        expected_structure = (
            base_parking.parking_lot_size.perpendicular_horizontal if side in [1, 2]
            else base_parking.parking_lot_size.perpendicular_vertical
        )

    # Compare NumPy arrays correctly
    assert np.array_equal(result, expected_structure), (
        f"Parking structure mismatch for parking_type={parking_type}, side={side}"
    )


# ---------------------- Test `get_car_struct` ----------------------
@pytest.mark.parametrize('parking_type', ('parallel', 'perpendicular'))
@pytest.mark.parametrize('side', (1, 2, 3, 4))
def test_get_car_struct(base_parking, parking_type, side, sample_config):
    """Ensure the correct car structure is returned."""
    result = base_parking.get_car_struct(parking_type, side)

    assert isinstance(result, np.ndarray)
    assert result.shape == (4, 2)  # Car structure is always (4, 2)

    # Get expected car structure
    expected_structure = None
    if parking_type == "parallel":
        expected_structure = base_parking.car_size.car_struct if side in [1, 2] else base_parking.car_size.car_struct_2
    elif parking_type == "perpendicular":
        expected_structure = base_parking.car_size.car_struct if side in [3, 4] else base_parking.car_size.car_struct_2

    # Compare NumPy arrays correctly
    assert np.array_equal(result, expected_structure), (
        f"Car structure mismatch for parking_type={parking_type}, side={side}"
    )


# ---------------------- Test `set_initial_car_loc` ----------------------
@pytest.mark.parametrize('side, parking_loc', [
    (1, np.array([15.0, 2.5])), (2, np.array([15.0, 27.5])),
    (3, np.array([2.5, 15.0])), (4, np.array([37.5, 15.0]))
])
def test_set_initial_car_loc(base_parking, side, parking_loc):
    """Check if `set_initial_car_loc` correctly positions the car."""

    initial_distance_range = (7.5, 15.0)
    car_loc_randomized_range = (-5.0, 5.0)

    car_loc = base_parking.set_initial_car_loc(side, parking_loc, initial_distance_range, car_loc_randomized_range)

    # Ensure the return type and shape
    assert isinstance(car_loc, np.ndarray)
    assert len(car_loc) == 2  # Shall return (x, y) coordinates

    # Validate x and y values depending on parking side
    if side == 1:
        x_car_loc_min = parking_loc[0] + car_loc_randomized_range[0]
        x_car_loc_max = parking_loc[0] + car_loc_randomized_range[1]
        y_car_loc_min = parking_loc[1] + initial_distance_range[0]
        y_car_loc_max = parking_loc[1] + initial_distance_range[1]

    elif side == 2:
        x_car_loc_min = parking_loc[0] + car_loc_randomized_range[0]
        x_car_loc_max = parking_loc[0] + car_loc_randomized_range[1]
        y_car_loc_min = parking_loc[1] - initial_distance_range[1]  # Negative since it's above
        y_car_loc_max = parking_loc[1] - initial_distance_range[0]

    elif side == 3:
        x_car_loc_min = parking_loc[0] + initial_distance_range[0]
        x_car_loc_max = parking_loc[0] + initial_distance_range[1]
        y_car_loc_min = parking_loc[1] + car_loc_randomized_range[0]
        y_car_loc_max = parking_loc[1] + car_loc_randomized_range[1]

    elif side == 4:
        x_car_loc_min = parking_loc[0] - initial_distance_range[1]  # Negative since it's to the right
        x_car_loc_max = parking_loc[0] - initial_distance_range[0]
        y_car_loc_min = parking_loc[1] + car_loc_randomized_range[0]
        y_car_loc_max = parking_loc[1] + car_loc_randomized_range[1]
    else:
        raise ValueError('side must be from 1 to 4.')

    # Assert car is within expected bounds
    assert x_car_loc_min <= car_loc[0] <= x_car_loc_max, (
        f"Car x-location {car_loc[0]} is out of expected range {x_car_loc_min} to {x_car_loc_max}"
    )
    assert y_car_loc_min <= car_loc[1] <= y_car_loc_max, (
        f"Car y-location {car_loc[1]} is out of expected range {y_car_loc_min} to {y_car_loc_max}"
    )


# ---------------------- Test `set_initial_parking_loc` ----------------------
@pytest.mark.parametrize('side', (1, 2, 3, 4))
def test_set_initial_parking_loc(base_parking, side):
    """Check that the parking location is within the expected boundaries."""
    window_w, window_h = 800, 600
    window_w_offset, window_h_offset = 100, 50

    parking_loc = base_parking.set_initial_parking_loc(side, window_w, window_h, window_w_offset, window_h_offset)

    # Ensure return type and shape
    assert isinstance(parking_loc, np.ndarray)
    assert len(parking_loc) == 2  # Shall return (x, y) coordinates

    # Validate x and y values depending on parking side
    if side in [1, 2]:  # Horizontal placement
        x_parking_min = window_w_offset * PIXEL_TO_METER_SCALE
        x_parking_max = (window_w - window_w_offset) * PIXEL_TO_METER_SCALE

        if side == 1:  # Bottom side
            y_parking = window_h_offset * PIXEL_TO_METER_SCALE  # Fixed y-position at bottom

        elif side == 2:  # Top side
            y_parking = (window_h - window_h_offset) * PIXEL_TO_METER_SCALE  # Fixed y-position at top

        assert x_parking_min <= parking_loc[0] <= x_parking_max, (
            f"Parking x-location {parking_loc[0]} is out of range {x_parking_min} to {x_parking_max}"
        )
        assert parking_loc[1] == y_parking, (
            f"Parking y-location {parking_loc[1]} should be fixed at {y_parking} for side {side}"
        )

    elif side in [3, 4]:  # Vertical placement
        y_parking_min = window_h_offset * PIXEL_TO_METER_SCALE
        y_parking_max = (window_h - window_h_offset) * PIXEL_TO_METER_SCALE

        if side == 3:  # Left side
            x_parking = window_h_offset * PIXEL_TO_METER_SCALE  # Fixed x-position at left

        elif side == 4:  # Right side
            x_parking = (window_w - window_w_offset) * PIXEL_TO_METER_SCALE  # Fixed x-position at right
        assert parking_loc[0] == x_parking, (
            f"Parking x-location {parking_loc[0]} should be fixed at {x_parking} for side {side}"
        )
        assert y_parking_min <= parking_loc[1] <= y_parking_max, (
            f"Parking y-location {parking_loc[1]} is out of range {y_parking_min} to {y_parking_max}"
        )

    else:
        raise ValueError('side must be from 1 to 4.')


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

    # Ensure each static car is a NumPy array with the correct shape
    for car in static_cars:
        assert isinstance(car, np.ndarray), "Each static car must be a NumPy array"
        assert car.shape == (4, 2), "Car structure must have shape (4,2)"

    # Ensure each static parking space is a NumPy array with the correct shape
    for parking in static_parking:
        assert isinstance(parking, np.ndarray), "Each static parking space must be a NumPy array"
        assert parking.shape == (4, 2), "Parking structure must have shape (4,2)"

    # Validate correct offsets
    offset = parallel_parking.parking_lot_size.offset_parallel
    if side in [1, 2]:  # Horizontal alignment
        expected_positions = [
            [parking_lot[0] + offset, parking_lot[1]],
            [parking_lot[0] - offset, parking_lot[1]],
        ]
    else:  # Vertical alignment
        expected_positions = [
            [parking_lot[0], parking_lot[1] + offset],
            [parking_lot[0], parking_lot[1] - offset],
        ]

    for loc, car in zip(expected_positions, static_cars):
        expected_vertices = parallel_parking.get_car_struct("parallel", side) + np.array(loc)
        assert np.array_equal(car, expected_vertices), (
            f"Static car at {loc} is not correctly placed for side={side}"
        )


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

    # Ensure each static car is a NumPy array with the correct shape
    for car in static_cars:
        assert isinstance(car, np.ndarray), "Each static car must be a NumPy array"
        assert car.shape == (4, 2), "Car structure must have shape (4,2)"

    # Ensure each static parking space is a NumPy array with the correct shape
    for parking in static_parking:
        assert isinstance(parking, np.ndarray), "Each static parking space must be a NumPy array"
        assert parking.shape == (4, 2), "Parking structure must have shape (4,2)"

    # Validate correct offsets
    offset = perpendicular_parking.parking_lot_size.offset_perpendicular
    if side in [1, 2]:  # Horizontal alignment
        expected_positions = [
            [parking_lot[0] + offset, parking_lot[1]],
            [parking_lot[0] - offset, parking_lot[1]],
        ]
    else:  # Vertical alignment
        expected_positions = [
            [parking_lot[0], parking_lot[1] + offset],
            [parking_lot[0], parking_lot[1] - offset],
        ]

    for loc, car in zip(expected_positions, static_cars):
        expected_vertices = perpendicular_parking.get_car_struct("perpendicular", side) + np.array(loc)
        assert np.array_equal(car, expected_vertices), (
            f"Static car at {loc} is not correctly placed for side={side}"
        )
