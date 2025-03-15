import pytest
import pygame
import numpy as np
from sim_env.renderer import Renderer, meters_to_pixels
from sim_env.parameters import RenderConfig, WheelSize, PI, Config
from sim_env.car import Car
from sim_env.parking import BaseParking, ParallelParking, PerpendicularParking


@pytest.fixture
def render_config():
    """Fixture to create a sample RenderConfig."""
    return RenderConfig()


@pytest.fixture
def wheel_size():
    """Fixture to create a sample WheelSize."""
    return WheelSize()


@pytest.fixture
def renderer(render_config, wheel_size):
    """Fixture to create an instance of Renderer."""
    return Renderer(render_config, wheel_size)


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


def test_initialize_window(renderer):
    """Test if the window initializes correctly."""
    renderer.initialize_window()
    assert renderer.window is not None
    assert isinstance(renderer.window, pygame.Surface)
    assert renderer.clock is not None
    assert isinstance(renderer.clock, pygame.time.Clock)
    assert renderer.font is not None
    assert isinstance(renderer.font, pygame.font.Font)


def test_meters_to_pixels():
    """Test meters to pixels conversion function."""
    meters = 10.0
    expected_pixels = meters / 0.05  # PIXEL_TO_METER_SCALE
    assert np.allclose(meters_to_pixels(meters), expected_pixels, atol=1e-5)


@pytest.mark.parametrize('parking_type', ['perpendicular', 'parallel'])
@pytest.mark.parametrize('side', (1, 2, 3, 4))
def test_render(renderer, side, parking_type, base_parking, parallel_parking, perpendicular_parking):
    """Test the main render function."""
    renderer.initialize_window()

    # Prepare for static objects
    window_w, window_h = 800, 600
    window_w_offset, window_h_offset = 50, 50

    if parking_type == 'perpendicular':
        parking_strategy = perpendicular_parking
    if parking_type == 'parallel':
        parking_strategy = parallel_parking

    parking_lot = base_parking.set_initial_parking_loc(side, window_w, window_h, window_w_offset, window_h_offset)
    parking_lot_vertices = (parking_lot + parking_strategy.get_parking_struct(parking_type, side))
    static_cars_vertices, static_parking_lot_vertices = parking_strategy.generate_static_obstacles(parking_lot, side)

    # Draw static objects
    renderer.draw_static_elements(parking_lot_vertices, static_parking_lot_vertices, static_cars_vertices)

    # Prepare for dynamic objects
    initial_distance_range = (7.5, 15.0)
    car_loc_randomized_range = (-5.0, 5.0)
    car_loc_old = np.array([20, 20])
    car_loc = base_parking.set_initial_car_loc(side, parking_lot, initial_distance_range, car_loc_randomized_range)
    car = Car(car_loc, base_parking.set_initial_heading(parking_type, side), Config())
    car.v = 2.0

    # Draw dynamic objects
    renderer.render(car, car_loc_old=car_loc_old)

    # Ensure the text surface is updated
    assert renderer.window is not None
    assert isinstance(renderer.window, pygame.Surface), "window shall be a pygame.Surface"
    assert renderer.surf_parkinglot is not None
    assert isinstance(renderer.surf_parkinglot, pygame.Surface), "surf_parkinglot shall be a pygame.Surface"
    assert renderer.surf_car is not None
    assert isinstance(renderer.surf_car, pygame.Surface), "surf_car shall be a pygame.Surface"
    assert renderer.surf_text is not None
    assert isinstance(renderer.surf_text, pygame.Surface), "surf_text shall be a pygame.Surface"
