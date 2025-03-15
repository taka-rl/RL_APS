import pytest
from sim_env.parameters import Config, RenderConfig, WheelSize
from sim_env.parking import BaseParking, ParallelParking, PerpendicularParking
from sim_env.renderer import Renderer


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


@pytest.fixture
def sample_config():
    """Returns a default Config object for testing."""
    return Config()


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

