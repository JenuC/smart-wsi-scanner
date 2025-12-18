"""Tests for the configuration system."""

import pytest
from pathlib import Path
from smart_wsi_scanner.config import (
    sp_position,
    ConfigManager,
    sp_microscope_settings,
    sp_stage_settings,
    sp_objective_lens,
    sp_detector,
    sp_imaging_mode,
    sp_microscope,
    _limits
)


def test_position_creation():
    """Test creating a position object."""
    pos = sp_position(x=100.0, y=200.0, z=300.0)
    assert pos.x == 100.0
    assert pos.y == 200.0
    assert pos.z == 300.0


def test_position_with_none_values():
    """Test creating a position with None values."""
    pos = sp_position(x=100.0, y=None, z=300.0)
    assert pos.x == 100.0
    assert pos.y is None
    assert pos.z == 300.0


def test_position_populate_missing():
    """Test populating missing coordinates."""
    pos1 = sp_position(x=100.0, y=None, z=None)
    pos2 = sp_position(x=200.0, y=300.0, z=400.0)
    
    pos1.populate_missing(pos2)
    assert pos1.x == 100.0  # Should keep original value
    assert pos1.y == 300.0  # Should populate from pos2
    assert pos1.z == 400.0  # Should populate from pos2


def test_position_repr():
    """Test string representation of position."""
    pos = sp_position(x=100.0, y=200.0, z=300.0)
    repr_str = repr(pos)
    assert "x=100.0" in repr_str
    assert "y=200.0" in repr_str
    assert "z=300.0" in repr_str
    assert "sp_position" in repr_str


def test_limits_creation():
    """Test creating limits object."""
    limits = _limits(low=0.0, high=100.0)
    assert limits.low == 0.0
    assert limits.high == 100.0


def test_limits_auto_swap():
    """Test that limits automatically swap if low > high."""
    limits = _limits(low=100.0, high=0.0)
    assert limits.low == 0.0
    assert limits.high == 100.0


def test_stage_settings():
    """Test stage settings creation."""
    x_limit = _limits(low=0.0, high=1000.0)
    y_limit = _limits(low=0.0, high=1000.0)
    z_limit = _limits(low=0.0, high=500.0)
    
    stage = sp_stage_settings(
        x_limit=x_limit,
        y_limit=y_limit,
        z_limit=z_limit
    )
    
    assert stage.x_limit is not None
    assert stage.x_limit.low == 0.0
    assert stage.x_limit.high == 1000.0
    assert stage.y_limit is not None
    assert stage.y_limit.low == 0.0
    assert stage.y_limit.high == 1000.0
    assert stage.z_limit is not None
    assert stage.z_limit.low == 0.0
    assert stage.z_limit.high == 500.0


def test_objective_lens():
    """Test objective lens creation."""
    lens = sp_objective_lens(
        name="20X",
        magnification=20.0,
        NA=0.75,
        WD=1.0
    )
    
    assert lens.name == "20X"
    assert lens.magnification == 20.0
    assert lens.NA == 0.75
    assert lens.WD == 1.0


def test_detector():
    """Test detector creation."""
    detector = sp_detector(width=1920, height=1080)
    assert detector.width == 1920
    assert detector.height == 1080


def test_imaging_mode():
    """Test imaging mode creation."""
    mode = sp_imaging_mode(
        name="High Resolution",
        pixel_size=0.5
    )
    assert mode.name == "High Resolution"
    assert mode.pixel_size == 0.5


def test_microscope():
    """Test microscope creation."""
    microscope = sp_microscope(
        name="PPM",
        type="Polarizing"
    )
    assert microscope.name == "PPM"
    assert microscope.type == "Polarizing"


def test_microscope_settings():
    """Test complete microscope settings."""
    microscope = sp_microscope(name="Test", type="Standard")
    stage = sp_stage_settings()
    lens = sp_objective_lens(name="10X", magnification=10.0, NA=0.3)
    detector = sp_detector(width=1024, height=768)
    mode = sp_imaging_mode(name="Standard", pixel_size=1.0)
    
    settings = sp_microscope_settings(
        path="/test/path",
        microscope=microscope,
        stage=stage,
        lens=lens,
        detector=detector,
        imaging_mode=mode
    )
    
    assert settings.path == "/test/path"
    assert settings.microscope is not None
    assert settings.microscope.name == "Test"
    assert settings.lens is not None
    assert settings.lens.magnification == 10.0
    assert settings.detector is not None
    assert settings.detector.width == 1024
    assert settings.imaging_mode is not None
    assert settings.imaging_mode.pixel_size == 1.0


def test_config_manager_initialization():
    """Test configuration manager initialization."""
    # Create a temporary config directory for testing
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = ConfigManager(config_dir=temp_dir)
        assert isinstance(manager, ConfigManager)
        assert hasattr(manager, 'load_config')
        assert hasattr(manager, 'save_config')
        assert hasattr(manager, 'get_config')
        assert hasattr(manager, 'list_configs')
        assert manager.config_dir == Path(temp_dir)


def test_config_manager_default_dir():
    """Test that ConfigManager uses default directory when none specified."""
    manager = ConfigManager()
    assert manager.config_dir.name == "configurations"
    assert manager.config_dir.parent.name == "smart_wsi_scanner"


def test_config_manager_list_configs():
    """Test listing configurations."""
    import tempfile
    import yaml
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create some test config files
        test_config = {
            "microscope": {"name": "Test", "type": "Standard"},
            "stage": {"x_limit": {"low": 0, "high": 1000}}
        }
        
        config_path = Path(temp_dir) / "test_config.yml"
        with open(config_path, "w") as f:
            yaml.dump(test_config, f)
        
        manager = ConfigManager(config_dir=temp_dir)
        configs = manager.list_configs()
        assert "test_config" in configs