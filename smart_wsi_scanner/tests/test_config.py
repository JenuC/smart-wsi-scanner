"""Tests for the configuration system."""

import pytest
from ..config import sp_position, sp_microscope_settings, ConfigManager

def test_position_creation():
    """Test creating a position object."""
    pos = sp_position(x=100.0, y=200.0, z=300.0)
    assert pos.x == 100.0
    assert pos.y == 200.0
    assert pos.z == 300.0

def test_position_repr():
    """Test string representation of position."""
    pos = sp_position(x=100.0, y=200.0, z=300.0)
    assert "x=100.0" in repr(pos)
    assert "y=200.0" in repr(pos)
    assert "z=300.0" in repr(pos)

def test_config_manager():
    """Test configuration manager functionality."""
    manager = ConfigManager()
    assert isinstance(manager, ConfigManager)
    assert hasattr(manager, 'load_config')
    assert hasattr(manager, 'save_config')
    assert hasattr(manager, 'get_config')
    assert hasattr(manager, 'list_configs') 