"""Tests for the smartpath module."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from ..smartpath import smartpath
from ..config import sp_position, sp_microscope_settings, sp_imaging_mode

@pytest.fixture
def mock_core():
    """Create a mock Core object."""
    core = MagicMock()
    core.get_x_position.return_value = 100.0
    core.get_y_position.return_value = 200.0
    core.get_position.return_value = 300.0
    core.get_focus_device.return_value = "ZStage"
    core.get_xy_stage_device.return_value = "XYStage"
    core.get_loaded_devices.return_value = MagicMock(size=lambda: 2)
    core.get_device_property_names.return_value = MagicMock(size=lambda: 2)
    core.get_property.return_value = "4X"
    return core

@pytest.fixture
def mock_settings():
    """Create mock microscope settings."""
    settings = MagicMock(spec=sp_microscope_settings)
    settings.stage.xlimit.low = 0
    settings.stage.xlimit.high = 1000
    settings.stage.ylimit.low = 0
    settings.stage.ylimit.high = 1000
    settings.stage.zlimit.low = 0
    settings.stage.zlimit.high = 1000
    settings.stage.z_stage = "ZStage"
    settings.focus_device = "Focus"
    settings.obj_slider = ("Objective", "Position")
    return settings

def test_get_current_position(mock_core):
    """Test getting current position."""
    position = smartpath.get_current_position(mock_core)
    assert isinstance(position, sp_position)
    assert position.x == 100.0
    assert position.y == 200.0
    assert position.z == 300.0

def test_is_coordinate_in_range(mock_settings):
    """Test coordinate range checking."""
    # Test valid position
    position = sp_position(x=500.0, y=500.0, z=500.0)
    assert smartpath.is_coordinate_in_range(mock_settings, position) is True
    
    # Test out of range position
    position = sp_position(x=2000.0, y=2000.0, z=2000.0)
    assert smartpath.is_coordinate_in_range(mock_settings, position) is False

def test_move_to_position(mock_core, mock_settings):
    """Test moving to a position."""
    position = sp_position(x=500.0, y=500.0, z=500.0)
    smartpath.move_to_position(mock_core, position, mock_settings)
    
    mock_core.set_position.assert_called_once_with(500.0)
    mock_core.set_xy_position.assert_called_once_with(500.0, 500.0)
    mock_core.wait_for_device.assert_any_call("XYStage")
    mock_core.wait_for_device.assert_any_call("ZStage")

def test_get_device_properties(mock_core):
    """Test getting device properties."""
    properties = smartpath.get_device_properties(mock_core)
    assert isinstance(properties, dict)
    mock_core.get_loaded_devices.assert_called_once()
    mock_core.get_device_property_names.assert_called()

def test_compare_dev_prop():
    """Test device property comparison."""
    dp1 = {"device1": {"prop1": "value1"}}
    dp2 = {"device1": {"prop1": "value2"}}
    
    result = smartpath.compare_dev_prop(dp1, dp2)
    assert isinstance(result, str)
    assert "device1" in result
    assert "value1" in result
    assert "value2" in result

def test_swap_objective_lens(mock_core, mock_settings):
    """Test objective lens swapping."""
    desired_mode = MagicMock(spec=sp_imaging_mode)
    desired_mode.objective_position_label = "20X"
    desired_mode.z = 500.0
    desired_mode.f = 100.0
    
    smartpath.swap_objective_lens(mock_core, mock_settings, desired_mode)
    mock_core.set_property.assert_called_with("Objective", "Position", "20X")
    mock_core.wait_for_device.assert_called()

@pytest.mark.parametrize("scope", ["used", "allowed"])
def test_get_device_properties_scopes(mock_core, scope):
    """Test getting device properties with different scopes."""
    properties = smartpath.get_device_properties(mock_core, scope=scope)
    assert isinstance(properties, dict)
    
    if scope == "used":
        mock_core.get_property.assert_called()
    else:
        mock_core.get_allowed_property_values.assert_called() 