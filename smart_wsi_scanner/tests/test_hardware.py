"""Tests for the hardware abstraction layer."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from ..hardware import PycromanagerHardware, MicroscopeHardware
from ..config import sp_position, sp_microscope_settings

@pytest.fixture
def mock_core():
    """Create a mock Core object."""
    core = MagicMock()
    core.get_x_position.return_value = 100.0
    core.get_y_position.return_value = 200.0
    core.get_position.return_value = 300.0
    core.get_focus_device.return_value = "ZStage"
    core.get_xy_stage_device.return_value = "XYStage"
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

@pytest.fixture
def mock_studio():
    """Create a mock Studio object."""
    studio = MagicMock()
    studio.live.return_value = MagicMock()
    return studio

@patch('smart_wsi_scanner.hardware.Studio')
def test_hardware_initialization(mock_studio_class, mock_core, mock_settings, mock_studio):
    """Test hardware initialization."""
    mock_studio_class.return_value = mock_studio
    hardware = PycromanagerHardware(mock_core, mock_settings)
    assert isinstance(hardware, MicroscopeHardware)
    assert hardware.core == mock_core
    assert hardware.settings == mock_settings
    assert hardware.studio == mock_studio

@patch('smart_wsi_scanner.hardware.Studio')
def test_get_current_position(mock_studio_class, mock_core, mock_settings, mock_studio):
    """Test getting current position."""
    mock_studio_class.return_value = mock_studio
    hardware = PycromanagerHardware(mock_core, mock_settings)
    position = hardware.get_current_position()
    assert isinstance(position, sp_position)
    assert position.x == 100.0
    assert position.y == 200.0
    assert position.z == 300.0

@patch('smart_wsi_scanner.hardware.Studio')
def test_move_to_position(mock_studio_class, mock_core, mock_settings, mock_studio):
    """Test moving to a position."""
    mock_studio_class.return_value = mock_studio
    hardware = PycromanagerHardware(mock_core, mock_settings)
    position = sp_position(x=500.0, y=500.0, z=500.0)
    hardware.move_to_position(position)
    
    mock_core.set_position.assert_called_once_with(500.0)
    mock_core.set_xy_position.assert_called_once_with(500.0, 500.0)
    mock_core.wait_for_device.assert_any_call("XYStage")
    mock_core.wait_for_device.assert_any_call("ZStage")

@patch('smart_wsi_scanner.hardware.Studio')
def test_move_to_position_out_of_range(mock_studio_class, mock_core, mock_settings, mock_studio):
    """Test moving to an out-of-range position."""
    mock_studio_class.return_value = mock_studio
    hardware = PycromanagerHardware(mock_core, mock_settings)
    position = sp_position(x=2000.0, y=2000.0, z=2000.0)
    
    with pytest.raises(ValueError, match="Position out of range"):
        hardware.move_to_position(position)

@patch('smart_wsi_scanner.hardware.Studio')
def test_snap_image(mock_studio_class, mock_core, mock_settings, mock_studio):
    """Test capturing an image."""
    mock_studio_class.return_value = mock_studio
    hardware = PycromanagerHardware(mock_core, mock_settings)
    
    # Mock the tagged image
    mock_tagged_image = MagicMock()
    mock_tagged_image.pix = np.zeros((100, 100), dtype=np.uint8)
    mock_tagged_image.tags = {"Width": 100, "Height": 100}
    mock_core.get_tagged_image.return_value = mock_tagged_image
    
    image, metadata = hardware.snap_image()
    assert isinstance(image, np.ndarray)
    assert isinstance(metadata, dict)
    assert "Width" in metadata
    assert "Height" in metadata

@patch('smart_wsi_scanner.hardware.Studio')
def test_set_objective(mock_studio_class, mock_core, mock_settings, mock_studio):
    """Test changing objective lens."""
    mock_studio_class.return_value = mock_studio
    hardware = PycromanagerHardware(mock_core, mock_settings)
    mock_core.get_property.return_value = "4X"
    
    hardware.set_objective("20X")
    mock_core.set_property.assert_called_with("Objective", "Position", "20X")
    mock_core.wait_for_device.assert_called()

@patch('smart_wsi_scanner.hardware.Studio')
def test_autofocus(mock_studio_class, mock_core, mock_settings, mock_studio):
    """Test autofocus functionality."""
    mock_studio_class.return_value = mock_studio
    hardware = PycromanagerHardware(mock_core, mock_settings)
    
    with patch('smart_wsi_scanner.smartpath.smartpath') as mock_smartpath:
        mock_smartpath.autofocus.return_value = 500.0
        result = hardware.autofocus()
        assert result == 500.0
        mock_smartpath.autofocus.assert_called_once_with(mock_core, mock_settings) 