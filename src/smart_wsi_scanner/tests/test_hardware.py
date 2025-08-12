"""Tests for the hardware abstraction layer."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from ..hardware import MicroscopeHardware, is_mm_running, is_coordinate_in_range
from ..hardware_pycromanager import PycromanagerHardware, init_pycromanager, obj_2_list
from ..config import sp_position, sp_microscope_settings, sp_imaging_mode, _limits, sp_stage_settings, sp_microscope


@pytest.fixture
def mock_core():
    """Create a mock Core object."""
    core = MagicMock()
    core.get_x_position.return_value = 100.0
    core.get_y_position.return_value = 200.0
    core.get_position.return_value = 300.0
    core.get_focus_device.return_value = "ZStage"
    core.get_xy_stage_device.return_value = "XYStage"
    core.get_pixel_size_um.return_value = 1.0
    core.is_sequence_running.return_value = False
    return core


@pytest.fixture
def mock_settings():
    """Create mock microscope settings."""
    settings = MagicMock(spec=sp_microscope_settings)
    
    # Create proper stage settings with limits
    stage = MagicMock()
    stage.x_limit = _limits(low=0, high=1000)
    stage.y_limit = _limits(low=0, high=1000)
    stage.z_limit = _limits(low=0, high=1000)
    stage.z_stage = "ZStage"
    stage.f_stage = "FStage"
    stage.r_stage = "RStage"
    settings.stage = stage
    
    # Create microscope info
    microscope = sp_microscope(name="Test", type="Standard")
    settings.microscope = microscope
    
    # Add other required attributes
    settings.focus_device = "Focus"
    settings.obj_slider = ("Objective", "Position")
    
    return settings


@pytest.fixture
def mock_studio():
    """Create a mock Studio object."""
    studio = MagicMock()
    live_mock = MagicMock()
    studio.live.return_value = live_mock
    return studio


@pytest.fixture
def hardware(mock_core, mock_studio, mock_settings):
    """Create PycromanagerHardware instance with mocked dependencies."""
    return PycromanagerHardware(mock_core, mock_studio, mock_settings)


class TestMicroscopeHardware:
    """Test the abstract MicroscopeHardware class."""
    
    def test_abstract_methods(self):
        """Test that MicroscopeHardware cannot be instantiated directly."""
        with pytest.raises(TypeError):
            MicroscopeHardware()


class TestHardwareUtilities:
    """Test hardware utility functions."""
    
    @patch('psutil.process_iter')
    @patch('platform.system')
    def test_is_mm_running_windows(self, mock_system, mock_process_iter):
        """Test is_mm_running on Windows when MM is running."""
        mock_system.return_value = "Windows"
        
        mock_proc = MagicMock()
        mock_proc.exe.return_value = "C:\\Program Files\\Micro-Manager\\ImageJ.exe"
        mock_process_iter.return_value = [mock_proc]
        
        assert is_mm_running() is True
    
    @patch('platform.system')
    def test_is_mm_running_non_windows(self, mock_system):
        """Test is_mm_running on non-Windows systems."""
        mock_system.return_value = "Linux"
        assert is_mm_running() is False
    
    def test_is_coordinate_in_range_valid(self, mock_settings):
        """Test coordinate range checking with valid position."""
        position = sp_position(x=500.0, y=500.0, z=500.0)
        assert is_coordinate_in_range(mock_settings, position) is True
    
    def test_is_coordinate_in_range_invalid_x(self, mock_settings):
        """Test coordinate range checking with out-of-range X."""
        position = sp_position(x=2000.0, y=500.0, z=500.0)
        with pytest.warns(UserWarning):
            assert is_coordinate_in_range(mock_settings, position) is False
    
    def test_is_coordinate_in_range_invalid_y(self, mock_settings):
        """Test coordinate range checking with out-of-range Y."""
        position = sp_position(x=500.0, y=2000.0, z=500.0)
        with pytest.warns(UserWarning):
            assert is_coordinate_in_range(mock_settings, position) is False
    
    def test_is_coordinate_in_range_invalid_z(self, mock_settings):
        """Test coordinate range checking with out-of-range Z."""
        position = sp_position(x=500.0, y=500.0, z=2000.0)
        with pytest.warns(UserWarning):
            assert is_coordinate_in_range(mock_settings, position) is False
    
    def test_is_coordinate_in_range_no_z(self, mock_settings):
        """Test coordinate range checking without Z coordinate."""
        position = sp_position(x=500.0, y=500.0, z=None)
        assert is_coordinate_in_range(mock_settings, position) is True


class TestPycromanagerHardware:
    """Test PycromanagerHardware implementation."""
    
    def test_initialization(self, mock_core, mock_studio, mock_settings):
        """Test hardware initialization."""
        hardware = PycromanagerHardware(mock_core, mock_studio, mock_settings)
        assert isinstance(hardware, MicroscopeHardware)
        assert hardware.core == mock_core
        assert hardware.settings == mock_settings
        assert hardware.studio == mock_studio
    
    def test_get_current_position(self, hardware):
        """Test getting current position."""
        position = hardware.get_current_position()
        assert isinstance(position, sp_position)
        assert position.x == 100.0
        assert position.y == 200.0
        assert position.z == 300.0
    
    def test_move_to_position(self, hardware, mock_core):
        """Test moving to a position."""
        position = sp_position(x=500.0, y=500.0, z=500.0)
        hardware.move_to_position(position)
        
        mock_core.set_position.assert_called_once_with(500.0)
        mock_core.set_xy_position.assert_called_once_with(500.0, 500.0)
        mock_core.wait_for_device.assert_any_call("XYStage")
        mock_core.wait_for_device.assert_any_call("ZStage")  # Changed from "Focus" to "ZStage"
    
    def test_move_to_position_with_missing_coords(self, hardware, mock_core):
        """Test moving to a position with missing coordinates."""
        position = sp_position(x=500.0, y=None, z=None)
        hardware.move_to_position(position)
        
        # Should populate missing coordinates from current position
        mock_core.set_position.assert_called_once_with(300.0)  # Current Z
        mock_core.set_xy_position.assert_called_once_with(500.0, 200.0)  # X=500, Y=current
    
    def test_move_to_position_out_of_range(self, hardware):
        """Test moving to an out-of-range position."""
        position = sp_position(x=2000.0, y=2000.0, z=2000.0)
        
        with pytest.raises(ValueError, match="Position out of range"):
            hardware.move_to_position(position)
    
    def test_snap_image(self, hardware, mock_core):
        """Test capturing an image."""
        # Mock the tagged image - pix should be a flat array
        mock_tagged_image = MagicMock()
        mock_tagged_image.pix = np.zeros((100 * 100,), dtype=np.uint8)  # Flat array
        mock_tagged_image.tags = {"Width": 100, "Height": 100, "Core-Camera": "TestCamera"}
        mock_core.get_tagged_image.return_value = mock_tagged_image
        
        # Mock device properties
        hardware.get_device_properties = MagicMock(return_value={
            "Core": {"Camera": "TestCamera"},
            "TestCamera": {"Color": "OFF"}
        })
        
        # Test should return None for unknown camera
        image, metadata = hardware.snap()
        assert image is None
        assert metadata is None
    
    def test_snap_image_color_camera(self, hardware, mock_core):
        """Test capturing an image with color camera."""
        # Mock the tagged image with 4 channels (BGRA)
        mock_tagged_image = MagicMock()
        mock_tagged_image.pix = np.zeros((100*100*4,), dtype=np.uint8)
        mock_tagged_image.tags = {"Width": 100, "Height": 100}
        mock_core.get_tagged_image.return_value = mock_tagged_image
        
        # Mock device properties for QCamera
        hardware.get_device_properties = MagicMock(return_value={
            "Core": {"Camera": "QCamera"},
            "QCamera": {"Color": "ON"}
        })
        
        image, metadata = hardware.snap()
        assert isinstance(image, np.ndarray)
        assert image.shape == (100, 100, 4)  # ARGB after flipping
    
    def test_get_fov(self, hardware, mock_core):
        """Test getting field of view."""
        # First call returns camera name, second returns resolution as int
        mock_core.get_property.side_effect = ["QCamera", 100]  # Return int, not string
        mock_core.get_pixel_size_um.return_value = 2.0
        
        fov_x, fov_y = hardware.get_fov()
        assert fov_x == 200.0  # 100 * 2.0
        assert fov_y == 200.0  # 100 * 2.0
    
    def test_autofocus(self, hardware, mock_core):
        """Test autofocus functionality."""
        # Mock snap method
        mock_image = np.random.rand(100, 100, 3) * 255
        hardware.snap = MagicMock(return_value=(mock_image.astype(np.uint8), {}))
        
        # Run autofocus with minimal steps
        result = hardware.autofocus(n_steps=3, search_range=10)
        
        # Check that it moved the stage and returned a Z position
        assert isinstance(result, float)
        assert mock_core.set_position.call_count >= 3  # At least n_steps movements
    
    def test_white_balance(self, hardware):
        """Test white balance functionality."""
        # Create test image
        test_image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        
        # Mock settings with white balance profile
        hardware.settings.white_balance = MagicMock()
        hardware.settings.white_balance.default = MagicMock()
        hardware.settings.white_balance.default.default = (1.0, 1.0, 1.0)
        
        # Test white balance
        result = hardware.white_balance(img=test_image, gain=1.0)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.uint8
        assert result.shape == test_image.shape
    
    def test_get_device_properties(self, hardware, mock_core):
        """Test getting device properties."""
        # Mock device list
        mock_devices = MagicMock()
        mock_devices.size.return_value = 2
        mock_devices.get.side_effect = ["Device1", "Device2"]
        mock_core.get_loaded_devices.return_value = mock_devices
        
        # Mock property names
        mock_props = MagicMock()
        mock_props.size.return_value = 1
        mock_props.get.return_value = "Property1"
        mock_core.get_device_property_names.return_value = mock_props
        
        # Mock property values
        mock_core.get_property.side_effect = ["Value1", "Value2"]
        
        properties = hardware.get_device_properties()
        assert isinstance(properties, dict)
        assert "Device1" in properties
        assert "Device2" in properties
        assert properties["Device1"]["Property1"] == "Value1"


class TestPycromanagerUtilities:
    """Test pycromanager utility functions."""
    
    def test_obj_2_list(self):
        """Test obj_2_list conversion."""
        mock_obj = MagicMock()
        mock_obj.size.return_value = 3
        mock_obj.get.side_effect = ["a", "b", "c"]
        
        result = obj_2_list(mock_obj)
        assert result == ["a", "b", "c"]
    
    @patch('src.smart_wsi_scanner.hardware_pycromanager.is_mm_running')
    @patch('src.smart_wsi_scanner.hardware_pycromanager.Core')
    @patch('src.smart_wsi_scanner.hardware_pycromanager.Studio')
    def test_init_pycromanager_success(self, mock_studio_class, mock_core_class, mock_is_running):
        """Test successful pycromanager initialization."""
        mock_is_running.return_value = True
        mock_core = MagicMock()
        mock_studio = MagicMock()
        mock_core_class.return_value = mock_core
        mock_studio_class.return_value = mock_studio
        
        core, studio = init_pycromanager()
        assert core == mock_core
        assert studio == mock_studio
        mock_core.set_timeout_ms.assert_called_once_with(20000)
    
    @patch('src.smart_wsi_scanner.hardware_pycromanager.is_mm_running')
    def test_init_pycromanager_mm_not_running(self, mock_is_running):
        """Test pycromanager initialization when MM is not running."""
        mock_is_running.return_value = False
        
        core, studio = init_pycromanager()
        assert core is None
        assert studio is None


class TestMicroscopeSpecificFeatures:
    """Test microscope-specific features."""
    
    def test_ppm_initialization(self, mock_core, mock_studio):
        """Test PPM-specific initialization."""
        settings = MagicMock(spec=sp_microscope_settings)
        settings.microscope = sp_microscope(name="PPM", type="Polarizing")
        settings.stage = MagicMock()
        settings.stage.r_stage = "RotationStage"
        
        hardware = PycromanagerHardware(mock_core, mock_studio, settings)
        assert hasattr(hardware, 'set_psg_ticks')
        assert hasattr(hardware, 'get_psg_ticks')
    
    def test_camm_initialization(self, mock_core, mock_studio):
        """Test CAMM-specific initialization."""
        settings = MagicMock(spec=sp_microscope_settings)
        settings.microscope = sp_microscope(name="CAMM", type="Standard")
        
        hardware = PycromanagerHardware(mock_core, mock_studio, settings)
        assert hasattr(hardware, 'swap_objective_lens')
    
    def test_camm_swap_objective_4x_to_20x(self, mock_core, mock_studio):
        """Test CAMM objective swapping from 4X to 20X."""
        settings = MagicMock(spec=sp_microscope_settings)
        settings.microscope = sp_microscope(name="CAMM", type="Standard")
        settings.obj_slider = ("Objective", "Position")
        settings.stage = MagicMock()
        settings.stage.z_stage = "ZStage"
        settings.stage.f_stage = "FStage"
        
        hardware = PycromanagerHardware(mock_core, mock_studio, settings)
        
        # Mock current objective
        mock_core.get_property.return_value = "4X"
        
        # Create desired imaging mode
        desired_mode = MagicMock(spec=sp_imaging_mode)
        desired_mode.name = "20X_Mode"
        desired_mode.objective_position_label = "20X"
        desired_mode.z = 500.0
        desired_mode.f = 100.0
        
        hardware.swap_objective_lens(desired_mode)
        
        # Verify the sequence of operations
        mock_core.set_property.assert_called_with("Objective", "Position", "20X")
        mock_core.wait_for_device.assert_called()
        mock_core.set_position.assert_called()