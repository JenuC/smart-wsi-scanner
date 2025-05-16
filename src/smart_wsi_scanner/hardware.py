"""Hardware abstraction layer for microscope control."""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional
import numpy as np
from collections import OrderedDict
import warnings

from .config import sp_microscope_settings, sp_position, sp_imaging_mode

# Try imports in order of preference
try:
    from pycromanager import Core, Studio
    PYCOMGR_AVAILABLE = True
except ImportError:
    PYCOMGR_AVAILABLE = False
    Core = None
    Studio = None

try:
    from pymmcore_plus import CMMCorePlus
    CMMCORE_PLUS_AVAILABLE = True
except ImportError:
    CMMCORE_PLUS_AVAILABLE = False
    CMMCorePlus = None

try:
    import pymmcore
    PYMMCORE_AVAILABLE = True
except ImportError:
    PYMMCORE_AVAILABLE = False
    pymmcore = None

class MicroscopeHardware(ABC):
    """Abstract base class for microscope hardware control."""
    
    @abstractmethod
    def move_to_position(self, position: sp_position) -> None:
        """Move stage to specified position."""
        pass
    
    @abstractmethod
    def get_current_position(self) -> sp_position:
        """Get current stage position."""
        pass
    
    @abstractmethod
    def snap_image(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Capture image and return image data with metadata."""
        pass
    
    @abstractmethod
    def set_objective(self, objective_name: str) -> None:
        """Change objective lens."""
        pass
    
    @abstractmethod
    def autofocus(self, **kwargs) -> float:
        """Perform autofocus and return best focus position."""
        pass

class PycromanagerHardware(MicroscopeHardware):
    """Implementation for Pycromanager-based microscopes."""
    
    def __init__(self, core: Core, settings: sp_microscope_settings, studio: Studio = None):
        if not PYCOMGR_AVAILABLE:
            raise ImportError("pycromanager is not installed")
        self.core = core
        self.settings = settings
        self.studio = studio or Studio()
        
    def move_to_position(self, position: sp_position) -> None:
        # Get current position and populate any missing coordinates
        current_position = self.get_current_position()
        position.populate_missing(current_position)
        
        if not self._is_coordinate_in_range(position):
            raise ValueError("Position out of range")
            
        if self.core.get_focus_device() != self.settings.stage.z_stage:
            self.core.set_focus_device(self.settings.focus_device)
            
        self.core.set_position(position.z)
        self.core.set_xy_position(position.x, position.y)
        self.core.wait_for_device(self.core.get_xy_stage_device())
        self.core.wait_for_device(self.core.get_focus_device())
        
    def get_current_position(self) -> sp_position:
        return sp_position(
            self.core.get_x_position(),
            self.core.get_y_position(), 
            self.core.get_position()
        )
        
    def snap_image(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        if self.core.is_sequence_running():
            self.studio.live().set_live_mode(False)
        self.core.snap_image()
        tagged_image = self.core.get_tagged_image()
        tags = OrderedDict(sorted(tagged_image.tags.items()))
        
        color_camera = False
        # Handle QCamera specific processing
        if self.core.get_property("Core", "Camera") == "QCamera":
            if self.core.get_property("QCamera", "Color") == "ON":
                color_camera = True
        if self.core.get_property("Core", "Camera") == "MicroPublisher6":
            if self.core.get_property("MicroPublisher6", "Color") == "ON":
                color_camera = True
        if color_camera:
            newshape=[tags["Height"], tags["Width"], 4]
            pixels = tagged_image.pix.reshape(newshape)
            pixels = pixels[:, :, 0:3]  # Remove alpha
            pixels = np.flip(pixels, 2)  # Flip channels
            return pixels, tags
        else:
            newshape=[tags["Height"], tags["Width"]]
            pixels = tagged_image.pix.reshape(newshape)
            return pixels, tags
        
    def set_objective(self, objective_name: str) -> None:
        current_slider_position = self.core.get_property(*self.settings.obj_slider)
        if objective_name != current_slider_position:
            if objective_name.startswith("4X"):
                self.core.set_focus_device(self.settings.stage.z_stage)
                self.core.set_position(self.settings.imaging_mode.z)
                self.core.wait_for_device(self.settings.stage.z_stage)
                self.core.set_property(*self.settings.obj_slider, objective_name)
                self.core.set_focus_device(self.settings.stage.f_stage)
                self.core.set_position(self.settings.imaging_mode.f)
                self.core.wait_for_system()
            elif objective_name.startswith("20X"):
                self.core.set_property(*self.settings.obj_slider, objective_name)
                self.core.wait_for_device(self.settings.obj_slider[0])
                self.core.set_focus_device(self.settings.stage.z_stage)
                self.core.set_position(self.settings.imaging_mode.z)
                self.core.set_focus_device(self.settings.stage.f_stage)
                self.core.set_position(self.settings.imaging_mode.f)
                self.core.wait_for_system()
                
            self.core.set_focus_device(self.settings.stage.z_stage)
            
    def autofocus(self, **kwargs) -> float:
        from .smartpath import smartpath
        return smartpath.autofocus(self.core, self.settings, **kwargs)
        
    def _is_coordinate_in_range(self, position: sp_position) -> bool:
        from .smartpath import smartpath
        return smartpath.is_coordinate_in_range(self.settings, position)

class PymmcoreplusHardware(MicroscopeHardware):
    """Implementation for pymmcoreplus-based microscopes."""
    
    def __init__(self, core: CMMCorePlus, settings: sp_microscope_settings):
        if not CMMCORE_PLUS_AVAILABLE:
            raise ImportError("pymmcore_plus is not installed")
        self.core = core
        self.settings = settings
        
    def move_to_position(self, position: sp_position) -> None:
        # Get current position and populate any missing coordinates
        current_position = self.get_current_position()
        position.populate_missing(current_position)
        
        if not self._is_coordinate_in_range(position):
            raise ValueError("Position out of range")
            
        # Set focus device if needed
        if self.core.getFocusDevice() != self.settings.stage.z_stage:
            self.core.setFocusDevice(self.settings.focus_device)
            
        # Move to position
        self.core.setPosition(position.z)
        self.core.setXYPosition(position.x, position.y)
        self.core.waitForDevice(self.core.getXYStageDevice())
        self.core.waitForDevice(self.core.getFocusDevice())
        
    def get_current_position(self) -> sp_position:
        return sp_position(
            self.core.getXPosition(),
            self.core.getYPosition(),
            self.core.getPosition()
        )
        
    def snap_image(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        self.core.snapImage()
        img = self.core.getImage()
        tags = {
            "Width": self.core.getImageWidth(),
            "Height": self.core.getImageHeight(),
            "PixelType": self.core.getPixelType(),
            "BytesPerPixel": self.core.getBytesPerPixel(),
            "BitDepth": self.core.getBitDepth(),
        }
        
        # Handle color cameras
        if self.core.getProperty("Core", "Camera") in ["QCamera", "MicroPublisher6"]:
            if self.core.getProperty("Color") == "ON":
                img = np.reshape(img, (tags["Height"], tags["Width"], 4))
                img = img[:, :, 0:3]  # Remove alpha
                img = np.flip(img, 2)  # Flip channels
            else:
                img = np.reshape(img, (tags["Height"], tags["Width"]))
        else:
            img = np.reshape(img, (tags["Height"], tags["Width"]))
            
        return img, tags
        
    def set_objective(self, objective_name: str) -> None:
        current_slider_position = self.core.getProperty(*self.settings.obj_slider)
        if objective_name != current_slider_position:
            if objective_name.startswith("4X"):
                self.core.setFocusDevice(self.settings.stage.z_stage)
                self.core.setPosition(self.settings.imaging_mode.z)
                self.core.waitForDevice(self.settings.stage.z_stage)
                self.core.setProperty(*self.settings.obj_slider, objective_name)
                self.core.setFocusDevice(self.settings.stage.f_stage)
                self.core.setPosition(self.settings.imaging_mode.f)
                self.core.waitForSystem()
            elif objective_name.startswith("20X"):
                self.core.setProperty(*self.settings.obj_slider, objective_name)
                self.core.waitForDevice(self.settings.obj_slider[0])
                self.core.setFocusDevice(self.settings.stage.z_stage)
                self.core.setPosition(self.settings.imaging_mode.z)
                self.core.setFocusDevice(self.settings.stage.f_stage)
                self.core.setPosition(self.settings.imaging_mode.f)
                self.core.waitForSystem()
                
            self.core.setFocusDevice(self.settings.stage.z_stage)
            
    def autofocus(self, **kwargs) -> float:
        from .smartpath import smartpath
        return smartpath.autofocus(self.core, self.settings, **kwargs)
        
    def _is_coordinate_in_range(self, position: sp_position) -> bool:
        from .smartpath import smartpath
        return smartpath.is_coordinate_in_range(self.settings, position)

class PymmcoreHardware(MicroscopeHardware):
    """Implementation for pymmcore-based microscopes."""
    
    def __init__(self, core: pymmcore.CMMCore, settings: sp_microscope_settings):
        if not PYMMCORE_AVAILABLE:
            raise ImportError("pymmcore is not installed")
        self.core = core
        self.settings = settings
        
    def move_to_position(self, position: sp_position) -> None:
        # Get current position and populate any missing coordinates
        current_position = self.get_current_position()
        position.populate_missing(current_position)
        
        if not self._is_coordinate_in_range(position):
            raise ValueError("Position out of range")
            
        # Set focus device if needed
        if self.core.getFocusDevice() != self.settings.stage.z_stage:
            self.core.setFocusDevice(self.settings.focus_device)
            
        # Move to position
        self.core.setPosition(position.z)
        self.core.setXYPosition(position.x, position.y)
        self.core.waitForDevice(self.core.getXYStageDevice())
        self.core.waitForDevice(self.core.getFocusDevice())
        
    def get_current_position(self) -> sp_position:
        return sp_position(
            self.core.getXPosition(),
            self.core.getYPosition(),
            self.core.getPosition()
        )
        
    def snap_image(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        self.core.snapImage()
        img = self.core.getImage()
        tags = {
            "Width": self.core.getImageWidth(),
            "Height": self.core.getImageHeight(),
            "PixelType": self.core.getPixelType(),
            "BytesPerPixel": self.core.getBytesPerPixel(),
            "BitDepth": self.core.getBitDepth(),
        }
        
        # Handle color cameras
        if self.core.getProperty("Core", "Camera") in ["QCamera", "MicroPublisher6"]:
            if self.core.getProperty("Color") == "ON":
                img = np.reshape(img, (tags["Height"], tags["Width"], 4))
                img = img[:, :, 0:3]  # Remove alpha
                img = np.flip(img, 2)  # Flip channels
            else:
                img = np.reshape(img, (tags["Height"], tags["Width"]))
        else:
            img = np.reshape(img, (tags["Height"], tags["Width"]))
            
        return img, tags
        
    def set_objective(self, objective_name: str) -> None:
        current_slider_position = self.core.getProperty(*self.settings.obj_slider)
        if objective_name != current_slider_position:
            if objective_name.startswith("4X"):
                self.core.setFocusDevice(self.settings.stage.z_stage)
                self.core.setPosition(self.settings.imaging_mode.z)
                self.core.waitForDevice(self.settings.stage.z_stage)
                self.core.setProperty(*self.settings.obj_slider, objective_name)
                self.core.setFocusDevice(self.settings.stage.f_stage)
                self.core.setPosition(self.settings.imaging_mode.f)
                self.core.waitForSystem()
            elif objective_name.startswith("20X"):
                self.core.setProperty(*self.settings.obj_slider, objective_name)
                self.core.waitForDevice(self.settings.obj_slider[0])
                self.core.setFocusDevice(self.settings.stage.z_stage)
                self.core.setPosition(self.settings.imaging_mode.z)
                self.core.setFocusDevice(self.settings.stage.f_stage)
                self.core.setPosition(self.settings.imaging_mode.f)
                self.core.waitForSystem()
                
            self.core.setFocusDevice(self.settings.stage.z_stage)
            
    def autofocus(self, **kwargs) -> float:
        from .smartpath import smartpath
        return smartpath.autofocus(self.core, self.settings, **kwargs)
        
    def _is_coordinate_in_range(self, position: sp_position) -> bool:
        from .smartpath import smartpath
        return smartpath.is_coordinate_in_range(self.settings, position)

def create_hardware(settings: sp_microscope_settings, config_path: Optional[str] = None) -> MicroscopeHardware:
    """Create a hardware instance using the best available backend.
    
    Args:
        settings: Microscope settings
        config_path: Path to Micro-Manager configuration file
        
    Returns:
        A hardware instance using the best available backend
        
    Raises:
        ImportError: If no suitable backend is available
    """
    # Try pycromanager first
    if PYCOMGR_AVAILABLE:
        try:
            from pycromanager import Core, Studio
            core = Core()
            if config_path:
                core.load_system_configuration(config_path)
            else:
                core.load_system_configuration()
            return PycromanagerHardware(core, settings)
        except Exception as e:
            warnings.warn(f"Failed to initialize pycromanager: {e}")
    
    # Try pymmcore_plus second
    if CMMCORE_PLUS_AVAILABLE:
        try:
            core = CMMCorePlus()
            if config_path:
                core.loadSystemConfiguration(config_path)
            else:
                core.loadSystemConfiguration()
            return PymmcoreplusHardware(core, settings)
        except Exception as e:
            warnings.warn(f"Failed to initialize pymmcore_plus: {e}")
    
    # Try pymmcore last
    if PYMMCORE_AVAILABLE:
        try:
            core = pymmcore.CMMCore()
            if config_path:
                core.loadSystemConfiguration(config_path)
            else:
                core.loadSystemConfiguration()
            return PymmcoreHardware(core, settings)
        except Exception as e:
            warnings.warn(f"Failed to initialize pymmcore: {e}")
    
    raise ImportError("No suitable Micro-Manager backend found. Please install one of: pycromanager, pymmcore_plus, or pymmcore") 