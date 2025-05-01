"""Hardware abstraction layer for microscope control."""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
import numpy as np
from collections import OrderedDict
from pycromanager import Core, Studio

from .config import sp_microscope_settings, sp_position, sp_imaging_mode

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
        self.core = core
        self.settings = settings
        self.studio = studio or Studio()
        
    def move_to_position(self, position: sp_position) -> None:
        if not self._is_coordinate_in_range(position):
            raise ValueError("Position out of range")
            
        if not position.x:
            position.x = self.get_current_position().x
        if not position.y:
            position.y = self.get_current_position().y
        if not position.z:
            position.z = self.get_current_position().z
            
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
            pixels = np.reshape(tagged_image.pix, 
                            newshape=[tags["Height"], tags["Width"], 4])
            pixels = pixels[:, :, 0:3]  # Remove alpha
            pixels = np.flip(pixels, 2)  # Flip channels
            return pixels, tags
        else:
            pixels = np.reshape(tagged_image.pix, 
                            newshape=[tags["Height"], tags["Width"]])
            return pixels, tags
        #return tagged_image.pix, tags
        
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