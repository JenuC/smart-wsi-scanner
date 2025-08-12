"""Hardware abstraction layer for microscope control."""

from abc import ABC, abstractmethod
from typing import Optional
import warnings

from .config import sp_microscope_settings, sp_position



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



def is_mm_running() -> bool:
    """Check if Micro-Manager is running as a Windows executable."""
    import platform
    import psutil

    if platform.system() != "Windows":
        return False

    for proc in psutil.process_iter(["name"]):
        try:
            if proc.exe().find("Micro-Manager") > 0:
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return False


def is_coordinate_in_range(
    settings: sp_microscope_settings, position: sp_position
) -> bool:  #: sp_microscope_settings,  #: sp_position
    _within_y_limit = _within_x_limit = False

    if settings.stage is not None and settings.stage.x_limit is not None:
        if (
            position.x is not None
            and settings.stage.x_limit.low is not None
            and settings.stage.x_limit.high is not None
            and settings.stage.x_limit.low < position.x < settings.stage.x_limit.high
        ):
            _within_x_limit = True
        else:
            warnings.warn(f" {position.x} out of range X {getattr(settings.stage, 'x_limit', None)}")
    else:
        warnings.warn(
            f" X limit values are not properly defined: {getattr(settings.stage, 'x_limit', None)}"
        )

    if settings.stage is not None and settings.stage.y_limit is not None:
        if (
            position.y is not None
            and settings.stage.y_limit.low is not None
            and settings.stage.y_limit.high is not None
            and settings.stage.y_limit.low < position.y < settings.stage.y_limit.high
        ):
            _within_y_limit = True
        else:
            warnings.warn(f" {position.y} out of range Y {getattr(settings.stage, 'y_limit', None)}")
    else:
        warnings.warn(
            f" Y limit values are not properly defined: {getattr(settings.stage, 'y_limit', None)}"
        )

    if not position.z:
        return _within_x_limit and _within_y_limit

    if (
        settings.stage is not None
        and getattr(settings.stage, "z_limit", None) is not None
        and getattr(settings.stage.z_limit, "low", None) is not None
        and getattr(settings.stage.z_limit, "high", None) is not None
    ):
        _within_z_limit = False
        z_limit = settings.stage.z_limit
        if z_limit is not None and z_limit.low is not None and z_limit.high is not None:
            if z_limit.low < position.z < z_limit.high:
                _within_z_limit = True
            else:
                warnings.warn(f" {position.z} out of range Z {z_limit}")
            return _within_x_limit and _within_y_limit and _within_z_limit
        else:
            warnings.warn(f" Z range undefined or limits not set: {z_limit}")
            return False
    else:
        warnings.warn(
            f" Z range undefined or limits not set: {getattr(settings.stage, 'z_limit', None)}"
        )
        return False
