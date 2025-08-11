"""Smart WSI Scanner - A smart whole slide image scanner with hardware abstraction."""

__version__ = "0.1.0"

from .hardware import MicroscopeHardware
from .hardware_pycromanager import (
    PycromanagerHardware,
    init_pycromanager,
)
from .config import ConfigManager, sp


__all__ = [
    "MicroscopeHardware",
    "ConfigManager",
    "sp",
    ## pycromanager specific imports
    "PycromanagerHardware",
    "init_pycromanager",
]
