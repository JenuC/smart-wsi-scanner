"""
Hardware abstraction layer for microscope control - COMPATIBILITY SHIM

This module is a backward-compatibility shim that re-exports classes
from the hardware package. New code should import directly from:

    from smart_wsi_scanner.hardware import Position, MicroscopeHardware
    from smart_wsi_scanner.hardware import is_mm_running, is_coordinate_in_range

This shim exists for backward compatibility with code that imported
from smart_wsi_scanner.hardware (the module) before it became a package.
"""

# Re-export from the hardware package
from .hardware import (
    Position,
    MicroscopeHardware,
    is_mm_running,
    is_coordinate_in_range,
)

__all__ = [
    "Position",
    "MicroscopeHardware",
    "is_mm_running",
    "is_coordinate_in_range",
]
