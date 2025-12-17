"""
Pycromanager Hardware Implementation - COMPATIBILITY SHIM

This module is a backward-compatibility shim that re-exports classes
from their new locations. New code should import directly from:

    from smart_wsi_scanner.hardware import PycromanagerHardware, init_pycromanager
    from smart_wsi_scanner.hardware.pycromanager import (
        PycromanagerHardware,
        init_pycromanager,
        ppm_psgticks_to_thor,
        ppm_thor_to_psgticks,
        obj_2_list,
    )

This shim exists for backward compatibility and will be deprecated
in a future release.
"""

# Re-export from new location
from .hardware.pycromanager import (
    PycromanagerHardware,
    init_pycromanager,
    ppm_psgticks_to_thor,
    ppm_thor_to_psgticks,
    obj_2_list,
)

__all__ = [
    "PycromanagerHardware",
    "init_pycromanager",
    "ppm_psgticks_to_thor",
    "ppm_thor_to_psgticks",
    "obj_2_list",
]
