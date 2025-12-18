"""
Smart WSI Scanner - A smart whole slide image scanner with hardware abstraction.

Package Structure
-----------------
The package is organized into subpackages by functionality:

    smart_wsi_scanner/
        hardware/       - Hardware abstraction (Position, MicroscopeHardware, PycromanagerHardware)
        config/         - Configuration management (ConfigManager)
        autofocus/      - Autofocus algorithms and benchmarking
        acquisition/    - Acquisition workflows and tile management
        imaging/        - Image processing (background correction, tissue detection)
        ppm/            - Polarized light microscopy tools
        server/         - Socket server for QuPath communication

Recommended Imports (New Style)
-------------------------------
    from smart_wsi_scanner.hardware import Position, MicroscopeHardware
    from smart_wsi_scanner.hardware.pycromanager import PycromanagerHardware, init_pycromanager
    from smart_wsi_scanner.config import ConfigManager
    from smart_wsi_scanner.autofocus import AutofocusUtils
    from smart_wsi_scanner.acquisition import TileConfigUtils, QuPathProject
    from smart_wsi_scanner.imaging import TifWriterUtils, BackgroundCorrectionUtils
    from smart_wsi_scanner.ppm import PolarizerCalibrationUtils

Legacy Imports (Backward Compatible)
------------------------------------
The following imports still work for backward compatibility:

    from smart_wsi_scanner import Position, MicroscopeHardware
    from smart_wsi_scanner import PycromanagerHardware, init_pycromanager
    from smart_wsi_scanner import ConfigManager
    from smart_wsi_scanner.qp_utils import AutofocusUtils, TileConfigUtils
"""

__version__ = "0.1.0"

# Core imports that don't require external dependencies
from smart_wsi_scanner.hardware import MicroscopeHardware, Position

# Optional imports - only available if scipy/numpy/etc. are installed
# These are wrapped to allow basic package usage without all dependencies
try:
    from smart_wsi_scanner.qp_utils import TifWriterUtils, TileConfigUtils, AutofocusUtils, QuPathProject
    _UTILS_AVAILABLE = True
except ImportError:
    TifWriterUtils = None
    TileConfigUtils = None
    AutofocusUtils = None
    QuPathProject = None
    _UTILS_AVAILABLE = False

# Optional pycromanager imports - only available if pycromanager is installed
try:
    from smart_wsi_scanner.hardware_pycromanager import (
        PycromanagerHardware,
        init_pycromanager,
    )
    _PYCROMANAGER_AVAILABLE = True
except ImportError:
    PycromanagerHardware = None
    init_pycromanager = None
    _PYCROMANAGER_AVAILABLE = False

# Optional config import - may have dependencies
try:
    from smart_wsi_scanner.config import ConfigManager
    _CONFIG_AVAILABLE = True
except ImportError:
    ConfigManager = None
    _CONFIG_AVAILABLE = False

__all__ = [
    # Core exports (always available)
    "MicroscopeHardware",
    "Position",
    # Optional exports (may be None if dependencies not installed)
    "TifWriterUtils",
    "TileConfigUtils",
    "AutofocusUtils",
    "QuPathProject",
    "ConfigManager",
    "PycromanagerHardware",
    "init_pycromanager",
]
