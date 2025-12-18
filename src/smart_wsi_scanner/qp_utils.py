"""
Utility classes for QuPath Scope Control - COMPATIBILITY SHIM

This module is a backward-compatibility shim that re-exports classes
from their new locations. New code should import directly from the
new package locations:

    - TileConfigUtils: from smart_wsi_scanner.acquisition import TileConfigUtils
    - AutofocusUtils: from smart_wsi_scanner.autofocus import AutofocusUtils
    - TifWriterUtils: from smart_wsi_scanner.imaging import TifWriterUtils
    - BackgroundCorrectionUtils: from smart_wsi_scanner.imaging import BackgroundCorrectionUtils
    - QuPathProject: from smart_wsi_scanner.acquisition import QuPathProject
    - PolarizerCalibrationUtils: from smart_wsi_scanner.ppm import PolarizerCalibrationUtils

This shim exists for backward compatibility and will be deprecated
in a future release.
"""

import warnings
import logging

logger = logging.getLogger(__name__)

# Re-export all classes from their new locations for backward compatibility

# Acquisition package
from smart_wsi_scanner.acquisition.tiles import TileConfigUtils
from smart_wsi_scanner.acquisition.project import QuPathProject

# Autofocus package
from smart_wsi_scanner.autofocus.core import AutofocusUtils

# Imaging package
from smart_wsi_scanner.imaging.writer import TifWriterUtils
from smart_wsi_scanner.imaging.background import BackgroundCorrectionUtils

# PPM package
from smart_wsi_scanner.ppm.calibration import PolarizerCalibrationUtils

# Define what gets exported with "from qp_utils import *"
__all__ = [
    "TileConfigUtils",
    "AutofocusUtils",
    "TifWriterUtils",
    "BackgroundCorrectionUtils",
    "QuPathProject",
    "PolarizerCalibrationUtils",
]

# Optional: Issue deprecation warning when module is imported
# Uncomment when ready to deprecate this shim
# warnings.warn(
#     "Importing from qp_utils is deprecated. "
#     "Please import from the specific packages (acquisition, autofocus, imaging, ppm) instead.",
#     DeprecationWarning,
#     stacklevel=2
# )
