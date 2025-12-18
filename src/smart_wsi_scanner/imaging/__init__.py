"""
Imaging package - General image processing utilities.

This package contains image processing functionality that is
NOT specific to any particular modality (PPM, brightfield, etc).

Modules:
    background: Background correction utilities (BackgroundCorrectionUtils)
    writer: TIFF writing utilities (TifWriterUtils)
    tissue_detection: Empty region / tissue detection (EmptyRegionDetector)
    jai_calibration: JAI camera white balance calibration (JAIWhiteBalanceCalibrator)
"""

from smart_wsi_scanner.imaging.writer import TifWriterUtils
from smart_wsi_scanner.imaging.background import BackgroundCorrectionUtils
from smart_wsi_scanner.imaging.tissue_detection import EmptyRegionDetector

__all__ = ["TifWriterUtils", "BackgroundCorrectionUtils", "EmptyRegionDetector"]

# Optional: JAI calibration (may not be needed on all systems)
try:
    from smart_wsi_scanner.imaging.jai_calibration import (
        JAIWhiteBalanceCalibrator,
        WhiteBalanceResult,
        CalibrationConfig,
    )
    __all__.extend(["JAIWhiteBalanceCalibrator", "WhiteBalanceResult", "CalibrationConfig"])
except ImportError:
    pass
