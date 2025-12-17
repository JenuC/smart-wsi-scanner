"""
Imaging package - General image processing utilities.

This package contains image processing functionality that is
NOT specific to any particular modality (PPM, brightfield, etc).

Modules:
    background: Background correction utilities (BackgroundCorrectionUtils)
    writer: TIFF writing utilities (TifWriterUtils)
    tissue_detection: Empty region / tissue detection (EmptyRegionDetector)
    debayering: Bayer pattern demosaicing (existing subpackage)
"""

from .writer import TifWriterUtils
from .background import BackgroundCorrectionUtils
from .tissue_detection import EmptyRegionDetector

__all__ = ["TifWriterUtils", "BackgroundCorrectionUtils", "EmptyRegionDetector"]
