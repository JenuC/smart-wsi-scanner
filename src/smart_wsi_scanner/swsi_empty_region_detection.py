"""
Empty Region Detection - COMPATIBILITY SHIM

This module is a backward-compatibility shim that re-exports EmptyRegionDetector
from its new location. New code should import directly from:

    from smart_wsi_scanner.imaging import EmptyRegionDetector
    from smart_wsi_scanner.imaging.tissue_detection import EmptyRegionDetector

This shim exists for backward compatibility and will be deprecated
in a future release.
"""

# Re-export from new location
from .imaging.tissue_detection import EmptyRegionDetector

__all__ = ["EmptyRegionDetector"]
