"""
Autofocus Metrics - COMPATIBILITY SHIM

This module is a backward-compatibility shim that re-exports AutofocusMetrics
from its new location. New code should import directly from:

    from smart_wsi_scanner.autofocus import AutofocusMetrics
    from smart_wsi_scanner.autofocus.metrics import AutofocusMetrics

This shim exists for backward compatibility and will be deprecated
in a future release.
"""

# Re-export from new location
from smart_wsi_scanner.autofocus.metrics import AutofocusMetrics

__all__ = ["AutofocusMetrics"]
