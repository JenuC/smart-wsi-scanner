"""
Autofocus Testing Module - COMPATIBILITY SHIM

This module is a backward-compatibility shim that re-exports autofocus testing
functions from their new location. New code should import directly from:

    from smart_wsi_scanner.autofocus.test import (
        test_standard_autofocus_at_current_position,
        test_adaptive_autofocus_at_current_position,
        test_autofocus_at_current_position,
    )

This shim exists for backward compatibility and will be deprecated
in a future release.
"""

# Re-export from new location
from smart_wsi_scanner.autofocus.test import (
    test_standard_autofocus_at_current_position,
    test_adaptive_autofocus_at_current_position,
    test_autofocus_at_current_position,
)

__all__ = [
    "test_standard_autofocus_at_current_position",
    "test_adaptive_autofocus_at_current_position",
    "test_autofocus_at_current_position",
]
