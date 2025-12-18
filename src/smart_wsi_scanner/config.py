"""
Configuration Manager - COMPATIBILITY SHIM

This module is a backward-compatibility shim that re-exports ConfigManager
from its new location. New code should import directly from:

    from smart_wsi_scanner.config import ConfigManager

This shim exists for backward compatibility and will be deprecated
in a future release.
"""

# Re-export from new location
from smart_wsi_scanner.config.manager import ConfigManager

__all__ = ["ConfigManager"]
