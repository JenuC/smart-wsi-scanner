"""
Text Pipeline - COMPATIBILITY SHIM

This module is a backward-compatibility shim that re-exports pipeline
functions from their new location. New code should import directly from:

    from smart_wsi_scanner.acquisition.pipeline import parse_steps, run_pipeline, ...

This shim exists for backward compatibility and will be deprecated
in a future release.
"""

# Re-export from new location
from smart_wsi_scanner.acquisition.pipeline import *
