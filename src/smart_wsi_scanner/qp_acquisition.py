"""
Acquisition Workflow - COMPATIBILITY SHIM

This module is a backward-compatibility shim that re-exports acquisition
workflow functions from their new location. New code should import directly from:

    from smart_wsi_scanner.acquisition.workflow import _acquisition_workflow, ...

This shim exists for backward compatibility and will be deprecated
in a future release.
"""

# Re-export from new location
from smart_wsi_scanner.acquisition.workflow import *

# Explicitly export key functions used by qp_server
from smart_wsi_scanner.acquisition.workflow import _acquisition_workflow

__all__ = ["_acquisition_workflow"]
