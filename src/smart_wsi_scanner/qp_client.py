"""
QuPath Client - COMPATIBILITY SHIM

This module is a backward-compatibility shim that re-exports client
utilities from their new location. New code should import directly from:

    from smart_wsi_scanner.server.client import get_stageXY, get_stageZ, ...

This shim exists for backward compatibility and will be deprecated
in a future release.
"""

# Re-export from new location
from .server.client import (
    get_stageXY,
    get_stageZ,
    move_stageZ,
    move_stageXY,
    get_stageR,
    move_stageR,
    shutdown_server,
    disconnect,
    get,
    main,
    HOST,
    PORT,
)

__all__ = [
    "get_stageXY",
    "get_stageZ",
    "move_stageZ",
    "move_stageXY",
    "get_stageR",
    "move_stageR",
    "shutdown_server",
    "disconnect",
    "get",
    "main",
    "HOST",
    "PORT",
]
