"""
Server Protocol Configuration - COMPATIBILITY SHIM

This module is a backward-compatibility shim that re-exports protocol
definitions from their new location. New code should import directly from:

    from smart_wsi_scanner.server.protocol import Command, ExtendedCommand, TCP_PORT, END_MARKER

This shim exists for backward compatibility and will be deprecated
in a future release.
"""

# Re-export from new location
from .server.protocol import Command, ExtendedCommand, TCP_PORT, END_MARKER

__all__ = ["Command", "ExtendedCommand", "TCP_PORT", "END_MARKER"]
