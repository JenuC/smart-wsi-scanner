"""
Server package - Socket server for QuPath communication.

This package contains the socket-based communication layer between
QuPath and the microscope control system.

Modules:
    qp_server: Main socket server implementation
    protocol: Command definitions and wire protocol
    client: Test client utilities
"""

from smart_wsi_scanner.server.protocol import Command, ExtendedCommand, TCP_PORT, END_MARKER

__all__ = ["Command", "ExtendedCommand", "TCP_PORT", "END_MARKER"]
