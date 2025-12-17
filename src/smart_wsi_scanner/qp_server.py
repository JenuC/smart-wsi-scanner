"""
QuPath Microscope Server - COMPATIBILITY SHIM

This module is a backward-compatibility shim that re-exports server
components from their new location. New code should import directly from:

    from smart_wsi_scanner.server.qp_server import run_server, ...

This shim exists for backward compatibility and will be deprecated
in a future release.

To run the server, use:
    python -m smart_wsi_scanner.server.qp_server
"""

# Re-export from new location
from .server.qp_server import *

# Also re-export the main function if module is run directly
if __name__ == "__main__":
    from .server.qp_server import main
    main()
