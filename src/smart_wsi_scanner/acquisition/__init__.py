"""
Acquisition package - Workflows for microscope image acquisition.

This package contains the core acquisition workflow logic,
tile configuration utilities, and QuPath project management.

Modules:
    workflow: Main acquisition workflow orchestration
    tiles: Tile configuration utilities (TileConfigUtils)
    project: QuPath project management (QuPathProject)
    pipeline: Text-based image processing pipeline
"""

from .tiles import TileConfigUtils
from .project import QuPathProject

__all__ = ["TileConfigUtils", "QuPathProject"]
