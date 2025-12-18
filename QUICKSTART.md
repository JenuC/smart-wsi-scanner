# Smart WSI Scanner - Quick Start Guide

Python-based microscope control for intelligent whole slide image acquisition with QuPath integration.

## What This Does

This package provides:
- **Microscope Control**: Stage movement, autofocus, image capture via Pycro-Manager
- **Multi-Modal Imaging**: Support for brightfield and polarized light microscopy (PPM)
- **QuPath Integration**: Socket-based communication for annotation-driven acquisition
- **Automated Acquisition**: Tile-based imaging with autofocus and background correction

## Package Organization

```
smart_wsi_scanner/
    hardware/       - Microscope control (stage, camera, autofocus)
    config/         - YAML configuration management
    autofocus/      - Focus algorithms and benchmarking
    acquisition/    - Tile acquisition workflows
    imaging/        - Image processing (background correction, etc.)
    ppm/            - Polarized microscopy tools
    server/         - QuPath communication server
```

## Running the Server

```bash
cd smart-wsi-scanner/src
python -m smart_wsi_scanner.server.qp_server
```

The server listens on port 5000 and accepts commands from QuPath.

## Configuration

Configuration files are in `configurations/`:
- `config_PPM.yml` - Main microscope settings
- `resources/resources_LOCI.yml` - Shared hardware definitions

## Key Server Commands

| Command | Description |
|---------|-------------|
| `ACQUIRE` | Run tile acquisition from QuPath annotations |
| `TESTAF` | Test autofocus at current position |
| `BGACQUIRE` | Acquire background images for correction |

## For Developers

See `ARCHITECTURE.md` for detailed package structure and APIs.

## Requirements

- Python 3.9+
- Micro-Manager 2.0 with Pycro-Manager
- numpy, scipy, scikit-image, PyYAML
