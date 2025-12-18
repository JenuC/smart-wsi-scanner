# Smart WSI Scanner - Architecture Documentation

A Python package for intelligent whole slide image acquisition with hardware abstraction, multi-modal imaging support, and QuPath integration.

## Package Structure

```
smart_wsi_scanner/
    hardware/           # Hardware abstraction layer
    config/             # Configuration management
    autofocus/          # Autofocus algorithms and benchmarking
    acquisition/        # Acquisition workflows and tile management
    imaging/            # Image processing utilities
    ppm/                # Polarized light microscopy tools
    server/             # Socket server for QuPath communication
    debayering/         # Bayer pattern demosaicing
    configurations/     # YAML configuration files
    tests/              # Unit tests
```

---

## Core Packages

### `hardware/` - Hardware Abstraction Layer

Provides a unified interface for microscope control, abstracting away the specifics of different microscope systems.

| Module | Description |
|--------|-------------|
| `base.py` | Abstract `MicroscopeHardware` class, `Position` dataclass, utility functions |
| `pycromanager.py` | Concrete implementation using Pycro-Manager/Micro-Manager |

**Key Classes:**
- `Position` - Represents XYZ stage coordinates
- `MicroscopeHardware` - Abstract base class for microscope control
- `PycromanagerHardware` - Pycro-Manager implementation with support for:
  - Stage movement (XY, Z, rotation)
  - Image capture with debayering
  - Standard and adaptive autofocus
  - PPM rotation stage control
  - Camera exposure and white balance

**Usage:**
```python
from smart_wsi_scanner.hardware import Position, MicroscopeHardware
from smart_wsi_scanner.hardware.pycromanager import PycromanagerHardware, init_pycromanager

core, studio = init_pycromanager()
hardware = PycromanagerHardware(core, studio, settings)
pos = hardware.get_current_position()
```

---

### `config/` - Configuration Management

Manages YAML-based microscope configurations with support for multiple microscopes, modalities, and acquisition profiles.

| Module | Description |
|--------|-------------|
| `manager.py` | `ConfigManager` class for loading/saving configurations |

**Key Features:**
- Loads all `.yml` files from `configurations/` directory
- Supports LOCI resource file lookups (shared hardware definitions)
- Provides typed accessors for common configuration elements
- Validates configuration structure

**Configuration Files (in `configurations/`):**
- `config_PPM.yml` - PPM microscope configuration
- `config_CAMM.yml` - CAMM microscope configuration
- `imageprocessing_PPM.yml` - Image processing settings
- `autofocus_PPM.yml` - Autofocus parameters
- `resources/resources_LOCI.yml` - Shared hardware definitions

**Usage:**
```python
from smart_wsi_scanner.config import ConfigManager

config_manager = ConfigManager()
settings = config_manager.get_config("config_PPM")
pixel_size = config_manager.get_pixel_size("LOCI_OBJECTIVE_10X", "LOCI_DETECTOR_JAI")
```

---

### `autofocus/` - Autofocus Algorithms

Provides focus optimization algorithms for microscope imaging.

| Module | Description |
|--------|-------------|
| `core.py` | `AutofocusUtils` - focus metrics and validation |
| `metrics.py` | `AutofocusMetrics` - focus quality scoring functions |
| `benchmark.py` | `AutofocusBenchmark` - systematic parameter testing |
| `test.py` | Interactive autofocus testing utilities |

**Autofocus Methods:**
1. **Standard Autofocus** - Samples Z positions, finds focus via interpolation
2. **Adaptive Autofocus** - Drift-check with minimal acquisitions

**Metrics Available:**
- Laplacian variance (default)
- Sobel gradient
- Brenner gradient

**Usage:**
```python
from smart_wsi_scanner.autofocus import AutofocusUtils

# Standard autofocus
best_z = hardware.autofocus(n_steps=21, search_range=30.0)

# Adaptive (drift-check) autofocus
best_z = hardware.autofocus_adaptive_search(initial_step_size=10)
```

---

### `acquisition/` - Acquisition Workflows

Orchestrates multi-tile, multi-angle image acquisition.

| Module | Description |
|--------|-------------|
| `workflow.py` | Main acquisition workflow with progress reporting |
| `tiles.py` | `TileConfigUtils` - tile grid configuration |
| `project.py` | `QuPathProject` - QuPath project management |
| `pipeline.py` | Text-based processing pipeline |

**Key Features:**
- Reads tile positions from QuPath-generated configuration files
- Supports multi-angle acquisition (PPM rotation sequences)
- Real-time progress reporting via socket
- Autofocus per-tile with manual focus fallback
- Background correction integration

---

### `imaging/` - Image Processing

General-purpose image processing utilities.

| Module | Description |
|--------|-------------|
| `writer.py` | `TifWriterUtils` - TIFF file writing |
| `background.py` | `BackgroundCorrectionUtils` - flat-field correction |
| `tissue_detection.py` | `EmptyRegionDetector` - tissue/background classification |
| `jai_calibration.py` | `JAIWhiteBalanceCalibrator` - JAI camera white balance |

**Usage:**
```python
from smart_wsi_scanner.imaging import BackgroundCorrectionUtils, TifWriterUtils

# Apply background correction
corrected = BackgroundCorrectionUtils.apply_correction(image, background)

# Save TIFF
TifWriterUtils.save_tiff(image, path, metadata)
```

---

### `ppm/` - Polarized Light Microscopy

Tools specific to Polarized light Microscopy (PPM) imaging.

| Module | Description |
|--------|-------------|
| `calibration.py` | `PolarizerCalibrationUtils` - polarizer stage calibration |
| `sensitivity_test.py` | `PPMRotationSensitivityTester` - rotation sensitivity analysis |
| `sensitivity_analysis.py` | `PPMRotationAnalyzer` - analyze sensitivity test results |
| `birefringence_test.py` | `PPMBirefringenceMaximizationTester` - optimize birefringence |

**Key Features:**
- Polarizer rotation stage calibration
- Multi-angle acquisition sequences
- Birefringence signal optimization
- Rotation sensitivity analysis

---

### `server/` - QuPath Communication

Socket-based server for remote microscope control from QuPath.

| Module | Description |
|--------|-------------|
| `protocol.py` | Command definitions (`Command`, `ExtendedCommand`) |
| `qp_server.py` | Main server implementation |
| `client.py` | Test client utilities |

**Server Commands:**

| Command | Description |
|---------|-------------|
| `GETXY` | Get current XY position |
| `GETZ` | Get current Z position |
| `MOVEZ` | Move Z stage |
| `MOVE` | Move XY stage |
| `ACQUIRE` | Start tile acquisition |
| `TESTAF` | Test standard autofocus |
| `TESTADAF` | Test adaptive autofocus |
| `AFBENCH` | Run autofocus benchmark |
| `BGACQUIRE` | Acquire background images |
| `POLCAL` | Calibrate polarizer |
| `PPMSENS` | PPM rotation sensitivity test |
| `PPMBIREF` | PPM birefringence optimization |
| `SNAP` | Simple image capture |

**Running the Server:**
```bash
cd src
python -m smart_wsi_scanner.server.qp_server
```

---

### `debayering/` - Bayer Pattern Processing

CPU and GPU implementations for Bayer pattern demosaicing (used with MicroPublisher6 camera).

| Module | Description |
|--------|-------------|
| `src/main_cpu.py` | `CPUDebayer` - CPU-based demosaicing |
| `src/main_gpu.py` | GPU-accelerated demosaicing (optional) |

---

## Compatibility Shims

The following files at the package root provide backward compatibility with older import paths. New code should import from the subpackages directly.

| Shim File | Redirects To |
|-----------|--------------|
| `qp_server.py` | `server.qp_server` |
| `qp_utils.py` | `acquisition`, `autofocus`, `imaging`, `ppm` |
| `hardware_pycromanager.py` | `hardware.pycromanager` |
| `config.py` | `config.manager` |
| `qp_client.py` | `server.client` |
| `qp_acquisition.py` | `acquisition.workflow` |
| `qp_autofocus_test.py` | `autofocus.test` |
| `qp_autofocus_benchmark.py` | `autofocus.benchmark` |

---

## Configuration Files

Located in `configurations/`:

```
configurations/
    config_PPM.yml              # Main PPM microscope config
    config_CAMM.yml             # CAMM microscope config
    config_template.yml         # Template for new configs
    imageprocessing_PPM.yml     # Image processing settings
    autofocus_PPM.yml           # Autofocus parameters
    resources/
        resources_LOCI.yml      # Shared hardware definitions
```

---

## Data Flow

```
QuPath Annotation
       |
       v
  Socket Server (qp_server.py)
       |
       v
  Acquisition Workflow (acquisition/workflow.py)
       |
       +---> Hardware Control (hardware/pycromanager.py)
       |           |
       |           v
       |     Micro-Manager / Pycro-Manager
       |           |
       |           v
       |     Microscope Hardware
       |
       +---> Autofocus (autofocus/core.py)
       |
       +---> Image Processing (imaging/)
       |
       v
  TIFF Files + Tile Configuration
       |
       v
  QuPath Stitching (external)
```

---

## Testing

Tests are in `tests/` and use pytest:

```bash
cd src
python -m pytest smart_wsi_scanner/tests/
```

---

## Development

### Adding a New Modality

1. Create handler in `ppm/` or new modality package
2. Add rotation angles and acquisition settings to config
3. Register with `ModalityRegistry` (QuPath side)
4. Add server command if needed

### Adding a New Camera

1. Add camera-specific logic to `hardware/pycromanager.py`
2. Add calibration utilities to `imaging/` if needed
3. Update configuration schema

### Adding a New Server Command

1. Add command to `server/protocol.py` (8 bytes, padded)
2. Add handler in `server/qp_server.py`
3. Add client method in QuPath's `MicroscopeSocketClient.java`
