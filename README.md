# Smart WSI Scanner

A smart whole slide image scanner with hardware abstraction and configuration management for the QPSC project.

## Features

- **Hardware abstraction layer** for microscope control via Pycro-Manager
- **Flexible configuration management** with YAML support and LOCI resource lookup
- **Multi-modal imaging support** including brightfield and polarized light microscopy (PPM)
- **Autofocus capabilities** with standard and adaptive search algorithms
- **Autofocus benchmarking** for parameter optimization
- **Image acquisition workflows** with tile management
- **Background correction** and tissue detection
- **PPM-specific tools** including polarizer calibration, rotation sensitivity analysis, and birefringence optimization
- **Socket-based server** for QuPath integration with real-time progress reporting

## Package Structure

```
smart_wsi_scanner/
    hardware/       - Hardware abstraction (Position, MicroscopeHardware, PycromanagerHardware)
    config/         - Configuration management (ConfigManager)
    autofocus/      - Autofocus algorithms, metrics, testing, and benchmarking
    acquisition/    - Acquisition workflows, tile management, and pipelines
    imaging/        - Image processing (TIF writing, background correction, tissue detection)
    ppm/            - Polarized light microscopy tools (calibration, sensitivity, birefringence)
    server/         - Socket server for QuPath communication (protocol, server, client)
```

## Quick Start

```python
from smart_wsi_scanner.hardware import Position, MicroscopeHardware
from smart_wsi_scanner.hardware.pycromanager import PycromanagerHardware, init_pycromanager
from smart_wsi_scanner.config import ConfigManager

# Initialize Pycro-Manager connection
core, studio = init_pycromanager()

# Load configuration
config_manager = ConfigManager()
settings = config_manager.load_config("config_PPM.yml", "resources_LOCI.yml")

# Initialize hardware
hardware = PycromanagerHardware(core, studio, settings)

# Get current position
pos = hardware.get_current_position()
print(f"Current position: X={pos.x}, Y={pos.y}, Z={pos.z}")

# Run autofocus
from smart_wsi_scanner.autofocus import AutofocusUtils
focus_z = hardware.autofocus(n_steps=21, search_range=30.0)
```

## Server Usage

The socket server enables QuPath to control the microscope:

```bash
# Start the server
python -m smart_wsi_scanner.server.qp_server --config config_PPM.yml

# Server listens on port 5000 by default
```

## Import Paths

### Recommended (New Style)
```python
from smart_wsi_scanner.hardware import Position, MicroscopeHardware
from smart_wsi_scanner.hardware.pycromanager import PycromanagerHardware, init_pycromanager
from smart_wsi_scanner.config import ConfigManager
from smart_wsi_scanner.autofocus import AutofocusUtils, AutofocusBenchmark
from smart_wsi_scanner.acquisition import TileConfigUtils, QuPathProject
from smart_wsi_scanner.imaging import TifWriterUtils, BackgroundCorrectionUtils
from smart_wsi_scanner.ppm import PolarizerCalibrationUtils
from smart_wsi_scanner.server.protocol import Command, ExtendedCommand
```

### Legacy (Backward Compatible)
```python
# These still work for backward compatibility
from smart_wsi_scanner import Position, MicroscopeHardware
from smart_wsi_scanner import PycromanagerHardware, init_pycromanager
from smart_wsi_scanner import ConfigManager
from smart_wsi_scanner.qp_utils import AutofocusUtils, TileConfigUtils
```

## Development Scripts

Located in `src/dev_tests/`:

```bash
# Run autofocus benchmark
python run_autofocus_benchmark.py --reference_z -5000.0 --output ./results --objective 20X

# PPM rotation sensitivity test
python -m smart_wsi_scanner.ppm.sensitivity_test config.yml --output ./results

# PPM birefringence optimization
python -m smart_wsi_scanner.ppm.birefringence_test config.yml --mode interpolate
```

## Server Commands

| Command | Description |
|---------|-------------|
| ACQUIRE | Start tile acquisition |
| TESTAF | Test standard autofocus at current position |
| TESTADAF | Test adaptive autofocus at current position |
| AFBENCH | Run autofocus parameter benchmark |
| POLCAL | Calibrate polarizer rotation stage |
| PPMSENS | PPM rotation sensitivity test |
| PPMBIREF | PPM birefringence maximization test |
| BGACQUIRE | Acquire background images |

## License

MIT License
