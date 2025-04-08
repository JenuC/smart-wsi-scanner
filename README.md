# Smart WSI Scanner

A smart whole slide image scanner with hardware abstraction and configuration management.

## Features

- Hardware abstraction layer for different microscope types
- Flexible configuration management with YAML support
- Support for multiple imaging modes and objectives
- Autofocus capabilities
- Image acquisition and processing

## Installation

### Prerequisites

- Python 3.12 or higher
- pip or uv package manager

### Using uv (recommended)

```bash
# Create a new virtual environment with Python 3.12
uv venv -p 3.12

# Activate the virtual environment
# On Windows:
.venv/Scripts/activate
# On Unix/MacOS:
source .venv/bin/activate

# Install the package
uv pip install -e .
```

### Using pip

```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows:
.venv/Scripts/activate
# On Unix/MacOS:
source .venv/bin/activate

# Install the package
pip install -e .
```

## Usage

```python
from smart_wsi_scanner import smartpath, PycromanagerHardware, ConfigManager
from pycromanager import Core

# Initialize hardware and config manager
core = Core()
hardware = PycromanagerHardware(core, settings)
config_manager = ConfigManager()

# Create smartpath instance
sp = smartpath(hardware, config_manager)

# Load a specific configuration
sp.load_config("my_microscope_config")

# Move to a position
position = sp_position(x=100, y=100, z=0)
sp.move_to_position(position)

# Capture an image
image, metadata = sp.snap_image()

# Save current configuration
sp.save_config("new_config")
```

## Development

### Setting up development environment

```bash
# Install development dependencies
uv pip install -e ".[dev]"
```

### Running tests

```bash
pytest
```

### Code formatting

```bash
# Format code
black .

# Run linter
ruff check .
```

## License

MIT License
