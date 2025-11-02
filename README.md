# Smart WSI Scanner

A smart whole slide image scanner with hardware abstraction and configuration management.

## Features

- **Hardware abstraction layer** for different microscope types
- **Flexible configuration management** with YAML support
- **Support for multiple imaging modes** and objectives
- **Autofocus capabilities** with adaptive search
- **Image acquisition and processing**
- **Adaptive exposure control** for background acquisition (automatically adjusts to target intensities)
- **Polarizer calibration utilities** for PPM rotation stage (sine fitting and minima detection)
- **Socket-based server** for QuPath integration with real-time progress reporting

``` python
from smart_wsi_scanner.qupath import QuPathProject, QuPathScanner
from smart_wsi_scanner.config import ConfigManager, sp_position
from smart_wsi_scanner.smartpath import smartpath, init_hardware

# Load connections hardware:pycromanager/pymmcore/pymmcore-plus
microscope = MicroscopeHardware()
if not microscope.core: fail

# Configurations
config_manager = ConfigManager() # knows resources
microscope_settings = config_manager.load_config("config_CAMM.yml")

# Create a QuPath project (exist_ok=True)
project = QuPathProject(
    projects_folder_path=r"C:\Users\lociuser\Codes\MikeN\data\slides",
    sample_label="Example_Sample",
    scan_type="20x_bf_1",
    region="region_1"
)

# Initialize smartpath
sp = smartpath(microscope,microscope_settings) # has current state and settings

# positions passed or created
positions = QuPathScanner.generate_grid_positions()
#Get autofocus positions
af_indices, min_distance = QuPathScanner.get_autofocus_positions(positions)
# Scan positions
QuPathScanner.scan_positions()

```

## License

MIT License
