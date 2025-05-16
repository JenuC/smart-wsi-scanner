from smart_wsi_scanner.config import ConfigManager, sp_position
from smart_wsi_scanner.hardware import PycromanagerHardware, PymmcoreHardware
from pymmcore import CMMCore
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

from smart_wsi_scanner.smartpath import is_mm_running
if is_mm_running():
    print("Micro-Manager is running. Pymmcore cant initialize")
    sys.exit(1)
    
# Initialize core
core = CMMCore()

## A Demo #"C:\Program Files\Micro-Manager-2.0\MMConfig_demo.cfg"
#mm_dir = r"C:\Program Files\Micro-Manager-2.0"
#core.loadSystemConfiguration(os.path.join(mm_dir, "MMConfig_demo.cfg"))

## B nightly on BOCK_00124_ 
## Mike's BGRB-correction + CFA-OFF loaded config
#"C:\Program Files\Micro-Manager-2.0\PPM-20241016-colorbalance-v3.cfg"
mm_dir =r'C:\Program Files\Micro-Manager-2.0-20250418'
core.setDeviceAdapterSearchPaths([mm_dir])
# config is on another folder!
config_location = r'C:\Program Files\Micro-Manager-2.0\PPM-20241016-colorbalance-v3.cfg'
core.loadSystemConfiguration(config_location)

print(core.getPosition())
print(core.getFocusDevice())
