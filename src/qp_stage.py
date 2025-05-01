
from smart_wsi_scanner.smartpath import init_pycromanager
from smart_wsi_scanner.config import ConfigManager, sp_position
from smart_wsi_scanner.hardware import PycromanagerHardware
core, studio = init_pycromanager()
config_manager = ConfigManager()
if not core:
    print("Failed to initialize Micro-Manager connection")
ppm_settings = config_manager.get_config('config_PPM')
hardware = PycromanagerHardware(core, ppm_settings, studio)

def get_stageXY():
    print(hardware.get_current_position())
        
def get_stageZ():
    print(hardware.get_current_position())
    
def get_stageP():
    print(hardware.get_current_position())
    
def move_stageXY():
    print(hardware.get_current_position())
    
def move_stageZ():
    print(hardware.get_current_position())
    
def move_stageP():
    print(hardware.get_current_position())

    
    
