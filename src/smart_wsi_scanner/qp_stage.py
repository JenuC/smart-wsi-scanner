from smart_wsi_scanner.smartpath import init_pycromanager
from smart_wsi_scanner.config import ConfigManager, sp_position
from smart_wsi_scanner.hardware import PycromanagerHardware
import argparse
import sys
import numpy as np
from skimage import img_as_ubyte, exposure

core, studio = init_pycromanager()
config_manager = ConfigManager()
if not core:
    print("Failed to initialize Micro-Manager connection")
ppm_settings = config_manager.get_config('config_PPM')
hardware = PycromanagerHardware(core, ppm_settings, studio)
brushless = "KBD101_Thor_Rotation"
current_position_xyz = hardware.get_current_position()

def get_stageXY():
    print(f'{current_position_xyz.x,current_position_xyz.y}')
        
def get_stageZ():
    print(current_position_xyz.z)

def get_position():
    print(hardware.get_current_position())
    
def move_stageXY():
    parser = argparse.ArgumentParser(description='Move XYZ stage')
    
    # All arguments use flags and are not positional
    parser.add_argument('-x', '--x', type=float, required=True, help='X position')
    parser.add_argument('-y', '--y', type=float, required=True, help='Y position')
    parser.add_argument('-z', '--z', type=float, required=False, help='Z position (optional)')

    args = parser.parse_args()

    pos_kwargs = {'x': args.x, 'y': args.y}
    if args.z is not None:
        pos_kwargs['z'] = args.z

    hardware.move_to_position(sp_position(**pos_kwargs))
    print(hardware.get_current_position())
    
def move_stageZ():
    parser = argparse.ArgumentParser(description='Move Z stage')
    parser.add_argument('-z', '--z', type=float, required=True, help='Z position')
    args = parser.parse_args()
    
    hardware.move_to_position(sp_position(z=args.z))
    print(hardware.get_current_position())
    
## Kinesis control for rotational stage for PPM
# TODO: need to move some of this to smartpath?   
def ppm_to_thor(angle):
    return (-2*angle + 276)

def thor_to_ppm(kinesis_pos):
    return (276 - kinesis_pos) / 2

def get_stageR():
    kinesis_pos = core.get_position(brushless)
    print(f'{thor_to_ppm(kinesis_pos):.2f}')
    
def move_stageR():
    """Move rotation stage to specified angle."""
    parser = argparse.ArgumentParser(description='Move rotation stage')
    parser.add_argument('angle', type=float, help='Rotation angle in degrees')
    args = parser.parse_args(sys.argv[2:])
    
    newAngle = ppm_to_thor(args.angle)
    core.set_position(brushless, newAngle)
    core.wait_for_device(brushless)
    get_stageR()
