

import os
import re

self_filename = r'C:\Users\lociuser\Codes\smart-wsi-scanner\minimal_qupathrunner.py'
projectsFolderPath =  r'C:\Users\lociuser\Codes\MikeN\data\slides'
sampleLabel = 'First_Test'
scan_type = '4x_bf_4' 
tile_configuration_folder  = 'bounds'
qupath_project_folder = os.path.join(projectsFolderPath, sampleLabel, scan_type)

def read_tile_config_text(tile_config_path):
    coordinates = []
    with open(tile_config_path, 'r') as file:
        for line in file:
            # Extract coordinates using regular expression
            match = re.search(r'\((-?\d+\.?\d*), (-?\d+\.?\d*)\)', line)
            if match:
                x, y = map(float, match.groups())
                coordinates.append([x, y])
    return coordinates

tile_config_path = os.path.join(qupath_project_folder, tile_configuration_folder, "TileConfiguration.txt")
print(read_tile_config_text(tile_config_path))
