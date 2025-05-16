import sys
from pathlib import Path
from smart_wsi_scanner.smartpath import smartpath, init_pycromanager
from smart_wsi_scanner.config import ConfigManager, sp_position
from smart_wsi_scanner.hardware import PycromanagerHardware

def main():
    # Parse command line arguments or use defaults
    if len(sys.argv) == 5:
        _, yaml_file_path, projects_folder_path, sample_label, scan_type, region = sys.argv
    else:
        yaml_file_path = r'D:\2025QPSC\smartpath_configurations\config_PPM.yml'
        projects_folder_path = r"D:\2025QPSC\data"
        sample_label = "2"
        scan_type = "BF_10x_1"
        region = "bounds"
        
        #  D:\2025QPSC\smartpath_configurations\microscopes\config_PPM.yml,
        #  D:\2025QPSC\data,
        #  2,
        #  BF_10x_1,
        #  bounds

    # Initialize Micro-Manager connection
    core, studio = init_pycromanager()
    if not core:
        print("Failed to initialize Micro-Manager connection")
        return

    # Initialize hardware and settings
    config_manager = ConfigManager()
    camm_settings = config_manager.load_config(yaml_file_path)

    loci_rsc = r'resources\resources_LOCI.yml'
    import pathlib
    loci_rsc = pathlib.Path(yaml_file_path).parent / loci_rsc
    import os
    if os.path.exists(loci_rsc):
        loci_settings = config_manager.load_config(loci_rsc)
        print("loci-rsc found!")

    # Create hardware instance
    hardware = PycromanagerHardware(core, camm_settings, studio)

    # image, metadata = hardware.snap_image()
    # import matplotlib.pyplot as plt
    # plt.imshow(image)
    # plt.show()

    # Initialize smartpath
    sp = smartpath(core)

    # Create project paths
    project_path = Path(projects_folder_path) / sample_label
    output_path = project_path / scan_type
    tile_config_path = output_path / "TileConfiguration.txt"

    # Read tile configuration
    positions = []
    if tile_config_path.exists():
        with open(tile_config_path, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    parts = line.strip().split(',')
                    if len(parts) >= 3:
                        x, y, z = map(float, parts[:3])
                        positions.append(sp_position(x, y, z))

    AUTOFOCUS = False
    
    if AUTOFOCUS:
        # Get autofocus positions
        af_positions = []
        if positions:
            # Select every 3rd position for autofocus
            af_positions = positions[::3]

    # Scan using positions
    suffix_length = "06"
    for i, pos in enumerate(positions):
        # Move to position
        hardware.move_to_position(pos)
        
        if AUTOFOCUS:
            # Autofocus if this is an autofocus position
            if pos in af_positions:
                hardware.autofocus()
        
        # Snap image
        image, metadata = hardware.snap_image()
        
        # Save image
        image_path = output_path / f"{sample_label}_{scan_type}_{i:0{suffix_length}}.tif"
        # TODO: Add image saving logic here

    # Write updated tile configuration
    if os.path.exists(tile_config_path):
        with open(tile_config_path, 'w') as f:
            f.write("# Define the number of dimensions we are working on\n")
            f.write("dim = 3\n\n")
            f.write("# Define the image coordinates\n")
            for i, pos in enumerate(positions):
                f.write(f"{sample_label}_{scan_type}_{i:0{suffix_length}}.tif; ; ({pos.x}, {pos.y}, {pos.z})\n")

if __name__ == "__main__":
    main() 