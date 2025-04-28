import sys
from pathlib import Path
from smart_wsi_scanner.smartpath import smartpath, init_pycromanager
from smart_wsi_scanner.config import ConfigManager, sp_position
from smart_wsi_scanner.hardware import PycromanagerHardware

def main():
    # Parse command line arguments or use defaults
    if len(sys.argv) == 5:
        _, projects_folder_path, sample_label, scan_type, region = sys.argv
    else:
        projects_folder_path = r"C:\Users\lociuser\Codes\MikeN\data\slides"
        sample_label = "First_Test3"
        scan_type = "4x_bf_1"
        region = "bounds"

    # Initialize Micro-Manager connection
    core, studio = init_pycromanager()
    if not core:
        print("Failed to initialize Micro-Manager connection")
        return

    # Initialize hardware and settings
    config_manager = ConfigManager()
    camm_settings = config_manager.load_config("config_CAMM.yml")
    loci_settings = config_manager.load_config("resources_LOCI.yml")

    # Create hardware instance
    hardware = PycromanagerHardware(core, camm_settings, studio)

    # Initialize smartpath
    sp = smartpath(core)

    # Set initial imaging mode based on current objective
    current_objective = core.get_property(*camm_settings.obj_slider)
    if current_objective == camm_settings.imaging_mode.BF_20X.objective_position_label:
        camm_settings.imaging_mode = camm_settings.imaging_mode.BF_20X
    elif current_objective == camm_settings.imaging_mode.BF_4X.objective_position_label:
        camm_settings.imaging_mode = camm_settings.imaging_mode.BF_4X

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
        
        # Autofocus if this is an autofocus position
        if pos in af_positions:
            hardware.autofocus()
        
        # Snap image
        image, metadata = hardware.snap_image()
        
        # Save image
        image_path = output_path / f"{sample_label}_{scan_type}_{i:0{suffix_length}}.tif"
        # TODO: Add image saving logic here

    # Write updated tile configuration
    with open(tile_config_path, 'w') as f:
        f.write("# Define the number of dimensions we are working on\n")
        f.write("dim = 3\n\n")
        f.write("# Define the image coordinates\n")
        for i, pos in enumerate(positions):
            f.write(f"{sample_label}_{scan_type}_{i:0{suffix_length}}.tif; ; ({pos.x}, {pos.y}, {pos.z})\n")

if __name__ == "__main__":
    main() 