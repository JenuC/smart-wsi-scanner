"""Example of using the QuPath integration module."""

from pathlib import Path
from smart_wsi_scanner.qupath import QuPathProject, QuPathScanner
from smart_wsi_scanner.config import ConfigManager, sp_position
from smart_wsi_scanner.smartpath import smartpath, init_pycromanager

def main():
    # Initialize Micro-Manager connection
    core, studio = init_pycromanager()
    if not core:
        print("Failed to initialize Micro-Manager connection")
        return

    # Initialize configuration
    config_manager = ConfigManager()
    camm_settings = config_manager.load_config("config_CAMM.yml")

    # Create a QuPath project
    project = QuPathProject(
        projects_folder_path=r"C:\Users\lociuser\Codes\MikeN\data\slides",
        sample_label="Example_Sample",
        scan_type="20x_bf_1",
        region="region_1"
    )

    # Create output directory if it doesn't exist
    project.path_output.mkdir(parents=True, exist_ok=True)

    # Initialize smartpath
    sp = smartpath(core)

    # Example 1: Generate a grid of positions
    start_pos = sp_position(x=1000.0, y=1000.0, z=0.0)
    positions = QuPathScanner.generate_grid_positions(
        n_x=3,  # 3 positions in x direction
        n_y=3,  # 3 positions in y direction
        start_pos=start_pos,
        settings=camm_settings
    )

    # Example 2: Get autofocus positions
    af_indices, min_distance = QuPathScanner.get_autofocus_positions(
        positions=positions,
        settings=camm_settings,
        n_tiles=1.5  # Autofocus every 1.5 tiles
    )

    # Example 3: Visualize autofocus locations
    QuPathScanner.visualize_autofocus_locations(
        positions=positions,
        settings=camm_settings,
        n_tiles=1.5
    )

    # Example 4: Write tile configuration
    QuPathScanner.write_tile_configuration(
        tile_config_path=project.path_tile_configuration,
        positions=positions,
        acquisition_id=project.acq_id,
        pixel_size=camm_settings.imaging_mode.pixelsize
    )

    # Example 5: Scan positions
    QuPathScanner.scan_positions(
        sp=sp,
        settings=camm_settings,
        save_folder=project.path_output,
        positions=positions,
        acquisition_id=project.acq_id,
        autofocus_indices=af_indices
    )

    # Example 6: Read tile configuration
    read_positions = QuPathScanner.read_tile_configuration(project.path_tile_configuration)
    print(f"Read {len(read_positions)} positions from tile configuration")

    # Example 7: Get distance-sorted positions
    sorted_positions = QuPathScanner.get_distance_sorted_positions(positions)
    print("Positions sorted by distance from left-bottom corner:")
    for idx, (pos, dist) in sorted_positions.items():
        print(f"Position {idx}: ({pos[0]:.1f}, {pos[1]:.1f}) - Distance: {dist:.1f}")

if __name__ == "__main__":
    main() 