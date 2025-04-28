"""QuPath integration module for smart-wsi-scanner."""

import uuid
import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import tifffile as tf
from scipy.spatial.distance import cdist

from .config import sp_microscope_settings, sp_position
from .smartpath import smartpath
from pycromanager import Core

class QuPathProject:
    """Represents a QuPath project for slide scanning."""
    
    def __init__(
        self,
        projects_folder_path: str = r"C:\Users\lociuser\Codes\MikeN\data\slides",
        sample_label: str = "2024_04_09_4",
        scan_type: str = "20x_bf_2",
        region: str = "1479_4696",
        tile_config: str = "TileConfiguration.txt",
    ):
        """Initialize a QuPath project.
        
        Args:
            projects_folder_path: Path to the projects folder
            sample_label: Label for the sample
            scan_type: Type of scan (e.g., "20x_bf_2")
            region: Region identifier
            tile_config: Name of the tile configuration file
        """
        self.path_tile_configuration = Path(projects_folder_path) / sample_label / scan_type / region / tile_config
        if self.path_tile_configuration.exists():
            self.path_qp_project = Path(projects_folder_path) / sample_label
            self.path_output = Path(projects_folder_path) / sample_label / scan_type / region
            self.acq_id = f"{sample_label}_ST_{scan_type}"
        else:
            self.path_qp_project = Path("undefined")
            self.path_output = Path("undefined")
            self.acq_id = "undefined_ScanType_undefined"

    def __repr__(self) -> str:
        return (
            f"QuPath project: {self.path_qp_project}\n"
            f"TIF files: {self.path_output}\n"
            f"Acquisition ID: {self.acq_id}"
        )

class QuPathScanner:
    """Handles scanning operations with QuPath integration."""
    
    @staticmethod
    def generate_uid() -> str:
        """Generate a unique identifier."""
        return uuid.uuid1().urn[9:]

    @staticmethod
    def read_tile_configuration(tile_config_path: Path) -> np.ndarray:
        """Read coordinates from a tile configuration file.
        
        Args:
            tile_config_path: Path to the tile configuration file
            
        Returns:
            Array of coordinates
        """
        coordinates = []
        with open(tile_config_path, "r") as file:
            for line in file:
                match = re.search(r"\((-?\d+\.\d+), (-?\d+\.\d+)\)", line)
                if match:
                    x, y = map(float, match.groups())
                    coordinates.append([x, y])
        return np.array(coordinates)

    @staticmethod
    def get_distance_sorted_positions(positions: np.ndarray) -> Dict[int, Tuple[np.ndarray, float]]:
        """Sort positions by distance from the left-bottom position.
        
        Args:
            positions: Array of positions
            
        Returns:
            Dictionary mapping indices to (position, distance) tuples
        """
        left_bottom = np.argmin(np.array([x[0] ** 2 + x[1] ** 2 for x in positions]))
        xa = positions[left_bottom]
        distances = np.round(cdist([xa], positions).ravel(), 2)
        positions_d = {
            ix: (positions[ix], distances[ix]) for ix in range(len(distances))
        }
        return dict(sorted(positions_d.items(), key=lambda item: item[1][1]))

    @staticmethod
    def get_field_of_view(settings: sp_microscope_settings) -> Tuple[float, float]:
        """Calculate field of view in settings.pixelsize units.
        
        Args:
            settings: Microscope settings
            
        Returns:
            Tuple of (fov_x, fov_y)
        """
        fov_y = settings.imaging_mode.pixelsize * settings.detector.height
        fov_x = settings.imaging_mode.pixelsize * settings.detector.width
        return fov_x, fov_y

    @staticmethod
    def generate_grid_positions(
        n_x: int,
        n_y: int,
        start_pos: sp_position,
        settings: sp_microscope_settings
    ) -> np.ndarray:
        """Generate a grid of positions.
        
        Args:
            n_x: Number of positions in x direction
            n_y: Number of positions in y direction
            start_pos: Starting position
            settings: Microscope settings
            
        Returns:
            Array of positions
        """
        fov_x, fov_y = QuPathScanner.get_field_of_view(settings)
        positions = []
        for k in range(n_x):
            for j in range(n_y):
                positions.append([start_pos.x + k * fov_x, start_pos.y + j * fov_y])
        return np.array(positions)

    @staticmethod
    def get_autofocus_positions(
        positions: List[np.ndarray],
        settings: sp_microscope_settings,
        n_tiles: float
    ) -> Tuple[List[int], float]:
        """Determine positions for autofocus.
        
        Args:
            positions: List of positions
            settings: Microscope settings
            n_tiles: Number of tiles between autofocus points
            
        Returns:
            Tuple of (autofocus_indices, min_distance)
        """
        fov_x, fov_y = QuPathScanner.get_field_of_view(settings)
        af_min_distance = cdist([[0, 0]], [[fov_x * n_tiles, fov_y * n_tiles]])[0][0]
        
        af_positions = []
        af_xy_pos = positions[0]
        af_positions.append(0)
        
        for ix, pos in enumerate(positions):
            dist_to_last_af = cdist([af_xy_pos], [pos])[0][0]
            if dist_to_last_af > af_min_distance:
                af_positions.append(ix)
                af_xy_pos = pos
                
        return af_positions, af_min_distance

    @staticmethod
    def visualize_autofocus_locations(
        positions: np.ndarray,
        settings: sp_microscope_settings,
        n_tiles: float = 1.35
    ) -> None:
        """Visualize autofocus locations.
        
        Args:
            positions: Array of positions
            settings: Microscope settings
            n_tiles: Number of tiles between autofocus points
        """
        af_indices, af_min_distance = QuPathScanner.get_autofocus_positions(
            positions, settings, n_tiles
        )
        
        ax = plt.subplot(111)
        for ix, pos in enumerate(positions):
            if ix in af_indices:
                circle = matplotlib.patches.Circle(
                    (pos[0], pos[1]),
                    af_min_distance,
                    fill=False,
                )
                ax.add_artist(circle)
                ax.plot(pos[0], pos[1], "s")
            else:
                ax.plot(pos[0], pos[1], "o", markeredgecolor="k")
                
        xstd = 5
        lims = np.array([
            [np.mean(positions, 0) - (np.std(positions, 0) * xstd)],
            [np.mean(positions, 0) + np.std(positions, 0) * xstd],
        ]).T.ravel()
        ax.axis(lims)
        ax.set_aspect("equal")

    @staticmethod
    def scan_positions(
        sp: smartpath,
        settings: sp_microscope_settings,
        save_folder: Path,
        positions: List[np.ndarray],
        acquisition_id: str = "Tile",
        suffix_length: str = "06",
        core: Optional[Core] = None,
        autofocus_indices: Optional[List[int]] = None,
    ) -> None:
        """Scan positions and save images.
        
        Args:
            sp: Smartpath instance
            settings: Microscope settings
            save_folder: Folder to save images
            positions: List of positions to scan
            acquisition_id: ID for the acquisition
            suffix_length: Length of the suffix for file names
            core: Pycromanager core instance
            autofocus_indices: Indices of positions for autofocus
        """
        if not core:
            core = sp.core
            
        starting_props = sp.get_device_properties(core)
        props_file = save_folder / "MM2_DeviceProperties.txt"
        with open(props_file, "w") as f:
            import pprint
            pprint.pprint(starting_props, stream=f)
            
        for ix, pos in enumerate(positions):
            sp.move_to_position(core, sp_position(pos[0], pos[1]), settings)
            
            if autofocus_indices and ix in autofocus_indices:
                _ = sp.autofocus(core=core, settings=settings)
                
            img, tags = sp.snap(core)
            file_id = f"{acquisition_id}_{ix:{suffix_length}}"
            
            QuPathScanner.save_image(
                filename=save_folder / f"{file_id}.tif",
                pixel_size_um=settings.imaging_mode.pixelsize,
                data=np.flipud(sp.white_balance(img)),
            )
            
            current_props = sp.get_device_properties(core)
            metadata_change = sp.compare_dev_prop(current_props, starting_props)
            if metadata_change:
                with open(save_folder / f"{file_id}_DPchanges.txt", "w") as f:
                    print(metadata_change, file=f)
                    
        with open(save_folder / "MM2_ImageTags.txt", "w") as f:
            import pprint
            pprint.pprint(QuPathScanner.format_image_tags(tags), stream=f)

    @staticmethod
    def write_tile_configuration(
        tile_config_path: Path,
        positions: List[np.ndarray],
        acquisition_id: str = "Tile",
        suffix_length: str = "06",
        pixel_size: float = 1.0,
    ) -> None:
        """Write tile configuration file.
        
        Args:
            tile_config_path: Path to write the configuration
            positions: List of positions
            acquisition_id: ID for the acquisition
            suffix_length: Length of the suffix for file names
            pixel_size: Size of pixels
        """
        with open(tile_config_path, "w") as f:
            print("dim = 2", file=f)
            for ix, pos in enumerate(positions):
                file_id = f"{acquisition_id}_{ix:{suffix_length}}"
                x, y = pos
                print(
                    f"{file_id}.tif; ; ({x/pixel_size:.3f}, {y/pixel_size:.3f})",
                    file=f,
                )

    @staticmethod
    def save_image(
        filename: Path,
        pixel_size_um: float,
        data: np.ndarray,
    ) -> None:
        """Save image with OME-TIFF metadata.
        
        Args:
            filename: Path to save the image
            pixel_size_um: Size of pixels in micrometers
            data: Image data
        """
        with tf.TiffWriter(filename) as tif:
            options = {
                "photometric": "rgb",
                "compression": "jpeg",
                "resolutionunit": "CENTIMETER",
                "maxworkers": 2,
            }
            tif.write(
                data,
                resolution=(1e4 / pixel_size_um, 1e4 / pixel_size_um),
                **options,
            )

    @staticmethod
    def format_image_tags(tags: Dict) -> Dict:
        """Format image tags into a nested dictionary.
        
        Args:
            tags: Dictionary of tags
            
        Returns:
            Nested dictionary of formatted tags
        """
        formatted = {}
        for k in set(key.split("-")[0] for key in tags):
            formatted[k] = {
                key: tags[key] for key in tags if key.startswith(k)
            }
        return formatted 