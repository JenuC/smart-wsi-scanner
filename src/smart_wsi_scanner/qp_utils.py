import pathlib
import re
from typing import List, Tuple, Optional
from .config import sp_position
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import matplotlib.patches
import tifffile as tf
import uuid

## read tile configuration file ( two versions)


class TileConfigUtils:
    def __init__(self):
        pass

    @staticmethod
    def read_tile_config(tile_config_path: pathlib.Path, core) -> List[Tuple[sp_position, str]]:
        """Claude: Read tile positions + filename from a QuPath-generated TileConfiguration.txt file."""
        positions: List[Tuple[sp_position, str]] = []
        if tile_config_path.exists():
            with open(tile_config_path, "r") as f:
                for line in f:
                    pattern = r"^([\w\-\.]+); ; \(\s*([\-\d.]+),\s*([\-\d.]+)"
                    m = re.match(pattern, line)
                    if m:
                        filename = m.group(1)
                        x = float(m.group(2))
                        y = float(m.group(3))
                        z = core.get_position()
                        positions.append((sp_position(x, y, z), filename))
        return positions

    @staticmethod
    def read_TileConfiguration_coordinates(tile_config_path) -> np.ndarray:
        """Read tile XY coordinates from a TileConfiguration.txt file."""
        coordinates = []
        with open(tile_config_path, "r") as file:
            for line in file:
                # Extract coordinates using regular expression
                match = re.search(r"\((-?\d+\.\d+), (-?\d+\.\d+)\)", line)
                if match:
                    x, y = map(float, match.groups())
                    coordinates.append([x, y])
        return np.array(coordinates)

    @staticmethod
    def write_tileconfig(
        tileconfig_path: Optional[str] = None,
        target_foldername: Optional[str] = None,
        positions: Optional[list] = None,  #:List,
        id1: str = "Tile",
        suffix_length: str = "06",
        pixel_size=1.0,
    ):
        if not tileconfig_path and target_foldername is not None:
            target_folder_path = pathlib.Path(target_foldername)
            tileconfig_path = str(target_folder_path / "TileConfiguration.txt")

        if tileconfig_path is not None and positions is not None:
            with open(tileconfig_path, "w") as text_file:
                print("dim = {}".format(2), file=text_file)
                for ix, pos in enumerate(positions):
                    file_id = f"{id1}_{ix:{suffix_length}}"
                    # file_id = f"{ix:{suffix_length}}"
                    x, y = pos
                    print(
                        f"{file_id}.tif; ; ({x/ pixel_size:.3f}, {y/ pixel_size:.3f})",
                        file=text_file,
                    )


class AutofocusUtils:
    def __init__(self):
        pass

    ## autofocus positions
    @staticmethod
    def get_distance_sorted_xy_dict(positions):
        ## test using radial distance: fails because distance without moving center would fail
        left_bottom = np.argmin(np.array([x[0] ** 2 + x[1] ** 2 for x in positions]))
        xa = positions[left_bottom]
        distances = np.round(cdist([xa], positions).ravel(), 2)
        positions_d = {ix: (positions[ix], distances[ix]) for ix in range(len(distances))}
        positions_d = dict(sorted(positions_d.items(), key=lambda item: item[1][1]))
        return positions_d

    @staticmethod
    def get_autofocus_positions(fov, positions: list[tuple[float, float]], ntiles: float):

        fov_x, fov_y = fov

        # Compute the minimum required distance between autofocus positions,
        af_min_distance = cdist([[0, 0]], [[fov_x * ntiles, fov_y * ntiles]])[0][0]

        # for each tile, if dist is higher, perform autofocus
        af_positions = []
        af_xy_pos = positions[0] if positions else None
        for ix, pos in enumerate(positions):
            if ix == 0:
                # Always autofocus at the first position
                af_positions.append(0)
                af_xy_pos = positions[0]
                dist_to_last_af_xy_pos = 0
            else:
                # Calculate distance from last AF position if both points are valid
                if af_xy_pos is not None and pos is not None:
                    dist_to_last_af_xy_pos = cdist([af_xy_pos], [pos])[0][0]
                else:
                    dist_to_last_af_xy_pos = 0
            # If we've moved more than the AF minimum distance, add new AF point
            if dist_to_last_af_xy_pos > af_min_distance:
                af_positions.append(ix)
                af_xy_pos = pos  # Update last autofocus position

                # Optional debug print
                # print(ix, af_positions, pos, np.around(dist_to_last_af_xy_pos, 2))

        return af_positions, af_min_distance

    @staticmethod
    def visualize_autofocus_locations(fov, positions, ntiles=1.35):
        af_positions, af_min_distance = AutofocusUtils.get_autofocus_positions(
            fov, positions, ntiles
        )
        ax = plt.subplot(111)
        for ix, pos in enumerate(positions):
            if ix in af_positions:
                crc = matplotlib.patches.Circle(
                    (pos[0], pos[1]),
                    af_min_distance,
                    # edgecolor='k',
                    fill=False,
                )
                ax.add_artist(crc)
                ax.plot(pos[0], pos[1], "s")
            else:
                ax.plot(pos[0], pos[1], "o", markeredgecolor="k")

        # ax.axis([17000,20000,13400,15500])
        xstd = 5
        lims = np.array(
            [
                [np.mean(positions, 0) - (np.std(positions, 0) * xstd)],
                [np.mean(positions, 0) + np.std(positions, 0) * xstd],
            ]
        ).T.ravel()
        ax.axis(tuple(lims))
        ax.set_aspect("equal")

        ax.set_title(f"Autofocus positions with {ntiles} tiles distance")
        ax.set_xlabel("X position (um)")
        ax.set_ylabel("Y position (um)")
        plt.show()
        return af_positions, af_min_distance


class TifWriterUtils:
    def __init__(self):
        pass

    ## OME-TIFF writing and metadata formatting
    @staticmethod
    def ome_writer(filename: str, pixel_size_um: float, data: np.ndarray):
        with tf.TiffWriter(
            filename,
            # bigtiff=True
        ) as tif:
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
    def format_imagetags(tags: dict):
        dx = {}
        for k in set([key.split("-")[0] for key in tags]):
            dx.update({k: {key: tags[key] for key in tags if key.startswith(k)}})
        return dx


class QuPathProject:
    """Class for managing QuPath project structure and file paths."""

    def __init__(
        self,
        projectsFolderPath: str = r"C:\Users\lociuser\Codes\MikeN\data\slides",
        sampleLabel: str = "2024_04_09_4",
        scan_type: str = "20x_bf_2",
        region: str = "1479_4696",
        tile_config: str = "TileConfiguration.txt",
    ):
        self.path_tile_configuration = pathlib.Path(
            projectsFolderPath, sampleLabel, scan_type, region, tile_config
        )
        if self.path_tile_configuration.exists():
            self.path_qp_project = pathlib.Path(projectsFolderPath, sampleLabel)
            self.path_output = pathlib.Path(
                projectsFolderPath, sampleLabel, scan_type, region
            )  # "data/acquisition"
            self.acq_id = sampleLabel + "_ST_" + scan_type
        else:
            self.path_qp_project = "undefined"
            self.path_output = "undefined"
            self.acq_id = "undefined" + "_ScanType_" + "undefined"

    @staticmethod
    def uid():
        return uuid.uuid1().urn[9:]

    def __repr__(self):
        return f"qupath project :{self.path_qp_project} \n tif files : {self.path_output} \n acq_id:{self.acq_id}"
