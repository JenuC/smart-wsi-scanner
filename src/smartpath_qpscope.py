import pprint
from smartpath import sp_microscope_settings, sp_position, smartpath, Core
import numpy as np

# af_min_distance
import matplotlib
import matplotlib.pyplot as plt

# slide scan
import tifffile as tf
import os

# QuPath Helper
import re

# qupath_helper
from scipy.spatial.distance import cdist


import uuid
import pathlib


class smartpath_qpscope:
    def __init__(self):
        pass

    @staticmethod
    def uid():
        return uuid.uuid1().urn[9:]

    @staticmethod
    def read_TileConfiguration_coordinates(tile_config_path) -> np.array:
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
    def get_distance_sorted_xy_dict(positions):
        ## test using radial distance: fails because distance without moving center would fail
        left_bottom = np.argmin(np.array([x[0] ** 2 + x[1] ** 2 for x in positions]))
        xa = positions[left_bottom]
        distances = np.round(cdist([xa], positions).ravel(), 2)
        positions_d = {
            ix: (positions[ix], distances[ix]) for ix in range(len(distances))
        }
        positions_d = dict(sorted(positions_d.items(), key=lambda item: item[1][1]))
        return positions_d

    @staticmethod
    def pltn(arr, *args, **kwargs):
        x = arr[:, 0]
        y = arr[:, 1]
        print(arr)
        plt.plot(x, y, *args, **kwargs)

    @staticmethod
    def get_fov(camm: sp_microscope_settings) -> (float, float):
        """returns field of view in settings.pixelsize units
        fov_x, fov_y"""
        fov_y = camm.imaging_mode.pixelsize * camm.detector.height
        fov_x = camm.imaging_mode.pixelsize * camm.detector.width
        return fov_x, fov_y

    @staticmethod
    def get_dummy_coordinates(
        nX: int, nY: int, p1: sp_position, camm: sp_microscope_settings
    ):
        fov_x, fov_y = smartpath_qpscope.get_fov(camm)
        positions = []
        for k in range(nX):
            for j in range(nY):
                positions.append([p1.x + k * fov_x, p1.y + j * fov_y])
        positions = np.array(positions)
        return positions

    @staticmethod
    def get_autofocus_positions(
        positions: list, camm: sp_microscope_settings, ntiles: float
    ):
        # find distance in terms of field of view or number of tiles
        fov_x, fov_y = smartpath_qpscope.get_fov(camm)
        af_min_distance = cdist([[0, 0]], [[fov_x * ntiles, fov_y * ntiles]])[0][0]
        # print(af_min_distance)
        af_positions = []
        for ix, pos in enumerate(positions):
            if ix == 0:
                af_positions.append(0)
                af_xy_pos = positions[0]
            dist_to_last_af_xy_pos = cdist([af_xy_pos], [pos])[0][0]
            if dist_to_last_af_xy_pos > af_min_distance:
                af_positions.append(ix)
                af_xy_pos = pos
                print(ix, af_positions, pos, np.around(dist_to_last_af_xy_pos, 2))
        return af_positions, af_min_distance

    @staticmethod
    def visualize_autofocus_locations(positions, camm, ntiles=1.35):
        af_position_indices, af_min_distance = (
            smartpath_qpscope.get_autofocus_positions(positions, camm, ntiles)
        )
        ax = plt.subplot(111)
        for ix, pos in enumerate(positions):
            if ix in af_position_indices:
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
        ax.axis(lims)
        ax.set_aspect("equal")

    @staticmethod
    def scan_using_positions(
        sp: smartpath,
        camm: sp_microscope_settings,
        save_folder: str = None,
        positions: list = None,
        id1: str = "Tile",
        suffix_length: str = "06",
        core: Core = None,
        autofocus_indices: list = None,
    ):
        """
        saves tiles to save folder : flipped + whiteblances

        """

        starting_props = smartpath.get_device_properties(core)

        with open(os.path.join(save_folder, "MM2_DeviceProperties.txt"), "w") as fid:
            pprint.pprint(starting_props, stream=fid)

        for ix, pos in enumerate(positions):
            sp.move_to_position(core, sp_position(pos[0], pos[1]), camm)

            if autofocus_indices and ix in autofocus_indices:
                _ = sp.autofocus(core=core, settings=camm)
            img, tags = sp.snap(core)
            file_id = f"{id1}_{ix:{suffix_length}}"
            # file_id = f"{ix:{suffix_length}}"

            smartpath_qpscope.ome_writer(
                filename=os.path.join(save_folder, file_id + ".tif"),
                pixel_size_um=camm.imaging_mode.pixelsize,
                data=sp.white_balance(img),
            )

            current_props = sp.get_device_properties(core)
            metadata_change = sp.compare_dev_prop(current_props, starting_props)
            if metadata_change:
                with open(
                    os.path.join(save_folder, file_id + "_DPchanges.txt"), "w"
                ) as fid:
                    print(metadata_change, file=fid)
        with open(os.path.join(save_folder, "MM2_ImageTags.txt"), "w") as fid:
            pprint.pprint(smartpath_qpscope.format_imagetags(tags), stream=fid)

    @staticmethod
    def write_tileconfig(
        tileconfig_path: str = None,
        target_foldername: str = None,
        positions: list = None,  #:List,
        id1: str = "Tile",
        suffix_length: str = "06",
        pixel_size=1.0,
    ):
        if not tileconfig_path:
            tileconfig_path = os.path.join(target_foldername, "TileConfiguration.txt")

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

    @staticmethod
    def ome_writer(filename: str, pixel_size_um: float, data: np.array):
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


class qpscope_project:
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
