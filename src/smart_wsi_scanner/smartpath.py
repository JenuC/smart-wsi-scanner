# Standard library imports
import difflib  # Used by smartpath class for metadata comparison
import os  # Used by smartpath_qpscope class for file operations
import pathlib  # Used by qpscope_project class
import pprint  # Used by smartpath class for metadata formatting
import re  # Used by smartpath_qpscope class for coordinate parsing
import uuid  # Used by smartpath_qpscope and qpscope_project classes
import warnings  # Used by smartpath class for validation warnings
from collections import OrderedDict  # Used by smartpath class for image tags

# Third-party imports
import matplotlib  # Used by smartpath_qpscope class for visualization
import matplotlib.patches  # Used by smartpath_qpscope class for visualization
import matplotlib.pyplot as plt  # Used by smartpath class for autofocus plots and smartpath_qpscope for visualization
import numpy as np  # Used by all classes for array operations
import scipy.interpolate  # Used by smartpath class for autofocus interpolation
import tifffile as tf  # Used by smartpath_qpscope class for image writing
from pycromanager import Core, Studio  # Used by smartpath class for microscope control
from scipy.spatial.distance import cdist  # Used by smartpath_qpscope class for distance calculations
from skimage import color  # Used by smartpath class for image processing
from skimage.filters import sobel  # Used by smartpath class for autofocus

# Local imports
from .config import sp_microscope_settings, sp_position, sp_imaging_mode  # Used by all classes





class sp_metadata:
    def __init__(self, core: CMMCore | CMMCorePlus |Core):
        self.core = core

    def get_mm_metadata_string(self, pretty_print=False):
        device_dict = self.get_device_properties(self.core)
        if pretty_print:
            return pprint.pformat(device_dict)

        str_output = ""
        for device_name, prop_val in device_dict.items():
            for prop_name, values in prop_val.items():
                str_output = str_output + f"{device_name},{prop_name}: {values}" + "\n"
        return str_output

    @staticmethod
    def obj_2_list(name):
        return [name.get(i) for i in range(name.size())]

    @staticmethod
    def get_device_properties(core, scope="used"):
        """
        get used/allowed properties in mm2-device manager
        as a dictionary
        """
        device_dict = {}
        for device_name in smartpath.obj_2_list(core.get_loaded_devices()):
            device_property_names = core.get_device_property_names(device_name)
            property_names = smartpath.obj_2_list(device_property_names)
            prop_dict = {}
            for prop in property_names:
                if scope == "allowed":
                    values = core.get_allowed_property_values(device_name, prop)
                    prop_dict.update({f"{prop}": smartpath.obj_2_list(values)})
                elif scope == "used":
                    values = core.get_property(device_name, prop)
                    prop_dict.update({f"{prop}": values})
                else:
                    warnings.warn(f" unknown metadata scope {scope} ")
            device_dict.update({f"{device_name}": prop_dict})
        return device_dict

    @staticmethod
    def _compare_dicts(d1: dict, d2: dict) -> str:
        """Compare two dictionaries and return their differences in a readable format.

        Args:
            d1: First dictionary to compare
            d2: Second dictionary to compare

        Returns:
            A string containing the differences between the dictionaries,
            formatted with line-by-line diffs
        """


        d1_lines = pprint.pformat(d1).splitlines()
        d2_lines = pprint.pformat(d2).splitlines()

        diff = difflib.ndiff(d1_lines, d2_lines)
        return "\n" + "\n".join(diff)

    @staticmethod
    def compare_dev_prop(dp1: dict, dp2: dict) -> str:
        if dp1 != dp2:
            changes = ""
            if len(dp1.keys() ^ dp2.keys()) == 0:
                keys = dp1.keys()
            else:
                print("Devices mismatch", dp1.keys() ^ dp2.keys())
                keys = list(set.intersection(set(dp1.keys()), set(dp2.keys())))
                changes = changes + "\n Device mismatch: ", dp1.keys() ^ dp2.keys()

            for key in keys:
                if len(set(dp1[key].items()) ^ set(dp2[key].items())) == 0:
                    pass
                else:
                    x = smartpath._compare_dicts(dp1[key], dp2[key])
                    for ix, k in enumerate(x.splitlines()):
                        if k.startswith("?"):
                            # print(x[ix-2:ix+1])
                            changes = changes + key
                            changes = changes + "\t" + (x.splitlines()[ix - 1]) + "\n"
            return changes
        else:
            return ""



class smartpath:
    """Main class for smart microscope path operations and image acquisition."""
    
    def __init__(self, core):
        self.core = core


    @staticmethod
    def is_coordinate_in_range(
        settings, position  #: sp_microscope_settings,  #: sp_position
    ) -> bool:
        _within_ylimit = _within_xlimit = False

        if settings.stage.xlimit.low < position.x < settings.stage.xlimit.high:
            _within_xlimit = True
        else:
            warnings.warn(f" {position.x} out of range X {settings.stage.xlimit}")

        if settings.stage.ylimit.low < position.y < settings.stage.ylimit.high:
            _within_ylimit = True
        else:
            warnings.warn(f" {position.y} out of range Y {settings.stage.ylimit}")

        if not position.z:
            return _within_xlimit and _within_ylimit

        if settings.stage.zlimit:
            _within_zlimit = False
            if settings.stage.zlimit.low < position.z < settings.stage.zlimit.high:
                _within_zlimit = True
            else:
                warnings.warn(f" {position.z} out of range Z {settings.stage.zlimit}")
            return _within_xlimit and _within_ylimit and _within_zlimit
        else:
            warnings.warn(f" Z range undefined : {settings.stage.zlimit}")
            return False

    @staticmethod
    def get_current_position(core: Core):
        current_pos = sp_position(core.get_x_position(), core.get_y_position(), core.get_position())
        return current_pos

    @staticmethod
    def move_to_position(core: Core, position: sp_position, settings: sp_microscope_settings):
        # Get current position and populate any missing coordinates
        current_position = smartpath.get_current_position(core)
        position.populate_missing(current_position)

        # check position in range
        if smartpath.is_coordinate_in_range(settings, position):
            # verify focus device: bcs user can leave it on F mode
            if core.get_focus_device() != settings.stage.z_stage:
                warnings.warn(f" Changing focus device {core.get_focus_device()}")
                core.set_focus_device(settings.focus_device)

            # movement
            core.set_position(position.z)
            core.set_xy_position(position.x, position.y)
            core.wait_for_device(core.get_xy_stage_device())
            core.wait_for_device(core.get_focus_device())

        else:
            print(" Movement Cancelled ")

    @staticmethod
    def swap_objective_lens(
        core: Core,
        camm: sp_microscope_settings,
        desired_imaging_mode: sp_imaging_mode,
    ):
        """ " 4x->20x moves O first, then Z
        and
        20x->4x moves z first"""

        current_slider_position = core.get_property(*camm.obj_slider)
        if desired_imaging_mode.objective_position_label != current_slider_position:

            if desired_imaging_mode.name.startswith("4X"):
                core.set_focus_device(camm.stage.z_stage)
                core.set_position(desired_imaging_mode.z)
                core.wait_for_device(camm.stage.z_stage)
                core.set_property(*camm.obj_slider, desired_imaging_mode.objective_position_label)
                core.set_focus_device(camm.stage.f_stage)
                core.set_position(desired_imaging_mode.f)
                core.wait_for_system()
            if desired_imaging_mode.name.startswith("20X"):
                core.set_property(*camm.obj_slider, desired_imaging_mode.objective_position_label)
                core.wait_for_device(camm.obj_slider[0])
                core.set_focus_device(camm.stage.z_stage)
                core.set_position(desired_imaging_mode.z)
                core.set_focus_device(camm.stage.f_stage)
                core.set_position(desired_imaging_mode.f)
                core.wait_for_system()

            core.set_focus_device(camm.stage.z_stage)
            camm.imaging_mode = desired_imaging_mode

    @staticmethod
    def snap(core: Core, flip_channel=True, background_correction=False, remove_alpha=True):
        """Snaps an Image using MM Core and returns img,tags"""
        if core.is_sequence_running():
            studio = get_studio()
            if studio is not None:
                studio.live().set_live_mode(False)
        core.snap_image()
        tagged_image = core.get_tagged_image()
        tags = OrderedDict(sorted(tagged_image.tags.items()))

        # TODO: make this detector agnostic using sp_detector + MM2 Core,Camera
        # QCAM
        # if tags["Core-Camera"] == "QCamera":
        #    if tags["QCamera-Color"] == "ON":
        if smartpath.get_device_properties(core)["Core"]["Camera"] == "QCamera":
            if smartpath.get_device_properties(core)["QCamera"]["Color"] == "ON":
                pixels = np.reshape(
                    tagged_image.pix,
                    newshape=[tags["Height"], tags["Width"], 4],
                )

                if remove_alpha:
                    pixels = pixels[:, :, 0:3]
                if flip_channel:
                    pixels = np.flip(pixels, 2)
                if background_correction:
                    pass
                    # pixels = smartpath.background_correction(pixels)

            else:
                pixels = np.reshape(tagged_image.pix, newshape=[tags["Height"], tags["Width"]])
        # OpenScan
        elif smartpath.get_device_properties(core)["Core"]["Camera"] == "OSc-LSM":
            # tags["Core-Camera"] == "OSc-LSM":
            pixels = np.reshape(
                tagged_image.pix,
                newshape=[tags["Height"], tags["Width"]],
            )
        else:
            print(f"Capture Failed: SP doesn't recognize : {tags['Core-Camera']=}")
            return None, tags

        return pixels, tags

    @staticmethod
    def autofocus(
        core: Core,
        settings: sp_microscope_settings,
        n_steps=5,
        search_range=45,
        interp_strength=100,
        interp_kind="quadratic",
        score_metric=sobel,
        pop_a_plot=False,
        move_stage_to_estimate=True,
    ) -> float:
        """
        score metric options : shannon_entropy, sobel
        """

        steps = np.linspace(0, search_range, n_steps) - (search_range / 2)
        current_pos = smartpath.get_current_position(core)
        z_steps = current_pos.z + steps

        try:
            scores = []
            for step_number in range(n_steps):
                new_pos = sp_position(
                    current_pos.x, current_pos.y, current_pos.z + steps[step_number]
                )
                smartpath.move_to_position(core, new_pos, settings)
                # print(smartpath.get_current_position(core))
                img, tags = smartpath.snap(core)
                img_gray = color.rgb2gray(img)
                score = score_metric(img_gray)
                if score.ndim == 2:
                    score = np.mean(score)
                scores.append(score)

            # interpolation
            interp_x = np.linspace(z_steps[0], z_steps[-1], n_steps * interp_strength)
            interp_y = scipy.interpolate.interp1d(z_steps, scores, kind=interp_kind)(interp_x)
            new_z = interp_x[np.argmax(interp_y)]

            if pop_a_plot:
                plt.figure()
                plt.bar(z_steps, scores)
                plt.plot(interp_x, interp_y, "k")
                plt.plot(interp_x[np.argmax(interp_y)], interp_y.max(), "or")
                plt.xlabel("Z-axis")
                plt.title(f"X,Y = ({current_pos.x:.1f} , {current_pos.y:.1f})")

            if move_stage_to_estimate:
                new_pos = current_pos
                new_pos.z = new_z
                smartpath.move_to_position(core, new_pos, settings)
                return new_z
                # core.set_position(new_z)
        except Exception as e:
            smartpath.move_to_position(core, current_pos, settings)
            raise e

    @staticmethod
    def white_balance(img=None, background_image=None, gain=1.0):
        white_balance_profiles = {
            "4x_binLi_Jul2021": [0.927, 1.0, 0.947],
            "20x_binLi_May2022": [1.0, 0.989, 0.803],
            "20x_jenu_Mar2024": [0.918, 1.0, 0.806],
        }
        if background_image:
            r, g, b = background_image.mean((0, 1))
            r1, g1, b1 = (r, g, b) / max(r, g, b)
        else:
            r1, g1, b1 = white_balance_profiles["20x_jenu_Mar2024"]

        img_wb = img.astype(np.float64) * gain / [r1, g1, b1]
        return np.clip(img_wb, 0, 255).astype(np.uint8)


# ============================================================================
# smartpath_qpscope Class - QuPath Integration and Scanning Functionality
# ============================================================================

class smartpath_qpscope:
    """Class for QuPath integration and advanced scanning functionality."""
    
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
        positions_d = {ix: (positions[ix], distances[ix]) for ix in range(len(distances))}
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
    def get_dummy_coordinates(nX: int, nY: int, p1: sp_position, camm: sp_microscope_settings):
        fov_x, fov_y = smartpath_qpscope.get_fov(camm)
        positions = []
        for k in range(nX):
            for j in range(nY):
                positions.append([p1.x + k * fov_x, p1.y + j * fov_y])
        positions = np.array(positions)
        return positions

    @staticmethod
    def get_autofocus_positions(positions: list, camm: sp_microscope_settings, ntiles: float):
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
        af_position_indices, af_min_distance = smartpath_qpscope.get_autofocus_positions(
            positions, camm, ntiles
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
                with open(os.path.join(save_folder, file_id + "_DPchanges.txt"), "w") as fid:
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


# ============================================================================
# qpscope_project Class - Project Management
# ============================================================================

class qpscope_project:
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
