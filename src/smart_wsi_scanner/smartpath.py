import numpy as np
import warnings

# used for mm-pix.tags
from collections import OrderedDict
import matplotlib.pyplot as plt


# is_background and autofocus
from skimage import color

# autofocus
from skimage.filters import sobel

# from skimage.measure import shannon_entropy

# white balance
import scipy.interpolate
from .config import sp_microscope_settings, sp_position, sp_imaging_mode

# flat-field
# from skimage.util import view_as_windows, crop, img_as_float, exposure

# background image
# import tifffile as tf

# metadata
import difflib
import pprint

from pycromanager import Core, Studio


def is_mm_running() -> bool:
    """Check if Micro-Manager is running as a Windows executable."""
    import platform
    import psutil

    if platform.system() != "Windows":
        return False

    for proc in psutil.process_iter(["name"]):
        try:
            if proc.exe().find("Micro-Manager") > 0:
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return False


def init_pycromanager():
    """Initialize Pycromanager connection."""
    if not is_mm_running():
        print("Micro-Manager is not running. Please start Micro-Manager before initializing.")
        return None, None
    core = Core()
    studio = Studio()
    core.set_timeout_ms(20000)
    return core, studio


# Initialize core and studio only when needed
_core = None
_studio = None


def get_core():
    """Get or initialize the core instance."""
    global _core, _studio
    if _core is None:
        _core, _studio = init_pycromanager()
    return _core


def get_studio():
    """Get or initialize the studio instance."""
    global _core, _studio
    if _studio is None:
        _core, _studio = init_pycromanager()
    return _studio


class smartpath:
    def __init__(self, core):
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
        import difflib
        import pprint

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
