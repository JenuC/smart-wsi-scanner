import numpy as np
import warnings

## used for mm-pix.tags
from collections import OrderedDict
import matplotlib.pyplot as plt
from dataclasses import dataclass, field

## is_background and autofocus
from skimage import color

# from skimage.util import view_as_windows, crop
# from skimage import img_as_float, exposure
## autofocus
from skimage.filters import sobel

# from skimage.measure import shannon_entropy
## white balance
import scipy.interpolate

## background image
import tifffile as tf

## metadata
import difflib
import pprint


from pycromanager import Core, Studio


def init_pycromanager():
    core = Core()
    studio = Studio()
    core.set_timeout_ms(20000)
    return core, studio


core, studio = init_pycromanager()


@dataclass
class sp_microscope_settings:
    x_range: list
    y_range: list
    z_range: list = field(default=None)
    f_range: list = field(default=None)
    focus_device: str = field(default="ZStage:Z:32")
    condenser_device: str = field(default="FStage:Z:32")
    lens_device: str = field(default="FStage:Z:32")


@dataclass
class sp_position:
    x: float
    y: float
    z: float = field(default=None)
    f: float = field(default=None)
    o: float = field(default=None)

    def __repr__(self):
        kws_values = [
            f"{key}={value:.1f}" for key, value in self.__dict__.items() if value
        ]
        kws_none = [
            f"{key}={value!r}" for key, value in self.__dict__.items() if not value
        ]
        kws = kws_values + kws_none
        return f"{type(self).__name__}({', '.join(kws)})"


class smartpath:

    def __init__(self, core):
        self.core = core

    def get_mm_metadata_string(self, pretty_print=False):

        device_dict = self.get_device_properties(self.core)
        if pretty_print:
            return pprint.pformat(device_dict)

        str = ""
        for device_name, prop_val in device_dict.items():
            for prop_name, values in prop_val.items():
                str = str + f"{device_name},{prop_name}: {values}" + "\n"
        return str

    @staticmethod
    def _mm_object_to_list(name):
        return [name.get(i) for i in range(name.size())]

    @staticmethod
    def get_device_properties(core, scope="used"):
        """
        get used/allowed properties in mm2-device manager
        as a dictionary
        """
        device_dict = {}
        for device_name in smartpath._mm_object_to_list(core.get_loaded_devices()):
            device_property_names = core.get_device_property_names(device_name)
            property_names = smartpath._mm_object_to_list(device_property_names)
            prop_dict = {}
            for prop in property_names:
                if scope == "allowed":
                    values = core.get_allowed_property_values(device_name, prop)
                    prop_dict.update({f"{prop}": smartpath._mm_object_to_list(values)})
                elif scope == "used":
                    values = core.get_property(device_name, prop)
                    prop_dict.update({f"{prop}": values})
                else:
                    warnings.warn(f" unknown metadata scope {scope} ")
            device_dict.update({f"{device_name}": prop_dict})
        return device_dict

    @staticmethod
    def _compare_dicts(d1, d2):
        """
        Adapted version from cpython unit-test
        https://stackoverflow.com/questions/12956957/print-diff-of-python-dictionaries
        https://github.com/python/cpython/blob/01fd68752e2d2d0a5f90ae8944ca35df0a5ddeaa/Lib/unittest/case.py#L1091
        """
        return "\n" + "\n".join(
            difflib.ndiff(
                pprint.pformat(d1).splitlines(), pprint.pformat(d2).splitlines()
            )
        )

    @staticmethod
    def compare_device_properties(dp1, dp2):
        if len(dp1.keys() ^ dp2.keys()) == 0:
            keys = dp1.keys()
        else:
            return -1
        for key in keys:
            if len(set(dp1[key].items()) ^ set(dp2[key].items())):
                compared_output = smartpath._compare_dicts(dp1[key], dp2[key])
        for ix, k in enumerate(compared_output.splitlines()):
            if k.startswith("?"):
                print(compared_output.splitlines()[ix - 1])
                print(k)

    @staticmethod
    def is_coordinate_in_range(settings, position) -> bool:

        _within_ylimit = _within_xlimit = False
        if settings.x_range[0] < position.x < settings.x_range[1]:
            _within_xlimit = True
        else:
            warnings.warn(f" {position.x} out of range X {settings.x_range}")
        if settings.y_range[0] < position.y < settings.y_range[1]:
            _within_ylimit = True
        else:
            warnings.warn(f" {position.y} out of range Y {settings.y_range}")

        if not position.z:
            return _within_xlimit and _within_ylimit
        if settings.z_range:
            _within_zlimit = False
            if settings.z_range[0] < position.z < settings.z_range[1]:
                _within_zlimit = True
            else:
                warnings.warn(f" {position.z} out of range Y {settings.z_range}")
            return _within_xlimit and _within_ylimit and _within_zlimit
        else:
            warnings.warn(f" Z range undefined : {settings.z_range}")
            return False

    @staticmethod
    def get_current_position(core):
        current_pos = sp_position(
            core.get_x_position(), core.get_y_position(), core.get_position()
        )
        return current_pos

    @staticmethod
    def move_to_position(
        position: sp_position, settings: sp_microscope_settings, core: Core
    ):

        if position.f or position.o:
            warnings.warn(" F and O stage movements are not implemented yet")
            return 0

        if not position.x:
            position.x = smartpath.get_current_position(core).x
        if not position.y:
            position.y = smartpath.get_current_position(core).y
        if not position.z:
            position.z = smartpath.get_current_position(core).z

        # check position in range
        if smartpath.is_coordinate_in_range(settings, position):

            # verify focus device: bcs user can change it to F mode
            if core.get_focus_device() != settings.focus_device:
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
    def snap(
        core: Core, flip_channel=True, background_correction=False, remove_alpha=True
    ):
        if core.is_sequence_running():
            studio.live().set_live_mode(False)
        core.snap_image()
        tagged_image = core.get_tagged_image()
        tags = OrderedDict(sorted(tagged_image.tags.items()))
        # QCAM
        if tags["Core-Camera"] == "QCamera":
            pixels = np.reshape(
                tagged_image.pix,
                newshape=[tags["Height"], tags["Width"], 4],
            )

            if remove_alpha:
                pixels = pixels[:, :, 0:3]
            if flip_channel:
                pixels = np.flip(pixels, 2)
            if background_correction:
                pixels = smartpath.background_correction(pixels)
        # OpenScan
        elif tags["Core-Camera"] == "OSc-LSM":
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
        core,
        settings: sp_microscope_settings,
        nsteps=5,
        search_range=45,
        interp_strength=100,
        interp_kind="quadratic",
        score_metric=sobel,
        pop_a_plot=False,
        move_stage_to_estimate=True,
    ):
        """
        score metric options : shannon_entropy, sobel
        """

        steps = np.linspace(0, search_range, nsteps) - (search_range / 2)
        current_pos = smartpath.get_current_position(core)
        z_steps = current_pos.z + steps

        try:
            scores = []
            for step_number in range(nsteps):
                new_pos = sp_position(
                    current_pos.x, current_pos.y, current_pos.z + steps[step_number]
                )
                smartpath.move_to_position(new_pos, settings, core)
                # print(smartpath.get_current_position(core))
                img, tags = smartpath.snap(core)
                img_gray = color.rgb2gray(img)
                score = score_metric(img_gray)
                if score.ndim == 2:
                    score = np.mean(score)
                scores.append(score)
            # interpolation
            interp_x = np.linspace(z_steps[0], z_steps[-1], nsteps * interp_strength)
            interp_y = scipy.interpolate.interp1d(z_steps, scores, kind=interp_kind)(
                interp_x
            )
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
                smartpath.move_to_position(new_pos, settings, core)
                return new_z
                # core.set_position(new_z)
        except Exception as e:
            smartpath.move_to_position(current_pos, settings, core)
            return e

    @staticmethod
    def white_balance(img=None, img_background=None, gain=1.0):

        white_balance_profiles = {
            "4x_binLi_Jul2021": [0.927, 1.0, 0.947],
            "20x_binLi_May2022": [1.0, 0.989, 0.803],
            "20x_jenu_Mar2024": [0.918, 1.0, 0.806],
        }
        if img_background:
            r, g, b = img_background.mean((0, 1))
            r_bkg, g_bkg, b_bkg = (r, g, b) / max(r, g, b)
        else:
            r_bkg, g_bkg, b_bkg = white_balance_profiles["20x_jenu_Mar2024"]

        imgx = img.astype(np.float64) * gain / [r_bkg, g_bkg, b_bkg]
        return np.clip(imgx, 0, 255).astype(np.uint8)
