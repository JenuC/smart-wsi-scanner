import numpy as np
import warnings

# used for mm-pix.tags
from collections import OrderedDict
import matplotlib.pyplot as plt
from dataclasses import dataclass, field

# is_background and autofocus
from skimage import color

# autofocus
from skimage.filters import sobel

# from skimage.measure import shannon_entropy

# white balance
import scipy.interpolate


# flat-field
# from skimage.util import view_as_windows, crop, img_as_float, exposure

# background image
# import tifffile as tf

# metadata
import difflib
import pprint
from dataclasses import asdict, fields
from pycromanager import Core, Studio


def init_pycromanager():
    core = Core()
    studio = Studio()
    core.set_timeout_ms(20000)
    return core, studio


core, studio = init_pycromanager()


@dataclass
class _limits:
    low: float
    high: float

    def __post_init__(self):
        if self.low > self.high:
            self.low, self.high = self.high, self.low


@dataclass
class sp_stage_settings:
    xlimit: _limits = field(default=None)
    ylimit: _limits = field(default=None)
    zlimit: _limits = field(default=None)


@dataclass
class sp_camm_stage(sp_stage_settings):
    z_stage: str = field(default="ZStage:Z:32")
    f_stage: str = field(default="ZStage:F:32")
    xlimit: _limits = _limits(0, 40000.0)
    ylimit: _limits = _limits(0, 30000.0)
    zlimit: _limits = _limits(-10600.0, 100)
    flimit: _limits = _limits(-18000.0, 0)


@dataclass
class sp_objective_lens:
    """
    See https://www.microscope.healthcare.nikon.com/products/optics/selector
    """

    name: str
    magnification: float
    NA: float
    WD: float = field(default=None)
    description: str = field(default=None)
    manufacturer_id: str = field(default=None)


# TODO: camera features
@dataclass
class sp_detector:
    """Placeholder for detector variables used in operation
    : resolution, binning etc
    """

    width: int = field(default=None)
    height: int = field(default=None)


@dataclass
class sp_imaging_mode:
    name: str = field(default=None)
    pixelsize: float = field(default=None)


@dataclass
class sp_camm_imaging_mode(sp_imaging_mode):
    z: float = field(default=None)
    f: float = field(default=None)
    o: str = field(default=None)


class loci_instruments:
    def __init__(self):
        pass

    @property
    def lens_20x(self):
        return sp_objective_lens(
            name="20x",
            magnification=20,
            NA=0.75,
            WD=0.8,
            description="CFI Plan Apochromat Lambda D 20X, 0.75",
            manufacturer_id="MRD70270",
        )

    @property
    def lens_4x(self):
        return sp_objective_lens(
            name="4x",
            magnification=4,
            NA=0.2,
            WD=20,
            description="CFI Plan Apochromat Lambda D 4X",
            manufacturer_id="MRD70040",
        )

    @property
    def camera_QCAM(self):
        return sp_detector(1392, 1040)

    @property
    def camera_OSc(self):
        return sp_detector(512, 512)

    @property
    def stage_ASI(self):
        return sp_camm_stage()

    @property
    def CAMM_4X_BF(self):
        return sp_camm_imaging_mode("4x BrightField", 1.105, -9.2, -3200, "Position-2")

    @property
    def CAMM_20X_BF(self):
        return sp_camm_imaging_mode(
            "20x BrightField", 0.222, -10502, -13350, "Position-1"
        )

    @property
    def CAMM_20X_MPM(self):
        return sp_camm_imaging_mode("20x MPM", 1.105, -10160, -16000, "Position-1")
        # f used to be  bf-15800 shg-18500


camm_ = loci_instruments()


@dataclass
class sp_microscope_settings:
    stage: sp_stage_settings = field(default=None)
    lens: sp_objective_lens = field(default=None)
    detector: sp_detector = field(default=None)
    imaging_mode: sp_imaging_mode = field(default=None)


@dataclass
class sp_camm_settings(sp_microscope_settings):
    lamp: tuple = field(default=("LED-Dev1ao0", "Voltage"))
    obj_slider: tuple = field(default=("Turret:O:35", "Label"))
    slide_size: tuple = field(
        default=(40000.0, 20000.0)
    )  # (width, height) (70000, -20000)
    CAMM_20X_BF_XYoffset: tuple = field(
        default=(-600, 10)
    )  # 4x + this value to 20x // (-590, 74)
    CAMM_20X_MPM_XYoffset: tuple = field(
        default=(-580, -280)
    )  # 4x + this value to shg // (-580, -172)
    CAMM_4X_BF_lampintensity: float = field(default=4)
    CAMM_20X_BF_lampintensity: float = field(default=5)


camm = sp_camm_settings(
    stage=camm_.stage_ASI,
    lens=camm_.lens_4x,
    detector=camm_.camera_QCAM,
    imaging_mode=camm_.CAMM_4X_BF,
)


@dataclass
class sp_position:
    x: float
    y: float
    z: float = field(default=None)

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
        settings: sp_microscope_settings, position: sp_position
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
        current_pos = sp_position(
            core.get_x_position(), core.get_y_position(), core.get_position()
        )
        return current_pos

    @staticmethod
    def move_to_position(
        core: Core, position: sp_position, settings: sp_microscope_settings
    ):
        if not position.x:
            position.x = smartpath.get_current_position(core).x
        if not position.y:
            position.y = smartpath.get_current_position(core).y
        if not position.z:
            position.z = smartpath.get_current_position(core).z

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
    def snap(
        core: Core, flip_channel=True, background_correction=False, remove_alpha=True
    ):
        """Snaps an Image using MM Core and returns img,tags"""
        if core.is_sequence_running():
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
                    pixels = smartpath.background_correction(pixels)
            else:
                pixels = np.reshape(
                    tagged_image.pix, newshape=[tags["Height"], tags["Width"]]
                )
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
