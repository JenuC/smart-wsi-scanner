import difflib  # Used by smartpath class for metadata comparison
import os  # Used by smartpath_qpscope class for file operations
import pathlib  # Used by qpscope_project class
import pprint  # Used by smartpath class for metadata formatting
import warnings  # Used by smartpath class for validation warnings
from collections import OrderedDict  # Used by smartpath class for image tags


from pycromanager import Core, Studio
from .hardware import MicroscopeHardware, is_mm_running, is_coordinate_in_range
from .config import sp_microscope_settings, sp_position, sp_imaging_mode

import numpy as np  #  image processing on tagged_image.pix
import skimage.color
import skimage.filters  # Used by smartpath class for autofocus
import scipy.interpolate  # Used by smartpath class for autofocus interpolation
import matplotlib.pyplot as plt  # Used by smartpath class for autofocus plots


def obj_2_list(name):
    return [name.get(i) for i in range(name.size())]


def init_pycromanager():
    """Initialize Pycromanager connection."""
    if not is_mm_running():
        print("Micro-Manager is not running. Please start Micro-Manager before initializing.")
        return None, None
    core = Core()
    studio = Studio()
    core.set_timeout_ms(20000)  # type: ignore
    return core, studio


def ppm_psgticks_to_thor(bi_angle: float) -> float:
    """Convert PPM angle (in degrees) to Thor rotation stage position."""
    return -2 * bi_angle + 276


def ppm_thor_to_psgticks(kinesis_pos: float) -> float:
    """Convert Thor rotation stage position to PPM angle (in degrees)."""
    return (276 - kinesis_pos) / 2


class PycromanagerHardware(MicroscopeHardware):
    """Implementation for Pycromanager-based microscopes."""

    def __init__(self, core: Core, studio: Studio, settings: sp_microscope_settings):
        self.core = core
        self.studio = studio
        self.settings = settings
        print(self.settings.microscope)
        if self.settings.microscope.name == "PPM":
            self.set_psg_ticks = self._ppm_set_psgticks
            self.get_psg_ticks = self._ppm_get_psgticks
        if settings.microscope.name == "CAMM":
            self.swap_objective_lens = self._camm_swap_objective_lens

    def move_to_position(self, position: sp_position) -> None:
        # Get current position and populate any missing coordinates
        current_position = self.get_current_position()
        position.populate_missing(current_position)

        if not is_coordinate_in_range(self.settings, position):
            raise ValueError("Position out of range")

        if self.core.get_focus_device() != self.settings.stage.z_stage:  # type: ignore
            self.core.set_focus_device(self.settings.focus_device)  # type: ignore

        self.core.set_position(position.z)  # type: ignore
        self.core.set_xy_position(position.x, position.y)  # type: ignore
        self.core.wait_for_device(self.core.get_xy_stage_device())  # type: ignore
        self.core.wait_for_device(self.core.get_focus_device())  # type: ignore

    def get_current_position(self) -> sp_position:
        return sp_position(
            self.core.get_x_position(), self.core.get_y_position(), self.core.get_position()  # type: ignore
        )

    def snap(self, background_correction=False, remove_alpha=True):
        """Snaps an Image using MM Core and returns img,tags"""
        if self.core.is_sequence_running() and self.studio is not None:  # type: ignore
            self.studio.live().set_live_mode(False)  # type: ignore
        self.core.snap_image()  # type: ignore

        tagged_image = self.core.get_tagged_image()  # type: ignore
        ## TODO : check if ordering helps in presentation?
        tags = OrderedDict(sorted(tagged_image.tags.items()))
        ## tags = tagged_image.tags

        pixels = tagged_image.pix
        total_pixels = pixels.shape[0]
        height, width = tags["Height"], tags["Width"]
        assert (total_pixels % (height * width)) == 0
        nchannels = total_pixels // (height * width)
        if nchannels > 1:
            pixels = pixels.reshape(height, width, nchannels)
        else:
            pixels = pixels.reshape(height, width)

        camera = self.get_device_properties()["Core"]["Camera"]
        if camera in ["QCamera", "MicroPublisher6"]:
            # flip BGRA to ARGB
            pixels = pixels[:, :, ::-1]
            if self.get_device_properties()[camera]["Color"] == "ON":
                if (camera == "MicroPublisher6") and (remove_alpha):
                    pixels = pixels[:, :, 1:]  # ARGB the alpha-channel is all zeros by default?
                    ## TODO verify if QCamera is BGRA
                if background_correction:
                    pass

        elif camera == "OSc-LSM":
            pass
        else:
            print(f"Capture Failed: SP doesn't recognize : {tags['Core-Camera']=}")
            return None, None

        return pixels, tags

    def autofocus(
        self,
        n_steps=5,
        search_range=45,
        interp_strength=100,
        interp_kind="quadratic",
        score_metric=skimage.filters.sobel,
        pop_a_plot=False,
        move_stage_to_estimate=True,
    ) -> float:
        """
        score metric options : shannon_entropy, sobel
        """
        steps = np.linspace(0, search_range, n_steps) - (search_range / 2)
        current_pos = self.get_current_position()
        z_steps = current_pos.z + steps  # type: ignore
        try:
            scores = []
            for step_number in range(n_steps):
                new_pos = sp_position(
                    current_pos.x, current_pos.y, current_pos.z + steps[step_number]
                )
                self.move_to_position(new_pos)
                # print(smartpath.get_current_position(core))
                img, tags = self.snap()
                img_gray = skimage.color.rgb2gray(img)
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
                self.move_to_position(new_pos)
                return new_z
                # core.set_position(new_z)
        except Exception as e:
            print("Autofocus failed: ", e)
            self.move_to_position(current_pos)
            raise e

    def white_balance(self, img=None, background_image=None, gain=1.0, white_balance_profile=None):
        if white_balance_profile is None:
            # load from profile
            # white_balance_profile = self.settings.white_balance.ppm.uncrossed
            white_balance_profile = self.settings.white_balance.default.default

        if img is None:
            raise ValueError("Input image 'img' must not be None for white balancing.")

        if background_image is not None:
            r, g, b = background_image.mean((0, 1))
            r1, g1, b1 = (r, g, b) / max(r, g, b)
        else:
            r1, g1, b1 = white_balance_profile

        img_wb = img.astype(np.float64) * gain / [r1, g1, b1]
        return np.clip(img_wb, 0, 255).astype(np.uint8)

    def get_device_properties(self, scope: str = "used") -> dict:
        """
        get used/allowed properties in mm2-device manager
        as a dictionary
        """
        device_dict = {}
        for device_name in obj_2_list(self.core.get_loaded_devices()):  # type: ignore
            device_property_names = self.core.get_device_property_names(device_name)  # type: ignore
            property_names = obj_2_list(device_property_names)
            prop_dict = {}
            for prop in property_names:
                if scope == "allowed":
                    values = self.core.get_allowed_property_values(device_name, prop)  # type: ignore
                    prop_dict.update({f"{prop}": obj_2_list(values)})
                elif scope == "used":
                    values = self.core.get_property(device_name, prop)  # type: ignore
                    prop_dict.update({f"{prop}": values})
                else:
                    warnings.warn(f" unknown metadata scope {scope} ")
            device_dict.update({f"{device_name}": prop_dict})
        return device_dict

    def _ppm_set_psgticks(self, theta: float) -> None:
        """Set the rotation stage to a specific angle and wait for completion."""
        theta_thor = ppm_psgticks_to_thor(theta)
        self.core.set_position(self.settings.stage.r_stage, theta_thor)  # type: ignore
        self.core.wait_for_device(self.settings.stage.r_stage)  # type: ignore

    def _ppm_get_psgticks(self) -> float:
        """Set the rotation stage to a specific angle and wait for completion."""
        return ppm_thor_to_psgticks(self.core.get_position(self.settings.stage.r_stage))  # type: ignore

    def _camm_swap_objective_lens(
        self,
        desired_imaging_mode: sp_imaging_mode,
    ):
        """ " 4x->20x moves O first, then Z
        and
        20x->4x moves z first"""

        current_slider_position = self.core.get_property(*self.settings.obj_slider)  # type: ignore
        if desired_imaging_mode.objective_position_label != current_slider_position:  # type: ignore

            if desired_imaging_mode.name.startswith("4X"):  # type: ignore
                self.core.set_focus_device(self.settings.stage.z_stage)  #  type: ignore
                self.core.set_position(desired_imaging_mode.z)  # type: ignore
                self.core.wait_for_device(self.settings.stage.z_stage)  # type: ignore
                self.core.set_property(*self.settings.obj_slider, desired_imaging_mode.objective_position_label)  # type: ignore
                self.core.set_focus_device(self.settings.stage.f_stage)  # type: ignore
                self.core.wait_for_system()  # type: ignore
            if desired_imaging_mode.name.startswith("20X"):  # type: ignore
                self.core.set_property(*self.settings.obj_slider, desired_imaging_mode.objective_position_label)  # type: ignore
                self.core.wait_for_device(self.settings.obj_slider[0])  # type: ignore
                self.core.set_focus_device(self.settings.stage.z_stage)  # type: ignore
                self.core.set_position(desired_imaging_mode.z)  # type: ignore
                self.core.set_focus_device(self.settings.stage.f_stage)  # type: ignore
                self.core.set_position(desired_imaging_mode.f)  # type: ignore
                self.core.wait_for_system()  # type: ignore

            self.core.set_focus_device(self.settings.stage.z_stage)  # type: ignore
            self.settings.imaging_mode = desired_imaging_mode
