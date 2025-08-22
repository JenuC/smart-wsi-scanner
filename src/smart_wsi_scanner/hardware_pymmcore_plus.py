import warnings  # Used by smartpath class for validation warnings
from collections import OrderedDict  # Used by smartpath class for image tags


from pymmcore_plus import CMMCorePlus
from .hardware import MicroscopeHardware, is_coordinate_in_range
from .config import sp_microscope_settings, sp_position, sp_imaging_mode

import numpy as np  #  image processing on tagged_image.pix
import skimage.color
import skimage.filters  # Used by smartpath class for autofocus
import scipy.interpolate  # Used by smartpath class for autofocus interpolation
import matplotlib.pyplot as plt  # Used by smartpath class for autofocus plots
from .debayering import CPUDebayer


def obj_2_list(name):
    return [name.get(i) for i in range(name.size())]


def init_pymmcore_plus():
    """Initialize PyMMCorePlus connection."""
    core = CMMCorePlus.instance()
    core.loadSystemConfiguration("TODO")
    return core


def ppm_psgticks_to_thor(bi_angle: float) -> float:
    """Convert PPM angle (in degrees) to Thor rotation stage position."""
    return -2 * bi_angle + 276


def ppm_thor_to_psgticks(kinesis_pos: float) -> float:
    """Convert Thor rotation stage position to PPM angle (in degrees)."""
    return (276 - kinesis_pos) / 2


class PyMMCorePlusHardware(MicroscopeHardware):
    """Implementation for PyMMCorePlus-based microscopes."""

    def __init__(self, core: CMMCorePlus, settings: sp_microscope_settings):
        self.core = core
        # FIXME: Add back GUI capabilities
        # self.studio = studio
        self.settings = settings
        print(self.settings.microscope)
        if self.settings.microscope.name == "PPM":  # type: ignore
            self.set_psg_ticks = self._ppm_set_psgticks
            self.get_psg_ticks = self._ppm_get_psgticks
        if settings.microscope.name == "CAMM":  # type: ignore
            self.swap_objective_lens = self._camm_swap_objective_lens

    def move_to_position(self, position: sp_position) -> None:
        # Get current position and populate any missing coordinates
        current_position = self.get_current_position()
        position.populate_missing(current_position)

        if not is_coordinate_in_range(self.settings, position):
            raise ValueError("Position out of range")

        if self.core.getAutoFocusDevice() != self.settings.stage.z_stage:  # type: ignore
            self.core.setAutoFocusDevice(self.settings.focus_device)  # type: ignore

        self.core.setPosition(position.z)  # type: ignore
        self.core.setXYPosition(position.x, position.y)  # type: ignore
        self.core.waitForDevice(self.core.getXYStageDevice())  # type: ignore
        self.core.waitForDevice(self.core.getFocusDevice())  # type: ignore

    def get_current_position(self) -> sp_position:
        return sp_position(
            self.core.getXPosition(), self.core.getYPosition(), self.core.getPosition()  # type: ignore
        )

    def snap_image(self, background_correction=False, remove_alpha=True, debayering=False):
        """Snaps an Image using MM Core and returns img,tags"""
        # FIXME: Add back GUI functionality
        # if self.core.isSequenceRunning() and self.studio is not None:  # type: ignore
        #     self.studio.live().set_live_mode(False)  # type: ignore

        camera = self.get_device_properties()["Core"]["Camera"]

        if debayering and (camera == "MicroPublisher6"):
            self.core.setProperty("MicroPublisher6", "Color", "OFF")  # type:ignore

        self.core.snapImage()  # type: ignore

        tagged_image = self.core.getTaggedImage()  # type: ignore
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

        if debayering and (camera == "MicroPublisher6"):
            debayerx = CPUDebayer(
                pattern="GRBG",
                image_bit_clipmax=(2**14) - 1,
                image_dtype=np.uint16,
                convolution_mode="wrap",
            )

            pixels = debayerx.debayer(pixels)
            print("before uint16-uint14 scaling", pixels.mean((0, 1)))
            pixels = ((pixels / ((2**14) + 1)) * 255).astype(np.uint8)
            pixels = np.clip(pixels, 0, 255).astype(np.uint8)
            print("after uint14-uint8 scaling", pixels.mean((0, 1)))
            self.core.setProperty("MicroPublisher6", "Color", "ON")  # type:ignore

            return pixels, tags

        if camera in ["QCamera", "MicroPublisher6"]:
            # flip BGRA to ARGB
            if nchannels > 1:
                pixels = pixels[:, :, ::-1]
                if (camera == "MicroPublisher6") and (remove_alpha):
                    pixels = pixels[:, :, 1:]  # ARGB the alpha-channel is all zeros by default?
                    ## TODO verify if QCamera is BGRA
                if background_correction:
                    ## currently implemented in the qp-acquisition
                    pass

        elif camera == "OSc-LSM":
            pass
        else:
            print(f"Capture Failed: SP doesn't recognize : {tags['Core-Camera']=}")
            return None, None

        return pixels, tags

    def get_fov(self) -> tuple[float, float]:
        """returns field of view in settings.pixelsize units fov_x, fov_y"""
        camera = self.core.getProperty("Core", "Camera")  # type: ignore
        if camera == "OSc-LSM":
            height = int(self.core.getProperty(camera, "LSM-Resolution"))  # type: ignore
            width = height
        elif camera in ["QCamera", "MicroPublisher6"]:
            height = int(self.core.getProperty(camera, "Y-dimension"))  # type: ignore
            width = int(self.core.getProperty(camera, "X-dimension"))  # type: ignore
        else:
            raise ValueError(f"Unknown camera type: {camera}")
        pixel_size_um = self.core.getPixelSizeUm()  # type: ignore
        fov_y = height * pixel_size_um
        fov_x = width * pixel_size_um
        return fov_x, fov_y

    def autofocus(
        self,
        n_steps=5,
        search_range=45,
        interp_strength=100,
        interp_kind="quadratic",
        score_metric=skimage.filters.sobel,
        pop_a_plot=False,
        move_stage_to_estimate=True,
    ) -> float:  # type: ignore
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
                img, tags = self.snap_image()
                green1 = img[0::2, 0::2]
                green2 = img[1::2, 1::2]
                img_gray = ((green1 + green2) / 2.0).astype(np.float32)
                # img_gray = skimage.color.rgb2gray(img)
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

    #TODO DANGER MIKE ADDED CODE, FOR INITIAL SEARCH, MAYBE?
    def autofocus_adaptive_search(
        self,
        initial_step_size=10,    # Initial step size in microns
        min_step_size=2,         # Minimum step size before stopping
        focus_threshold=0.95,    # Threshold for "good enough" focus (relative to max seen)
        max_total_steps=25,      # Safety limit on total acquisitions
        score_metric=None,
        pop_a_plot=False,
        move_stage_to_estimate=True,
    ) -> float:
        """
        Adaptive autofocus that starts at current Z and searches outward.
        Minimizes acquisitions by stopping when focus is "good enough".
        """
        # Import the autofocus utilities
        from .qp_utils import AutofocusUtils
        
        # Use Laplacian variance from AutofocusUtils by default
        if score_metric is None:
            score_metric = AutofocusUtils.autofocus_profile_laplacian_variance
        
        current_pos = self.get_current_position()
        initial_z = current_pos.z
        
        # Get Z limits with safer attribute access
        z_min = -1000
        z_max = 1000
        if hasattr(self.settings, 'stage') and hasattr(self.settings.stage, 'z_limit'):
            if hasattr(self.settings.stage.z_limit, 'low'): #type: ignore
                z_min = self.settings.stage.z_limit.low #type: ignore
            if hasattr(self.settings.stage.z_limit, 'high'): #type: ignore
                z_max = self.settings.stage.z_limit.high #type: ignore
        
        # Keep track of all measurements
        z_positions = []
        scores = []
        
        # Helper function to acquire and score at a position
        def measure_at_z(z):
            if z < z_min + 5 or z > z_max - 5:  # Stay away from limits
                return -np.inf
            
            self.move_to_position(sp_position(current_pos.x, current_pos.y, z))
            img, tags = self.snap_image()
            
            # Check if image acquisition failed
            if img is None:
                print(f"Failed to acquire image at Z={z}")
                return -np.inf
            
            # Process image - check shape instead of ndim for safety
            if len(img.shape) == 2:  # Bayer pattern
                green1 = img[0::2, 0::2]
                green2 = img[1::2, 1::2]
                img_gray = ((green1 + green2) / 2.0).astype(np.float32)
            elif len(img.shape) == 3:  # RGB image
                img_gray = skimage.color.rgb2gray(img)
            else:
                img_gray = img.astype(np.float32)
            
            score = score_metric(img_gray)
            if hasattr(score, 'ndim') and score.ndim == 2:
                score = np.mean(score)
            
            return float(score)
        
        # Start with current position
        current_score = measure_at_z(initial_z)
        if current_score == -np.inf:
            print("Failed to acquire initial image")
            return initial_z #type: ignore
        
        z_positions.append(initial_z)
        scores.append(current_score)
        
        best_z = initial_z
        best_score = current_score
        
        # Adaptive search
        step_size = initial_step_size
        search_direction = None  # Will be determined by first measurements
        total_steps = 1
        
        while step_size >= min_step_size and total_steps < max_total_steps:
            # Measure above and below current best
            z_above = best_z + step_size #type: ignore
            z_below = best_z - step_size #type: ignore
            
            # Only measure positions we haven't checked yet
            positions_to_check = []
            if not any(abs(z - z_above) < 0.1 for z in z_positions) and z_above < z_max - 5:
                positions_to_check.append(('above', z_above))
            if not any(abs(z - z_below) < 0.1 for z in z_positions) and z_below > z_min + 5:
                positions_to_check.append(('below', z_below))
            
            if not positions_to_check:
                # Can't go further, reduce step size
                step_size /= 2
                continue
            
            # Measure new positions
            improved = False
            for direction, z_pos in positions_to_check:
                score = measure_at_z(z_pos)
                if score == -np.inf:
                    continue  # Skip failed acquisitions
                    
                z_positions.append(z_pos)
                scores.append(score)
                total_steps += 1
                
                if score > best_score:
                    best_score = score
                    best_z = z_pos
                    improved = True
                    search_direction = direction
            
            # Check if we're "good enough"
            if len(scores) > 3:  # Need some history
                max_seen = max(scores)
                if best_score >= focus_threshold * max_seen:
                    print(f"Found acceptable focus after {total_steps} steps")
                    break
            
            if improved:
                # Continue in the improving direction with same step size
                if search_direction == 'above':
                    next_z = best_z + step_size #type: ignore
                    if next_z < z_max - 5 and not any(abs(z - next_z) < 0.1 for z in z_positions):
                        continue
                else:  # below
                    next_z = best_z - step_size #type: ignore
                    if next_z > z_min + 5 and not any(abs(z - next_z) < 0.1 for z in z_positions):
                        continue
            else:
                # No improvement at this step size, refine
                step_size /= 2
        
        # Optional: Do a final fine interpolation around the best point
        if len(z_positions) > 2:
            # Sort by position for interpolation
            sorted_indices = np.argsort(z_positions)
            z_sorted = np.array(z_positions)[sorted_indices]
            scores_sorted = np.array(scores)[sorted_indices]
            
            # Find points around the best
            best_idx = np.where(z_sorted == best_z)[0][0]
            start_idx = max(0, best_idx - 2)
            end_idx = min(len(z_sorted), best_idx + 3)
            
            if end_idx - start_idx >= 3:  # Need at least 3 points
                z_local = z_sorted[start_idx:end_idx]
                scores_local = scores_sorted[start_idx:end_idx]
                
                # Quadratic interpolation
                interp_z = np.linspace(z_local[0], z_local[-1], 50)
                interp_scores = scipy.interpolate.interp1d(z_local, scores_local, kind='quadratic')(interp_z)
                best_z = interp_z[np.argmax(interp_scores)]
        
        if pop_a_plot:
            plt.figure(figsize=(10, 6))
            plt.scatter(z_positions, scores, c=range(len(scores)), cmap='viridis', s=50)
            plt.plot(best_z, max(scores), 'r*', markersize=15) #type: ignore
            plt.xlabel('Z position (µm)')
            plt.ylabel('Focus score')
            plt.title(f'Adaptive autofocus: {total_steps} acquisitions')
            plt.colorbar(label='Acquisition order')
            plt.show()
        
        print(f"Autofocus complete: Z={best_z:.1f} after {total_steps} acquisitions")
        
        if move_stage_to_estimate:
            self.move_to_position(sp_position(current_pos.x, current_pos.y, best_z))
        
        return best_z #type: ignore


    def white_balance(self, img=None, background_image=None, gain=1.0, white_balance_profile=None):
        if white_balance_profile is None:
            # load from profile
            # white_balance_profile = self.settings.white_balance.ppm.uncrossed
            white_balance_profile = self.settings.white_balance.default.default  # type: ignore

        if img is None:
            raise ValueError("Input image 'img' must not be None for white balancing.")

        if background_image is not None:
            r, g, b = background_image.mean((0, 1))
            r1, g1, b1 = (r, g, b) / max(r, g, b)
        else:
            r1, g1, b1 = white_balance_profile

        img_wb = img.astype(np.float64) * gain / [r1, g1, b1]

        # img_wb = img_wb * (255.0 / img_wb.max())

        return np.clip(img_wb, 0, 255).astype(np.uint8)

    def get_device_properties(self, scope: str = "used") -> dict:
        """
        get used/allowed properties in mm2-device manager
        as a dictionary
        """
        device_dict = {}
        for device_name in obj_2_list(self.core.getLoadedDevices()):  # type: ignore
            device_property_names = self.core.getDevicePropertyNames(device_name)  # type: ignore
            property_names = obj_2_list(device_property_names)
            prop_dict = {}
            for prop in property_names:
                if scope == "allowed":
                    values = self.core.getAllowedPropertyValues(device_name, prop)  # type: ignore
                    prop_dict.update({f"{prop}": obj_2_list(values)})
                elif scope == "used":
                    values = self.core.getProperty(device_name, prop)  # type: ignore
                    prop_dict.update({f"{prop}": values})
                else:
                    warnings.warn(f" unknown metadata scope {scope} ")
            device_dict.update({f"{device_name}": prop_dict})
        return device_dict

    def _ppm_set_psgticks(self, theta: float) -> None:
        """Set the rotation stage to a specific angle and wait for completion."""
        theta_thor = ppm_psgticks_to_thor(theta)
        self.core.setPosition(self.settings.stage.r_stage, theta_thor)  # type: ignore
        self.core.waitForDevice(self.settings.stage.r_stage)  # type: ignore

    def _ppm_get_psgticks(self) -> float:
        """Set the rotation stage to a specific angle and wait for completion."""
        return ppm_thor_to_psgticks(self.core.getPosition(self.settings.stage.r_stage))  # type: ignore

    def _camm_swap_objective_lens(
        self,
        desired_imaging_mode: sp_imaging_mode,
    ):
        """ " 4x->20x moves O first, then Z
        and
        20x->4x moves z first"""

        current_slider_position = self.core.getProperty(*self.settings.obj_slider)  # type: ignore
        if desired_imaging_mode.objective_position_label != current_slider_position:  # type: ignore

            if desired_imaging_mode.name.startswith("4X"):  # type: ignore
                self.core.setFocusDevice(self.settings.stage.z_stage)  #  type: ignore
                self.core.setPosition(desired_imaging_mode.z)  # type: ignore
                self.core.waitForDevice(self.settings.stage.z_stage)  # type: ignore
                self.core.setProperty(*self.settings.obj_slider, desired_imaging_mode.objective_position_label)  # type: ignore
                self.core.setFocusDevice(self.settings.stage.f_stage)  # type: ignore
                self.core.waitForSystem()  # type: ignore
            if desired_imaging_mode.name.startswith("20X"):  # type: ignore
                self.core.setProperty(*self.settings.obj_slider, desired_imaging_mode.objective_position_label)  # type: ignore
                self.core.waitForDevice(self.settings.obj_slider[0])  # type: ignore
                self.core.setFocusDevice(self.settings.stage.z_stage)  # type: ignore
                self.core.setPosition(desired_imaging_mode.z)  # type: ignore
                self.core.setFocusDevice(self.settings.stage.f_stage)  # type: ignore
                self.core.setPosition(desired_imaging_mode.f)  # type: ignore
                self.core.waitForSystem()  # type: ignore

            self.core.setFocusDevice(self.settings.stage.z_stage)  # type: ignore
            self.settings.imaging_mode = desired_imaging_mode
