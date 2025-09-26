"""Pycromanager hardware implementation for microscope control."""

import warnings
from collections import OrderedDict
from typing import Dict, Any, Optional, Tuple, List
import logging
import time
from pycromanager import Core, Studio
from .hardware import MicroscopeHardware, is_mm_running, is_coordinate_in_range, Position
from .qp_utils import AutofocusUtils

import numpy as np
import skimage.color
import skimage.filters
import scipy.interpolate
import matplotlib.pyplot as plt
from .debayering import CPUDebayer

logger = logging.getLogger(__name__)


def obj_2_list(name):
    """Convert Java object to Python list."""
    return [name.get(i) for i in range(name.size())]


def init_pycromanager():
    """Initialize Pycromanager connection."""
    if not is_mm_running():
        print("Micro-Manager is not running. Please start Micro-Manager before initializing.")
        return None, None
    core = Core()
    studio = Studio()
    core.set_timeout_ms(20000)
    return core, studio


def ppm_psgticks_to_thor(bi_angle: float) -> float:
    """Convert PPM angle (in degrees) to Thor rotation stage position."""
    return -2 * bi_angle + 276


def ppm_thor_to_psgticks(kinesis_pos: float) -> float:
    """Convert Thor rotation stage position to PPM angle (in degrees)."""
    return (276 - kinesis_pos) / 2


class PycromanagerHardware(MicroscopeHardware):
    """Implementation for Pycromanager-based microscopes."""

    def __init__(self, core: Core, studio: Studio, settings: Dict[str, Any]):
        """
        Initialize PycromanagerHardware with dictionary-based settings.

        Args:
            core: Pycromanager Core object
            studio: Pycromanager Studio object
            settings: Dictionary containing microscope configuration
        """
        self.core = core
        self.studio = studio
        self.settings = settings
        self.psg_angle = None
        self.rotation_device = None
        # Log microscope info
        microscope_info = settings.get("microscope", {})
        logger.info(
            f"Initializing hardware for microscope: {microscope_info.get('name', 'Unknown')}"
        )

        # Set up microscope-specific methods based on name
        microscope_name = microscope_info.get("name", "")

        if microscope_name == "PPM":
            self.set_psg_ticks = self._ppm_set_psgticks
            self.get_psg_ticks = self._ppm_get_psgticks
            ppm_config = self.settings.get("modalities", {}).get("ppm", {})
            r_device_name = ppm_config.get("rotation_stage", {}).get("device")
            self.rotation_device = (
                self.settings.get("id_stage", {}).get(r_device_name, {}).get("device")
            )
            if not self.rotation_device:
                # Fallback to looking for r_stage in stage config
                self.rotation_device = self.settings.get("stage", {}).get("r_stage")
            if not self.rotation_device:
                raise ValueError("No rotation stage device found in configuration")
            _ = self._ppm_get_psgticks()  # initialize psg_angle
            logger.info("PPM-specific methods initialized")

        if microscope_name == "CAMM":
            self.swap_objective_lens = self._camm_swap_objective_lens
            logger.info("CAMM-specific methods initialized")

    def move_to_position(self, position: Position) -> None:
        """Move stage to specified position."""
        # Get current position and populate any missing coordinates
        current_position = self.get_current_position()
        position.populate_missing(current_position)

        # Validate position is within range
        if not is_coordinate_in_range(self.settings, position):
            raise ValueError(f"Position out of range: {position}")

        # Get focus device from settings if available
        stage_config = self.settings.get("stage", {})
        z_stage_device = stage_config.get("z_stage", None)

        if z_stage_device and self.core.get_focus_device() != z_stage_device:
            self.core.set_focus_device(z_stage_device)

        # Move to position
        self.core.set_position(position.z)
        self.core.set_xy_position(position.x, position.y)
        self.core.wait_for_device(self.core.get_xy_stage_device())
        self.core.wait_for_device(self.core.get_focus_device())

        logger.debug(f"Moved to position: {position}")

    def get_current_position(self) -> Position:
        """Get current stage position."""
        return Position(
            self.core.get_x_position(), self.core.get_y_position(), self.core.get_position()
        )

    def snap_image(self, background_correction=False, remove_alpha=True, debayering=False):
        """
        Snap an image using MM Core and return img, tags.

        Args:
            background_correction: Apply background correction (if implemented)
            remove_alpha: Remove alpha channel from BGRA images
            debayering: Apply debayering for MicroPublisher6

        Returns:
            Tuple of (image_array, metadata_tags)
        """
        if self.core.is_sequence_running() and self.studio is not None:
            self.studio.live().set_live_mode(False)

        camera = self.get_device_properties()["Core"]["Camera"]

        # Handle debayering for MicroPublisher6
        if debayering and (camera == "MicroPublisher6"):
            self.core.set_property("MicroPublisher6", "Color", "OFF")

        # Handle white balance for JAI
        if camera == "JAICamera":
            self.core.set_property("JAICamera", "WhiteBalance", "Off")

        # Capture image
        self.core.snap_image()
        tagged_image = self.core.get_tagged_image()

        # Sort tags for consistency
        tags = OrderedDict(sorted(tagged_image.tags.items()))

        # Process pixels
        pixels = tagged_image.pix
        total_pixels = pixels.shape[0]
        height, width = tags["Height"], tags["Width"]
        assert (total_pixels % (height * width)) == 0
        nchannels = total_pixels // (height * width)

        if nchannels > 1:
            pixels = pixels.reshape(height, width, nchannels)
        else:
            pixels = pixels.reshape(height, width)

        # Apply debayering if requested
        if debayering and (camera == "MicroPublisher6"):
            debayerx = CPUDebayer(
                pattern="GRBG",
                image_bit_clipmax=(2**14) - 1,
                image_dtype=np.uint16,
                convolution_mode="wrap",
            )

            pixels = debayerx.debayer(pixels)
            logger.debug(f"Before uint16-uint14 scaling: mean {pixels.mean((0, 1))}")
            pixels = ((pixels / ((2**14) + 1)) * 255).astype(np.uint8)
            pixels = np.clip(pixels, 0, 255).astype(np.uint8)
            logger.debug(f"After uint14-uint8 scaling: mean {pixels.mean((0, 1))}")
            self.core.set_property("MicroPublisher6", "Color", "ON")

            return pixels, tags

        # Handle different camera types
        if camera in ["QCamera", "MicroPublisher6", "JAICamera"]:
            if nchannels > 1:
                pixels = pixels[:, :, ::-1]  # BGRA to ARGB
                if (camera != "QCamera") and remove_alpha:
                    pixels = pixels[:, :, 1:]  # Remove alpha channel

        elif camera == "OSc-LSM":
            pass
        else:
            logger.error(
                f"Capture Failed: Unrecognized camera: {tags.get('Core-Camera', 'Unknown')}"
            )
            return None, None

        return pixels, tags

    def get_fov(self) -> Tuple[float, float]:
        """
        Get field of view in micrometers.

        Returns:
            Tuple of (fov_x, fov_y) in micrometers
        """
        camera = self.core.get_property("Core", "Camera")

        if camera == "OSc-LSM":
            height = int(self.core.get_property(camera, "LSM-Resolution"))
            width = height
        elif camera == "JAICamera":
            height = self.settings["id_detector"]["LOCI_DETECTOR_JAI_001"]["height_px"]
            width = self.settings["id_detector"]["LOCI_DETECTOR_JAI_001"]["width_px"]

        elif camera in ["QCamera", "MicroPublisher6"]:
            height = int(self.core.get_property(camera, "Y-dimension"))
            width = int(self.core.get_property(camera, "X-dimension"))

        else:
            raise ValueError(f"Unknown camera type: {camera}")

        pixel_size_um = self.core.get_pixel_size_um()
        fov_y = height * pixel_size_um
        fov_x = width * pixel_size_um

        return fov_x, fov_y

    def set_exposure(self, exposure_ms: float) -> None:
        """Set camera exposure time in milliseconds."""
        camera = self.core.get_property("Core", "Camera")
        if camera == "JAICamera":
            frame_rate_min = 0.125
            frame_rate_max = 38.0
            margin = 1.01
            exposure_s = exposure_ms / 1000.0
            required_frame_rate = round(1.0 / (exposure_s * margin), 3)
            frame_rate = min(max(required_frame_rate, frame_rate_min), frame_rate_max)
            self.core.set_property("JAICamera", "FrameRateHz", frame_rate)
            self.core.set_property("JAICamera", "Exposure", exposure_ms)
        else:
            self.core.set_exposure(exposure_ms)
        self.core.wait_for_device(camera)

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
        Perform autofocus using specified score metric.

        Args:
            n_steps: Number of Z positions to sample
            search_range: Total Z range to search in micrometers
            interp_strength: Interpolation density factor
            interp_kind: Type of interpolation ('linear', 'quadratic', 'cubic')
            score_metric: Function to score image focus
            pop_a_plot: Whether to show a focus score plot
            move_stage_to_estimate: Whether to move to best focus position

        Returns:
            Best focus Z position
        """
        steps = np.linspace(0, search_range, n_steps) - (search_range / 2)

        current_pos = self.get_current_position()
        z_steps = current_pos.z + steps
        print(z_steps)
        try:
            scores = []
            for step_number in range(n_steps):
                new_pos = Position(current_pos.x, current_pos.y, current_pos.z + steps[step_number])
                self.move_to_position(new_pos)

                img, tags = self.snap_image()

                # Extract green channel for focus calculation
                if self.core.get_property("Core", "Camera") == "JAICamera":
                    img_gray = np.mean(img, 2)
                else:
                    # TODO: debayer to go to gray ?
                    # TODO support other cameras!
                    green1 = img[0::2, 0::2]
                    green2 = img[1::2, 1::2]
                    img_gray = ((green1 + green2) / 2.0).astype(np.float32)

                score = score_metric(img_gray)
                if hasattr(score, "ndim") and score.ndim == 2:
                    score = np.mean(score)
                scores.append(score)

            # Interpolate to find best focus
            interp_x = np.linspace(z_steps[0], z_steps[-1], n_steps * interp_strength)
            interp_y = scipy.interpolate.interp1d(z_steps, scores, kind=interp_kind)(interp_x)
            new_z = interp_x[np.argmax(interp_y)]

            if pop_a_plot:
                plt.figure()
                plt.bar(z_steps, scores)
                plt.plot(interp_x, interp_y, "k")
                plt.plot(interp_x[np.argmax(interp_y)], interp_y.max(), "or")
                plt.xlabel("Z-axis (µm)")
                plt.title(f"Autofocus at X={current_pos.x:.1f}, Y={current_pos.y:.1f}")
                plt.show()

            if move_stage_to_estimate:
                new_pos = Position(current_pos.x, current_pos.y, new_z)
                self.move_to_position(new_pos)

            return new_z

        except Exception as e:
            logger.error(f"Autofocus failed: {e}")
            self.move_to_position(current_pos)
            raise e

    def autofocus_adaptive_search(
        self,
        initial_step_size=10,
        min_step_size=2,
        focus_threshold=0.95,
        max_total_steps=25,
        score_metric=None,
        pop_a_plot=False,
        move_stage_to_estimate=True,
    ) -> float:
        """
        Adaptive autofocus that starts at current Z and searches outward.
        Minimizes acquisitions by stopping when focus is "good enough".
        """
        if score_metric is None:
            score_metric = AutofocusUtils.autofocus_profile_laplacian_variance

        current_pos = self.get_current_position()
        initial_z = current_pos.z

        # Get Z limits from settings
        stage_limits = self.settings.get("stage", {}).get("limits", {})
        z_limits = stage_limits.get("z_um", {})
        z_min = z_limits.get("low", -1000)
        z_max = z_limits.get("high", 1000)

        # Keep track of all measurements
        z_positions = []
        scores = []

        # Helper function to acquire and score at a position
        def measure_at_z(z):
            if z < z_min + 5 or z > z_max - 5:
                return -np.inf

            self.move_to_position(Position(current_pos.x, current_pos.y, z))
            img, tags = self.snap_image()

            if img is None:
                logger.error(f"Failed to acquire image at Z={z}")
                return -np.inf

            # Process image
            if len(img.shape) == 2:  # Bayer pattern
                green1 = img[0::2, 0::2]
                green2 = img[1::2, 1::2]
                img_gray = ((green1 + green2) / 2.0).astype(np.float32)
            elif len(img.shape) == 3:  # RGB image
                img_gray = skimage.color.rgb2gray(img)
            else:
                img_gray = img.astype(np.float32)

            score = score_metric(img_gray)
            if hasattr(score, "ndim") and score.ndim == 2:
                score = np.mean(score)

            return float(score)

        # Start with current position
        current_score = measure_at_z(initial_z)
        if current_score == -np.inf:
            logger.error("Failed to acquire initial image")
            return initial_z

        z_positions.append(initial_z)
        scores.append(current_score)

        best_z = initial_z
        best_score = current_score

        # Adaptive search
        step_size = initial_step_size
        search_direction = None
        total_steps = 1

        while step_size >= min_step_size and total_steps < max_total_steps:
            # Measure above and below current best
            z_above = best_z + step_size
            z_below = best_z - step_size

            positions_to_check = []
            if not any(abs(z - z_above) < 0.1 for z in z_positions) and z_above < z_max - 5:
                positions_to_check.append(("above", z_above))
            if not any(abs(z - z_below) < 0.1 for z in z_positions) and z_below > z_min + 5:
                positions_to_check.append(("below", z_below))

            if not positions_to_check:
                step_size /= 2
                continue

            improved = False
            for direction, z_pos in positions_to_check:
                score = measure_at_z(z_pos)
                if score == -np.inf:
                    continue

                z_positions.append(z_pos)
                scores.append(score)
                total_steps += 1

                if score > best_score:
                    best_score = score
                    best_z = z_pos
                    improved = True
                    search_direction = direction

            # Check if we're "good enough"
            if len(scores) > 3:
                max_seen = max(scores)
                if best_score >= focus_threshold * max_seen:
                    logger.info(f"Found acceptable focus after {total_steps} steps")
                    break

            if improved:
                if search_direction == "above":
                    next_z = best_z + step_size
                    if next_z < z_max - 5 and not any(abs(z - next_z) < 0.1 for z in z_positions):
                        continue
                else:  # below
                    next_z = best_z - step_size
                    if next_z > z_min + 5 and not any(abs(z - next_z) < 0.1 for z in z_positions):
                        continue
            else:
                step_size /= 2

        # Optional fine interpolation
        if len(z_positions) > 2:
            sorted_indices = np.argsort(z_positions)
            z_sorted = np.array(z_positions)[sorted_indices]
            scores_sorted = np.array(scores)[sorted_indices]

            best_idx = np.where(z_sorted == best_z)[0][0]
            start_idx = max(0, best_idx - 2)
            end_idx = min(len(z_sorted), best_idx + 3)

            if end_idx - start_idx >= 3:
                z_local = z_sorted[start_idx:end_idx]
                scores_local = scores_sorted[start_idx:end_idx]

                interp_z = np.linspace(z_local[0], z_local[-1], 50)
                interp_scores = scipy.interpolate.interp1d(z_local, scores_local, kind="quadratic")(
                    interp_z
                )
                best_z = interp_z[np.argmax(interp_scores)]

        if pop_a_plot:
            plt.figure(figsize=(10, 6))
            plt.scatter(z_positions, scores, c=range(len(scores)), cmap="viridis", s=50)
            plt.plot(best_z, max(scores), "r*", markersize=15)
            plt.xlabel("Z position (µm)")
            plt.ylabel("Focus score")
            plt.title(f"Adaptive autofocus: {total_steps} acquisitions")
            plt.colorbar(label="Acquisition order")
            plt.show()

        logger.info(f"Autofocus complete: Z={best_z:.1f} after {total_steps} acquisitions")

        if move_stage_to_estimate:
            self.move_to_position(Position(current_pos.x, current_pos.y, best_z))

        return best_z

    def white_balance(self, img=None, background_image=None, gain=1.0, white_balance_profile=None):
        """Apply white balance correction to image."""
        if white_balance_profile is None:
            # Try to get default from settings
            wb_settings = self.settings.get("white_balance", {})
            default_wb = wb_settings.get("default", {}).get("default", [1.0, 1.0, 1.0])
            white_balance_profile = default_wb

        if img is None:
            raise ValueError("Input image 'img' must not be None for white balancing.")

        if background_image is not None:
            r, g, b = background_image.mean((0, 1))
            r1, g1, b1 = (r, g, b) / max(r, g, b)
        else:
            r1, g1, b1 = white_balance_profile

        img_wb = img.astype(np.float64) * gain / [r1, g1, b1]
        return np.clip(img_wb, 0, 255).astype(np.uint8)

    def get_device_properties(self, scope: str = "used") -> Dict[str, Dict[str, Any]]:
        """
        Get device properties from MM device manager.

        Args:
            scope: 'used' for current values, 'allowed' for possible values

        Returns:
            Dictionary of device properties
        """
        device_dict = {}
        for device_name in obj_2_list(self.core.get_loaded_devices()):
            device_property_names = self.core.get_device_property_names(device_name)
            property_names = obj_2_list(device_property_names)
            prop_dict = {}

            for prop in property_names:
                if scope == "allowed":
                    values = self.core.get_allowed_property_values(device_name, prop)
                    prop_dict[prop] = obj_2_list(values)
                elif scope == "used":
                    values = self.core.get_property(device_name, prop)
                    prop_dict[prop] = values
                else:
                    warnings.warn(f"Unknown metadata scope {scope}")

            device_dict[device_name] = prop_dict

        return device_dict

    def wrap_angle_m180_p180(self, angle_deg):
        """
        Wrap an angle in degrees to the range [-180, 180)
        """
        wrapped = (angle_deg + 180) % 360 - 180
        return wrapped

    def get_ccw_rot_angle(self, theta, is_sequence_start=False):
        """
        Get counter clockwise rotation angle maintaining polarization state consistency.

        CRITICAL: Within an acquisition sequence (e.g., -90°, -7°, 0°, 7° for one tile),
        ALL angles must maintain the same optical polarization state to avoid
        alternating light/dark intensities. Large rotations that flip the optical
        element are ONLY allowed between acquisition sequences.

        Args:
            theta: Target optical angle (e.g., -90, -7, 0, 7)
            is_sequence_start: True if starting new acquisition sequence (allows large rotation)

        Returns:
            Next PPM tick value for motor positioning
        """
        current_angle_wrapped = self.get_psg_ticks()
        current_angle = self.psg_angle

        # Convert optical angle to PPM ticks (base positions)
        # Handle special known angles first
        special_angles = {
            -90: 90,   # -90° optical -> 90 PPM ticks
            90: 90,    # 90° optical -> 90 PPM ticks (same as -90°, 90° away from 0°)
            -7: 173,   # -7° optical -> 173 PPM ticks (180 - 7)
            0: 180,    # 0° optical -> 180 PPM ticks (or 0, but using 180)
            7: 7       # 7° optical -> 7 PPM ticks
        }

        if theta in special_angles:
            target_ppm_ticks = special_angles[theta]
        else:
            # For other angles, calculate PPM ticks
            # Convert angle to equivalent PPM ticks (0-179 range)
            if theta < 0:
                # Negative angles: convert to positive equivalent
                target_ppm_ticks = 180 + theta  # e.g., -5 -> 175
            else:
                # Positive angles: use as-is but ensure within range
                target_ppm_ticks = theta % 180

        if is_sequence_start:
            # Starting new acquisition sequence - force "a" polarization state
            # "a" positions are in even-numbered 360° cycles: 0-359, 720-1079, 1440-1799, etc.

            # Find the next even-numbered cycle (360° period)
            current_cycle = current_angle // 360

            # Target cycle should be even (for "a" position)
            if current_cycle % 2 != 0:
                # Currently in odd cycle ("b"), move to next even cycle ("a")
                target_cycle = current_cycle + 1
            else:
                # Currently in even cycle ("a")
                target_cycle = current_cycle

            # Calculate candidate position in target cycle
            candidate = target_ppm_ticks + (target_cycle * 360)

            # If we've already passed this position, move to next even cycle
            if candidate <= current_angle:
                target_cycle += 2 if target_cycle % 2 == 0 else 1
                candidate = target_ppm_ticks + (target_cycle * 360)

            return candidate

        else:
            # Within acquisition sequence - stay in the same 360° cycle
            # This ensures all angles in the sequence maintain the same polarization state

            current_cycle = current_angle // 360
            candidate = target_ppm_ticks + (current_cycle * 360)

            # If we can't reach this angle in the current cycle (already passed it),
            # we have a problem - the sequence should be designed to avoid this
            if candidate <= current_angle:
                # This should not happen in a well-designed sequence, but handle it
                logger.warning(f"Angle sequence issue: target {target_ppm_ticks} in cycle {current_cycle} "
                             f"would go backwards from {current_angle} to {candidate}")
                # Stay in same cycle but move to next logical position
                candidate = current_angle + (target_ppm_ticks % 180)

            return candidate

    def _ppm_set_psgticks(self, theta: float, is_sequence_start: bool = False) -> None:
        """Set the PPM rotation stage to a specific angle."""
        # Try to get rotation stage device from settings
        rotation_device = self.rotation_device
        new_theta = self.get_ccw_rot_angle(theta, is_sequence_start=is_sequence_start)
        theta_thor = ppm_psgticks_to_thor(new_theta)
        current_pos_thor = self.core.get_position(rotation_device)
        assert theta_thor < current_pos_thor
        self.core.set_position(rotation_device, theta_thor)
        self.core.wait_for_device(rotation_device)
        logger.debug(f"Set rotation angle to {theta}° (Thor position: {theta_thor})")
        print(
            f"[PPM Rotation Stage] Requested: {theta}°, "
            f"CCW-adjusted: {new_theta}°, "
            f"Current (Thor): {current_pos_thor}, "
            f"Target (Thor): {theta_thor}"
        )

    def _ppm_get_psgticks(self) -> float:
        """Get the current PPM rotation angle."""
        rotation_device = self.rotation_device
        thor_pos = self.core.get_position(rotation_device)
        self.psg_angle = ppm_thor_to_psgticks(thor_pos)
        angle_wrapped = self.wrap_angle_m180_p180(self.psg_angle)
        return angle_wrapped

    # def _ppm_set_psgticks(self, theta: float) -> None:
    #     """Set the PPM rotation stage to a specific angle."""
    #     # Try to get rotation stage device from settings
    #     rotation_device = self.rotation_device

    #     theta_thor = ppm_psgticks_to_thor(theta)
    #     self.core.set_position(rotation_device, theta_thor)
    #     self.core.wait_for_device(rotation_device)

    #     logger.debug(f"Set rotation angle to {theta}° (Thor position: {theta_thor})")

    # def _ppm_get_psgticks(self) -> float:
    #     """Get the current PPM rotation angle."""
    #     rotation_device = self.rotation_device
    #     thor_pos = self.core.get_position(rotation_device)
    #     return ppm_thor_to_psgticks(thor_pos)

    def _camm_swap_objective_lens(self, desired_imaging_mode: Dict[str, Any]):
        """
        Swap objective lens for CAMM microscope.

        Args:
            desired_imaging_mode: Dictionary containing imaging mode configuration
        """
        # Get objective slider device from settings
        obj_slider = self.settings.get("obj_slider")
        if not obj_slider:
            raise ValueError("No objective slider configuration found")

        current_slider_position = self.core.get_property(*obj_slider)
        desired_position = desired_imaging_mode.get("objective_position_label")

        if not desired_position:
            raise ValueError("No objective position label in imaging mode")

        if desired_position != current_slider_position:
            mode_name = desired_imaging_mode.get("name", "")
            stage_config = self.settings.get("stage", {})
            z_stage = stage_config.get("z_stage")
            f_stage = stage_config.get("f_stage")

            if not z_stage or not f_stage:
                raise ValueError("Stage devices not properly configured")

            # Handle different objectives differently
            if mode_name.startswith("4X"):
                self.core.set_focus_device(z_stage)
                self.core.set_position(desired_imaging_mode.get("z", 0))
                self.core.wait_for_device(z_stage)
                self.core.set_property(*obj_slider, desired_position)
                self.core.set_focus_device(f_stage)
                self.core.wait_for_system()

            elif mode_name.startswith("20X"):
                self.core.set_property(*obj_slider, desired_position)
                self.core.wait_for_device(obj_slider[0])
                self.core.set_focus_device(z_stage)
                self.core.set_position(desired_imaging_mode.get("z", 0))
                self.core.set_focus_device(f_stage)
                self.core.set_position(desired_imaging_mode.get("f", 0))
                self.core.wait_for_system()

            self.core.set_focus_device(z_stage)

            # Update current imaging mode in settings
            self.settings["imaging_mode"] = desired_imaging_mode
            logger.info(f"Swapped to objective: {desired_position}")
