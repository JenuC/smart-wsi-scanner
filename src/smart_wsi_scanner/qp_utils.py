"""
Utility classes for QuPath Scope Control
Contains utilities for tile configuration, autofocus, image writing, and background correction.
"""

import pathlib
import shutil
import re
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import matplotlib.patches
import tifffile as tf
import uuid
import skimage.morphology
import skimage.filters
import cv2
import logging

from .hardware import Position  # Use the new Position class instead of sp_position

logger = logging.getLogger(__name__)


class TileConfigUtils:
    """Utilities for reading and writing tile configuration files."""

    def __init__(self):
        pass

    @staticmethod
    def read_tile_config(tile_config_path: pathlib.Path, core) -> List[Tuple[Position, str]]:
        """
        Read tile positions + filename from a QuPath-generated TileConfiguration.txt file.

        Args:
            tile_config_path: Path to TileConfiguration.txt file
            core: Pycromanager Core object for getting Z position

        Returns:
            List of tuples containing (Position, filename)
        """
        positions: List[Tuple[Position, str]] = []
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
                        positions.append((Position(x, y, z), filename))
        else:
            logger.warning(f"Tile config file not found: {tile_config_path}")
        return positions

    @staticmethod
    def read_TileConfiguration_coordinates(tile_config_path) -> np.ndarray:
        """
        Read tile XY coordinates from a TileConfiguration.txt file.

        Args:
            tile_config_path: Path to TileConfiguration.txt file

        Returns:
            numpy array of XY coordinates
        """
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
        positions: Optional[list] = None,
        id1: str = "Tile",
        suffix_length: str = "06",
        pixel_size: float = 1.0,
    ):
        """
        Write a TileConfiguration.txt file.

        Args:
            tileconfig_path: Direct path to output file
            target_foldername: Folder to create file in
            positions: List of (x, y) positions
            id1: Prefix for tile names
            suffix_length: Number of digits for tile index
            pixel_size: Pixel size for scaling coordinates
        """
        if not tileconfig_path and target_foldername is not None:
            target_folder_path = pathlib.Path(target_foldername)
            tileconfig_path = str(target_folder_path / "TileConfiguration.txt")

        if tileconfig_path is not None and positions is not None:
            with open(tileconfig_path, "w") as text_file:
                print("dim = {}".format(2), file=text_file)
                for ix, pos in enumerate(positions):
                    file_id = f"{id1}_{ix:{suffix_length}}"
                    x, y = pos
                    print(
                        f"{file_id}.tif; ; ({x / pixel_size:.3f}, {y / pixel_size:.3f})",
                        file=text_file,
                    )


class AutofocusUtils:
    """Utilities for autofocus position calculation and focus metrics."""

    def __init__(self):
        pass

    @staticmethod
    def get_distance_sorted_xy_dict(positions):
        """Sort positions by radial distance from origin."""
        left_bottom = np.argmin(np.array([x[0] ** 2 + x[1] ** 2 for x in positions]))
        xa = positions[left_bottom]
        distances = np.round(cdist([xa], positions).ravel(), 2)
        positions_d = {ix: (positions[ix], distances[ix]) for ix in range(len(distances))}
        positions_d = dict(sorted(positions_d.items(), key=lambda item: item[1][1]))
        return positions_d

    @staticmethod
    def get_autofocus_positions(
        fov: Tuple[float, float], positions: List[Tuple[float, float]], n_tiles: float
    ) -> Tuple[List[int], float]:
        """
        Determine which tile positions require autofocus.

        Args:
            fov: Field of view (x, y) in micrometers
            positions: List of (x, y) tile positions
            n_tiles: Number of tiles between autofocus positions

        Returns:
            Tuple of (autofocus position indices, minimum distance)
        """
        fov_x, fov_y = fov

        # Compute the minimum required distance between autofocus positions
        af_min_distance = cdist([[0, 0]], [[fov_x * n_tiles, fov_y * n_tiles]])[0][0]

        # For each tile, if dist is higher, perform autofocus
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

        return af_positions, af_min_distance

    @staticmethod
    def visualize_autofocus_locations(
        fov: Tuple[float, float], positions: List[Tuple[float, float]], ntiles: float = 1.35
    ):
        """Visualize autofocus positions on a plot."""
        af_positions, af_min_distance = AutofocusUtils.get_autofocus_positions(
            fov, positions, ntiles
        )
        ax = plt.subplot(111)
        for ix, pos in enumerate(positions):
            if ix in af_positions:
                crc = matplotlib.patches.Circle(
                    (pos[0], pos[1]),
                    af_min_distance,
                    fill=False,
                )
                ax.add_artist(crc)
                ax.plot(pos[0], pos[1], "s", label="Autofocus" if ix == af_positions[0] else "")
            else:
                ax.plot(pos[0], pos[1], "o", markeredgecolor="k", label="Tile" if ix == 1 else "")

        # Set axis limits
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
        ax.set_xlabel("X position (µm)")
        ax.set_ylabel("Y position (µm)")
        ax.legend()
        plt.show()
        return af_positions, af_min_distance

    @staticmethod
    def autofocus_profile_laplacian_variance(image: np.ndarray) -> float:
        """Fast general sharpness metric - ~5ms for 2500x1900."""
        laplacian = skimage.filters.laplace(image)
        return float(laplacian.var())

    @staticmethod
    def autofocus_profile_sobel(image: np.ndarray) -> float:
        """Fast general sharpness metric - ~5ms for 2500x1900."""
        sobel = skimage.filters.sobel(image)
        return float(sobel.var())

    @staticmethod
    def autofocus_profile_brenner_gradient(image: np.ndarray) -> float:
        """Fastest option - ~3ms for 2500x1900."""
        gy, gx = np.gradient(image.astype(np.float32))
        return float(np.mean(gx**2 + gy**2))

    @staticmethod
    def autofocus_profile_robust_sharpness_metric(image: np.ndarray) -> float:
        """Particle-resistant but slower - ~20ms for 2500x1900."""
        # Median filter to remove particles (this is the slow part)
        filtered = skimage.filters.median(image, skimage.morphology.disk(3))

        # Calculate sharpness on filtered image
        laplacian = skimage.filters.laplace(filtered)

        # Exclude very dark regions from calculation
        threshold = skimage.filters.threshold_otsu(image)
        mask = image > (threshold * 0.5)

        return float(laplacian[mask].var()) if mask.any() else float(laplacian.var())

    @staticmethod
    def autofocus_profile_hybrid_sharpness_metric(image: np.ndarray) -> float:
        """Compromise: Fast with some particle resistance - ~8ms."""
        # Gaussian blur to reduce particle influence (faster than median)
        smoothed = skimage.filters.gaussian(image, sigma=1.5)

        # Use Brenner gradient on smoothed image
        gy, gx = np.gradient(smoothed.astype(np.float32))
        gradient_magnitude = gx**2 + gy**2

        # Soft masking: reduce weight of very dark/bright regions
        normalized = (image - image.min()) / (image.max() - image.min() + 1e-10)
        weight_mask = 1 - np.abs(normalized - 0.5) * 2  # Peak at mid-gray

        return float(np.mean(gradient_magnitude * weight_mask))


class TifWriterUtils:
    """Utilities for writing TIFF files and image processing."""

    def __init__(self):
        pass

    @staticmethod
    def ome_writer(filename: str, pixel_size_um: float, data: np.ndarray):
        """
        Write OME-TIFF file with metadata.

        Args:
            filename: Output filename
            pixel_size_um: Pixel size in micrometers
            data: Image data array
        """
        with tf.TiffWriter(filename) as tif:
            options = {
                "photometric": "rgb" if len(data.shape) == 3 else "minisblack",
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
    def format_imagetags(tags: dict) -> Dict[str, dict]:
        """Format image tags by grouping by prefix."""
        dx = {}
        for k in set([key.split("-")[0] for key in tags]):
            dx.update({k: {key: tags[key] for key in tags if key.startswith(k)}})
        return dx

    @staticmethod
    def create_birefringence_tile(
        pos_image: np.ndarray,
        neg_image: np.ndarray,
        output_dir: pathlib.Path,
        filename: str,
        pixel_size_um: float,
        tile_config_source: Optional[pathlib.Path] = None,
        logger=None,
    ) -> np.ndarray:
        """
        Create a single birefringence image from positive and negative angle images.

        Args:
            pos_image: Positive angle image
            neg_image: Negative angle image
            output_dir: Directory to save birefringence image
            filename: Output filename
            pixel_size_um: Pixel size for OME-TIFF metadata
            tile_config_source: Path to source TileConfiguration.txt to copy (optional)
            logger: Logger instance (optional)

        Returns:
            The birefringence image array
        """
        # Create output directory if it doesn't exist
        if not output_dir.exists():
            output_dir.mkdir(exist_ok=True)

            # Copy TileConfiguration.txt if source provided
            if tile_config_source and tile_config_source.exists():
                shutil.copy2(tile_config_source, output_dir / "TileConfiguration.txt")
                if logger:
                    logger.debug(f"Copied TileConfiguration.txt to {output_dir}")

        # Calculate birefringence
        output_path = output_dir / filename

        biref_img = TifWriterUtils.ppm_angle_difference(pos_image, neg_image)

        # Save grayscale version
        tf.imwrite(str(output_path)[:-4] + "_gray.tif", biref_img.astype(np.float32))

        # Normalize to 8-bit
        biref_img = biref_img * 255 / biref_img.max()
        biref_img = np.clip(biref_img, 0, 255).astype(np.uint8)

        # Save as RGB (grayscale replicated to 3 channels)
        TifWriterUtils.ome_writer(
            filename=str(output_path),
            pixel_size_um=pixel_size_um,
            data=np.stack([biref_img] * 3, axis=-1),
        )

        if logger:
            logger.info(f"  Created birefringence image: {filename}")

        return biref_img

    @staticmethod
    def subtraction_image(pos_image: np.ndarray, neg_image: np.ndarray) -> np.ndarray:
        """Simple subtraction of two images."""
        return pos_image.astype(np.float32) - neg_image.astype(np.float32)

    @staticmethod
    def ppm_angle_difference(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """
        Calculate angle difference for polarized microscopy images.
        Color represents retardation angle via interference colors.

        Args:
            img1: First image (RGB)
            img2: Second image (RGB)

        Returns:
            Difference image as single channel
        """
        # Convert to float for calculations
        img1_f = img1.astype(np.float32) / 255.0
        img2_f = img2.astype(np.float32) / 255.0

        # Method 1: Direct RGB difference (simple but effective)
        rgb_diff = np.sqrt(np.sum((img1_f - img2_f) ** 2, axis=2))

        # Alternative Method 2 (commented out): Hue-based for interference color progression
        # hsv1 = cv2.cvtColor(img1, cv2.COLOR_RGB2HSV)
        # hsv2 = cv2.cvtColor(img2, cv2.COLOR_RGB2HSV)
        #
        # # Hue difference with special handling for Michel-Lévy sequence
        # hue_diff = np.abs(hsv1[:, :, 0] - hsv2[:, :, 0])
        # hue_diff = np.minimum(hue_diff, 180 - hue_diff)
        #
        # # Weight by saturation (gray areas have no birefringence)
        # saturation_mask = (hsv1[:, :, 1] + hsv2[:, :, 1]) / 2.0 / 255.0
        # weighted_diff = hue_diff * saturation_mask

        return rgb_diff

    @staticmethod
    def apply_brightness_correction(image: np.ndarray, correction_factor: float) -> np.ndarray:
        """Apply brightness correction to an image."""
        return np.clip(image * correction_factor, 0, 255).astype(np.uint8)


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
        """
        Initialize QuPath project paths.

        Args:
            projectsFolderPath: Base path for all projects
            sampleLabel: Sample identifier
            scan_type: Type of scan performed
            region: Region identifier
            tile_config: Name of tile configuration file
        """
        self.path_tile_configuration = pathlib.Path(
            projectsFolderPath, sampleLabel, scan_type, region, tile_config
        )
        if self.path_tile_configuration.exists():
            self.path_qp_project = pathlib.Path(projectsFolderPath, sampleLabel)
            self.path_output = pathlib.Path(projectsFolderPath, sampleLabel, scan_type, region)
            self.acq_id = sampleLabel + "_ST_" + scan_type
        else:
            self.path_qp_project = pathlib.Path("undefined")
            self.path_output = pathlib.Path("undefined")
            self.acq_id = "undefined" + "_ScanType_" + "undefined"
            logger.warning(f"Tile configuration not found: {self.path_tile_configuration}")

    @staticmethod
    def uid() -> str:
        """Generate a unique identifier."""
        return uuid.uuid1().urn[9:]

    def __repr__(self):
        return (
            f"QuPath project: {self.path_qp_project}\n"
            f"TIF files: {self.path_output}\n"
            f"Acquisition ID: {self.acq_id}"
        )


class BackgroundCorrectionUtils:
    """Utilities for background and flat-field correction with modality support."""

    @staticmethod
    def calculate_background_color_from_mode(
        image: np.ndarray, bin_size: int = 5, mode_tolerance: int = 10
    ) -> Tuple[np.ndarray, float]:
        """
        Find background color using histogram mode approach.

        Args:
            image: RGB image array
            bin_size: Size of histogram bins (smaller = more precise but noisier)
            mode_tolerance: Pixels within this range of mode are considered background

        Returns:
            Tuple of (background_mean_rgb, confidence_score)
        """
        # Convert to grayscale for mode detection
        gray = np.mean(image, axis=2).astype(np.uint8)

        # Calculate histogram with reasonable bins
        hist, bins = np.histogram(gray, bins=256 // bin_size, range=(0, 255))

        # Find the mode (highest peak)
        mode_idx = np.argmax(hist)
        mode_value = (bins[mode_idx] + bins[mode_idx + 1]) / 2

        # Find pixels near the mode (likely background)
        background_mask = np.abs(gray - mode_value) < mode_tolerance

        # Calculate confidence (what fraction of image is background)
        confidence = np.sum(background_mask) / gray.size

        # Get RGB values of background pixels
        background_pixels = image[background_mask]

        if len(background_pixels) > 0:
            background_mean = background_pixels.mean(axis=0)
        else:
            # Fallback to image mean if no background found
            background_mean = image.mean(axis=(0, 1))
            confidence = 0.0

        return background_mean, confidence

    @staticmethod
    def get_modality_from_scan_type(scan_type: str) -> str:
        """
        Extract modality identifier from scan type.

        The scan type format is: ScanType_Objectivex_acquisitionnumber
        We need to extract: ScanType_Objectivex

        Examples:
        - "PPM_10x_1" -> "PPM_10x"
        - "PPM_40x_2" -> "PPM_40x"
        - "CAMM_20x_1" -> "CAMM_20x"
        - "PPM_4x_3" -> "PPM_4x"

        Args:
            scan_type: Full scan type string

        Returns:
            Modality string combining scan type and objective
        """
        parts = scan_type.split("_")

        # We need at least 3 parts: ScanType, Objective, and acquisition number
        if len(parts) >= 3:
            # Extract the first two parts (ScanType and Objective)
            scan_type_part = parts[0]  # e.g., "PPM" or "CAMM"
            objective_part = parts[1]  # e.g., "10x", "20x", "40x"

            # Validate that the second part contains 'x' (indicating objective magnification)
            if "x" in objective_part.lower():
                modality = f"{scan_type_part}_{objective_part}"
                return modality
            else:
                # If format doesn't match expected pattern, return full scan type
                logger.warning(f"Unexpected scan type format: {scan_type}")
                return scan_type
        else:
            logger.warning(f"Scan type has unexpected number of parts: {scan_type}")
            return scan_type

    @staticmethod
    def load_background_images(
        background_dir: pathlib.Path, modality: str, angles: List[float], logger=None
    ) -> Tuple[Dict[float, np.ndarray], Dict[float, float], Dict[float, List[float]]]:
        """
        Load background images and calculate consistent scaling factors for each angle.

        Returns:
            Tuple of (background_images_dict, scaling_factors_dict, white_balance_dict)
        """
        backgrounds = {}
        scaling_factors = {}
        white_balance_coeffs = {}

        modality_dir = background_dir / modality

        if not modality_dir.exists():
            if logger:
                logger.error(f"Modality directory not found: {modality_dir}")
            return backgrounds, scaling_factors, white_balance_coeffs

        for angle in angles:
            angle_dir = modality_dir / str(angle)
            background_file = angle_dir / "background.tif"

            if background_file.exists():
                try:
                    background_img = tf.imread(str(background_file))
                    backgrounds[angle] = background_img

                    # Calculate the scaling factor for this background
                    bg_float = background_img.astype(np.float32)

                    # Different scaling strategy for polarized vs brightfield
                    if angle == 90.0:  # Brightfield
                        # Scale to make background bright
                        bg_mean_all = bg_float.mean()
                        target_intensity = 240.0
                        scaling_factor = target_intensity / bg_mean_all if bg_mean_all > 0 else 1.0
                        if logger:
                            logger.info(f"  Background mean intensity at 90°: {bg_mean_all:.1f}")
                    else:  # Polarized angles (-5, 0, 5)
                        # Don't scale intensity - preserve dark backgrounds
                        scaling_factor = 1.0  # No intensity scaling!

                    scaling_factors[angle] = scaling_factor

                    # Calculate white balance from background
                    # Background should be neutral, so we calculate what correction is needed
                    bg_mean = background_img.mean(axis=(0, 1))  # Mean R,G,B

                    # Calculate coefficients to make all channels equal to the brightest
                    wb_coeffs = bg_mean / (bg_mean.max() + 1e-6)
                    white_balance_coeffs[angle] = wb_coeffs.tolist()

                    if logger:
                        logger.info(f"Loaded background for {modality} {angle}°")
                        logger.info(f"  Scaling factor: {scaling_factor:.3f}")
                        logger.info(f"  Background RGB means: {bg_mean}")
                        logger.info(f"  Auto WB coefficients: {wb_coeffs}")

                except Exception as e:
                    if logger:
                        logger.error(f"Failed to load background {background_file}: {e}")
            else:
                if logger:
                    logger.warning(f"No background found for {modality} {angle}°")

        return backgrounds, scaling_factors, white_balance_coeffs

    @staticmethod
    def validate_background_images(
        background_dir: pathlib.Path, modality: str, required_angles: List[float], logger=None
    ) -> Tuple[bool, List[float]]:
        """
        Validate that all required background images exist for a specific modality.

        Returns:
            (is_valid, missing_angles) tuple
        """
        missing_angles = []
        modality_dir = background_dir / modality

        if not modality_dir.exists():
            if logger:
                logger.error(f"Modality directory not found: {modality_dir}")
            return False, required_angles

        for angle in required_angles:
            angle_dir = modality_dir / str(angle)
            background_file = angle_dir / "background.tif"

            if not background_file.exists():
                missing_angles.append(angle)

        is_valid = len(missing_angles) == 0

        if not is_valid and logger:
            logger.error(f"Missing background images for {modality} angles: {missing_angles}")

        return is_valid, missing_angles

    @staticmethod
    def apply_flat_field_correction(
        image: np.ndarray,
        background: np.ndarray,
        scaling_factor: float,
        method: str = "divide",
    ) -> np.ndarray:
        """
        Apply flat-field correction with pre-calculated scaling for consistency.

        The scaling_factor is calculated once per background image and applied
        to all tiles to ensure seamless stitching.

        Args:
            image: Image to correct
            background: Background image
            scaling_factor: Pre-calculated scaling factor
            method: Correction method ('divide' or 'subtract')

        Returns:
            Corrected image
        """
        # Ensure float precision for calculations
        img_float = image.astype(np.float32)
        bg_float = background.astype(np.float32)

        # Prevent division by zero with small epsilon
        epsilon = 0.1
        bg_float = np.where(bg_float < epsilon, epsilon, bg_float)

        if method == "divide":
            # Apply the correction with consistent scaling
            corrected = (img_float / bg_float) * scaling_factor

            # For 8-bit images, we need to handle the range carefully
            if image.dtype == np.uint8:
                # Scale back to 8-bit range
                corrected = corrected * 240

                # Find where we're exceeding the range
                overflow_mask = corrected > 240  # Leave some headroom

                if np.any(overflow_mask):
                    # Compress the overflow region
                    overflow_values = corrected[overflow_mask]
                    # Map [240, max] -> [240, 254] with log compression
                    max_val = overflow_values.max()
                    if max_val > 240:
                        compressed = 240 + (254 - 240) * np.log1p(overflow_values - 240) / np.log1p(
                            max_val - 240
                        )
                        corrected[overflow_mask] = compressed

        elif method == "subtract":
            # For subtraction, we need a different approach
            # Scale the background to match the image intensity range
            bg_scaled = bg_float * scaling_factor
            corrected = img_float - (bg_float - bg_scaled)
            corrected = np.maximum(corrected, 0)  # No negative values

        # Final clipping to valid range
        if image.dtype == np.uint8:
            max_val = 255
        elif image.dtype == np.uint16:
            max_val = 65535
        else:
            max_val = 255  # Default for unknown types

        return np.clip(corrected, 0, max_val).astype(image.dtype)

    @staticmethod
    def acquire_background_image(
        hardware, output_path: pathlib.Path, angles: List[float], exposures: List[int], logger
    ) -> None:
        """
        Acquire single background images for flat-field correction.

        Args:
            hardware: Microscope hardware interface
            output_path: Base directory to save background images
            angles: List of angles to acquire
            exposures: List of exposure times
            logger: Logger instance
        """
        logger.info("=== ACQUIRING BACKGROUND IMAGES ===")

        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)

        for angle_idx, angle in enumerate(angles):
            # Create angle subdirectory
            angle_dir = output_path / str(angle)
            angle_dir.mkdir(exist_ok=True)

            # Set rotation angle if PPM
            if hasattr(hardware, "set_psg_ticks"):
                hardware.set_psg_ticks(angle)
                logger.info(f"Set angle to {angle}°")

            # Set exposure
            if angle_idx < len(exposures):
                hardware.core.set_exposure(exposures[angle_idx])
                logger.info(f"Set exposure to {exposures[angle_idx]}ms")

            # Acquire image with debayering
            image, metadata = hardware.snap_image(debayering=True)

            # Save as background.tif
            background_path = angle_dir / "background.tif"
            TifWriterUtils.ome_writer(
                filename=str(background_path),
                pixel_size_um=hardware.core.get_pixel_size_um(),
                data=image,
            )

            logger.info(f"Saved background for {angle}° to {background_path}")

        logger.info("=== BACKGROUND ACQUISITION COMPLETE ===")
