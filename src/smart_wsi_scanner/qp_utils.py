import pathlib
import shutil
import re
from typing import List, Tuple, Optional
from .config import sp_position
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import matplotlib.patches
import tifffile as tf
import uuid
import skimage.morphology
import skimage.filters
import cv2

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
    def get_autofocus_positions(fov, positions: list[tuple[float, float]], n_tiles: float):

        fov_x, fov_y = fov

        # Compute the minimum required distance between autofocus positions,
        af_min_distance = cdist([[0, 0]], [[fov_x * n_tiles, fov_y * n_tiles]])[0][0]

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

    @staticmethod
    def autofocus_profile_laplacian_variance(image):
        """Fast general sharpness metric - ~5ms for 2500x1900"""
        laplacian = skimage.filters.laplace(image)
        return laplacian.var()

    @staticmethod
    def autofocus_profile_sobel(image):
        """Fast general sharpness metric - ~5ms for 2500x1900"""
        laplacian = skimage.filters.sobel(image)
        return laplacian.var()

    @staticmethod
    def autofocus_profile_brenner_gradient(image):
        """Fastest option - ~3ms for 2500x1900"""
        gy, gx = np.gradient(image.astype(np.float32))
        return np.mean(gx**2 + gy**2)

    @staticmethod
    def autofocus_profile_robust_sharpness_metric(image):
        """Particle-resistant but slower - ~20ms for 2500x1900"""
        # Median filter to remove particles (this is the slow part)
        filtered = skimage.filters.median(image, skimage.morphology.disk(3))

        # Calculate sharpness on filtered image
        laplacian = skimage.filters.laplace(filtered)

        # Exclude very dark regions from calculation
        threshold = skimage.filters.threshold_otsu(image)
        mask = image > (threshold * 0.5)

        return laplacian[mask].var() if mask.any() else laplacian.var()

    @staticmethod
    def autofocus_profile_hybrid_sharpness_metric(image):
        """Compromise: Fast with some particle resistance - ~8ms"""
        # Gaussian blur to reduce particle influence (faster than median)
        smoothed = skimage.filters.gaussian(image, sigma=1.5)

        # Use Brenner gradient on smoothed image
        gy, gx = np.gradient(smoothed.astype(np.float32))
        gradient_magnitude = gx**2 + gy**2

        # Soft masking: reduce weight of very dark/bright regions
        normalized = (image - image.min()) / (image.max() - image.min() + 1e-10)
        weight_mask = 1 - np.abs(normalized - 0.5) * 2  # Peak at mid-gray

        return np.mean(gradient_magnitude * weight_mask)


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

    # TODO MIKE ADDED, SHOULD PROBABLY GO SOMEWHERE ELSE
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
        """Create a single birefringence image from positive and negative angle images.

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
        tf.imwrite(str(output_path)[:-4] + "_gray.tif", biref_img.astype(np.float32))
        biref_img = biref_img * 255 / biref_img.max()
        biref_img = np.clip(biref_img, 0, 255).astype(np.uint8)

        # Save the image        
        TifWriterUtils.ome_writer(
            filename=str(output_path),
            pixel_size_um=pixel_size_um,
            data=np.stack([biref_img] * 3, axis=-1),
        )

        if logger:
            logger.info(f"  Created birefringence image: {filename}")

        return biref_img

    @staticmethod
    def subtraction_image(pos_image, neg_image):
        return pos_image.astype(np.float32) - neg_image.astype(np.float32)

    @staticmethod
    def ppm_angle_difference(img1, img2):
        """
        Calculate angle difference for polarized microscopy images.
        Color represents retardation angle via interference colors.
        """
        # Convert to float for calculations
        img1_f = img1.astype(np.float32) / 255.0
        img2_f = img2.astype(np.float32) / 255.0

        # Method 1: Direct RGB difference (simple but effective)
        rgb_diff = np.sqrt(np.sum((img1_f - img2_f) ** 2, axis=2))

        # Method 2: Hue-based for interference color progression
        # hsv1 = cv2.cvtColor(img1, cv2.COLOR_RGB2HSV)
        # hsv2 = cv2.cvtColor(img2, cv2.COLOR_RGB2HSV)

        # Hue difference with special handling for Michel-Lévy sequence
        # hue_diff = np.abs(hsv1[:, :, 0] - hsv2[:, :, 0])
        # hue_diff = np.minimum(hue_diff, 180 - hue_diff)

        # Weight by saturation (gray areas have no birefringence)
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


# TODO MIKE EDITS, NOT SURE HOW TO IMPLEMENT BUT GETTING A START
class BackgroundCorrectionUtils:
    """Utilities for background and flat-field correction with modality support."""
    
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
        parts = scan_type.split('_')
        
        # We need at least 3 parts: ScanType, Objective, and acquisition number
        if len(parts) >= 3:
            # Extract the first two parts (ScanType and Objective)
            scan_type_part = parts[0]  # e.g., "PPM" or "CAMM"
            objective_part = parts[1]   # e.g., "10x", "20x", "40x"
            
            # Validate that the second part contains 'x' (indicating objective magnification)
            if 'x' in objective_part.lower():
                modality = f"{scan_type_part}_{objective_part}"
                return modality
            else:
                # If format doesn't match expected pattern, return full scan type
                # This provides a fallback for unexpected formats
                return scan_type
        else:
            #TODO this should probably throw an error or warning
            return scan_type
    
    
    @staticmethod
    def load_background_images(
        background_dir: pathlib.Path, 
        modality: str,
        angles: List[float], 
        logger=None
    ) -> Tuple[dict, dict]:
        """
        Load background images and calculate consistent scaling factors for each angle.
        
        Returns:
            Tuple of (background_images_dict, scaling_factors_dict)
        """
        backgrounds = {}
        scaling_factors = {}
        modality_dir = background_dir / modality
        
        if not modality_dir.exists():
            if logger:
                logger.error(f"Modality directory not found: {modality_dir}")
            return backgrounds, scaling_factors
        
        for angle in angles:
            angle_dir = modality_dir / str(angle)
            background_file = angle_dir / "background.tif"
            
            if background_file.exists():
                try:
                    background_img = tf.imread(str(background_file))
                    backgrounds[angle] = background_img
                    
                    # Calculate the scaling factor for this background
                    # This ensures consistent correction across all tiles
                    
                    # For 8-bit images, we need a careful approach
                    # We'll use the bright areas of the background as reference
                    bg_float = background_img.astype(np.float32)
                    
                    # Find the "typical" bright area intensity
                    # We use percentile to avoid outliers from dust/artifacts
                    bright_percentile = np.percentile(bg_float[bg_float > 50], 75)
                    
                    # The scaling factor that would bring typical background to ~200
                    # This leaves headroom for brighter sample areas
                    target_intensity = 200.0  # Conservative target for 8-bit
                    scaling_factor = target_intensity / bright_percentile if bright_percentile > 0 else 1.0
                    
                    # Store this factor for consistent use across all tiles
                    scaling_factors[angle] = scaling_factor
                    
                    if logger:
                        logger.info(f"Loaded background for {modality} {angle}°")
                        logger.info(f"  Background bright areas: {bright_percentile:.1f}")
                        logger.info(f"  Scaling factor: {scaling_factor:.3f}")
                        
                except Exception as e:
                    if logger:
                        logger.error(f"Failed to load background {background_file}: {e}")
            else:
                if logger:
                    logger.warning(f"No background found for {modality} {angle}°")
                    
        return backgrounds, scaling_factors
        
    @staticmethod
    def validate_background_images(
        background_dir: pathlib.Path,
        modality: str,
        required_angles: List[float], 
        logger=None
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
            # The scaling_factor is chosen to keep most values in range,
            # but we still need to handle outliers
            
            if image.dtype == np.uint8:
                # Soft clipping approach: compress the high values rather than hard clip
                # This preserves some detail in very bright areas
                
                # Find where we're exceeding the range
                overflow_mask = corrected > 240  # Leave some headroom
                
                if np.any(overflow_mask):
                    # Compress the overflow region
                    overflow_values = corrected[overflow_mask]
                    # Map [240, max] -> [240, 254] with log compression
                    max_val = overflow_values.max()
                    if max_val > 240:
                        compressed = 240 + (254 - 240) * np.log1p(overflow_values - 240) / np.log1p(max_val - 240)
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

    # COULD PASS A UNIQUE COMMAND FROM QUPATH, OR HAVE QUPATH POINT TO A DATA DIR FOR BACKGROUND IMAGES AND DO A STANDARD ACQUISITION
    @staticmethod
    def acquire_background_image(
        hardware, output_path: pathlib.Path, angles: List[float], exposures: List[int], logger
    ) -> None:
        """Acquire single background images for flat-field correction.

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

            # Set rotation angle
            if hasattr(hardware, "set_psg_ticks"):
                hardware.set_psg_ticks(angle)
                logger.info(f"Set angle to {angle}°")

            # Set exposure
            if angle_idx < len(exposures):
                hardware.core.set_exposure(exposures[angle_idx])
                logger.info(f"Set exposure to {exposures[angle_idx]}ms")

            # Acquire image with debayering
            image, metadata = hardware.snap_image(debayering=True)

            # Save as 0.tif
            background_path = angle_dir / "0.tif"
            TifWriterUtils.ome_writer(
                filename=str(background_path),
                pixel_size_um=hardware.core.get_pixel_size_um(),
                data=image,
            )

            logger.info(f"Saved background for {angle}° to {background_path}")

        logger.info("=== BACKGROUND ACQUISITION COMPLETE ===")
