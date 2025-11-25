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
            # Get Z position once (same for all tiles in initial configuration)
            z = core.get_position()

            with open(tile_config_path, "r") as f:
                for line in f:
                    pattern = r"^([\w\-\.]+); ; \(\s*([\-\d.]+),\s*([\-\d.]+)"
                    m = re.match(pattern, line)
                    if m:
                        filename = m.group(1)
                        x = float(m.group(2))
                        y = float(m.group(3))
                        # Use same Z for all tiles (avoids 100+ hardware calls)
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
        # Use average FOV dimension (not diagonal) for consistent spacing
        af_min_distance = ((fov_x + fov_y) / 2) * n_tiles

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

    @staticmethod
    def has_sufficient_tissue(
        image: np.ndarray,
        texture_threshold: float = 0.02,
        tissue_area_threshold: float = 0.15,
        modality: Optional[str] = None,
        logger=None,
        return_stats: bool = False,
        rgb_brightness_threshold: float = 225.0,
    ):
        """
        Determine if image has sufficient tissue texture for reliable autofocus.

        Args:
            image: Input image (grayscale or RGB)
            texture_threshold: Minimum texture variance (normalized)
            tissue_area_threshold: Minimum fraction of image that must contain tissue
            modality: Imaging modality for modality-specific adjustments
            logger: Optional logger instance
            return_stats: If True, return (bool, dict) with detection statistics
            rgb_brightness_threshold: Maximum average RGB brightness for tissue (default 225).
                Images brighter than this are considered blank/background. Set to None to disable.

        Returns:
            If return_stats=False: True if sufficient tissue is present for autofocus
            If return_stats=True: (bool, dict) where dict contains detection statistics
        """
        # EARLY REJECTION: Check RGB brightness to filter out blank tiles
        # Blank glass/background is very bright (RGB ~230-240)
        # Tissue is darker (RGB ~190-224)
        rgb_mean = None
        brightness_rejected = False

        if rgb_brightness_threshold is not None and len(image.shape) == 3:
            # Calculate mean RGB across entire image
            rgb_mean = np.mean(image, axis=(0, 1))
            avg_brightness = np.mean(rgb_mean)

            if avg_brightness > rgb_brightness_threshold:
                brightness_rejected = True
                if logger:
                    logger.info(
                        f"Blank tile detected: avg RGB brightness {avg_brightness:.1f} > {rgb_brightness_threshold:.1f} "
                        f"(RGB: [{rgb_mean[0]:.1f}, {rgb_mean[1]:.1f}, {rgb_mean[2]:.1f}])"
                    )

                if return_stats:
                    stats = {
                        "texture": 0.0,
                        "texture_threshold": texture_threshold,
                        "area": 0.0,
                        "area_threshold": tissue_area_threshold,
                        "sufficient_texture": False,
                        "sufficient_area": False,
                        "rgb_mean": rgb_mean.tolist() if rgb_mean is not None else None,
                        "avg_brightness": float(avg_brightness),
                        "brightness_threshold": rgb_brightness_threshold,
                        "brightness_rejected": True,
                    }
                    return False, stats
                else:
                    return False
        # Modality-specific parameter adjustments
        if modality:
            modality_lower = modality.lower()

            # Polarized light microscopy adjustments
            if "ppm" in modality_lower or "polarized" in modality_lower:
                # Polarized images can have wider intensity ranges and different tissue appearance
                # More inclusive tissue mask to capture birefringent structures
                tissue_mask_range = (0.05, 0.95)  # Wider range
                if texture_threshold == 0.02:  # Only adjust if using default
                    texture_threshold = 0.015  # Slightly more sensitive

            # Brightfield microscopy
            elif "bf" in modality_lower or "brightfield" in modality_lower:
                # Standard tissue detection works well for brightfield
                tissue_mask_range = (0.15, 0.85)  # Focus on typical tissue intensity

            # Multi-photon or SHG
            elif "shg" in modality_lower or "multiphoton" in modality_lower:
                # High contrast features, different background characteristics
                tissue_mask_range = (0.1, 0.9)
                if texture_threshold == 0.02:
                    texture_threshold = 0.025  # Slightly less sensitive due to sparse features

            else:
                # Default mask range
                tissue_mask_range = (0.1, 0.9)
        else:
            tissue_mask_range = (0.1, 0.9)
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            img_gray = np.mean(image, axis=2).astype(np.float32)
        elif len(image.shape) == 2:
            # Handle Bayer pattern
            if image.shape[0] % 2 == 0 and image.shape[1] % 2 == 0:
                green1 = image[0::2, 0::2]
                green2 = image[1::2, 1::2]
                img_gray = ((green1 + green2) / 2.0).astype(np.float32)
            else:
                img_gray = image.astype(np.float32)
        else:
            img_gray = image.astype(np.float32)

        # Normalize image to [0, 1] range
        img_norm = (img_gray - img_gray.min()) / (img_gray.max() - img_gray.min() + 1e-10)
        norm_p5 = np.percentile(img_norm, 5)
        norm_p95 = np.percentile(img_norm, 95)

        if logger:
            logger.debug(f"Normalized percentiles - 5th: {norm_p5:.3f}, 95th: {norm_p95:.3f}")

        # Adaptive tissue mask based on actual data distribution
        if norm_p95 - norm_p5 < 0.5:  # Very narrow distribution
            # Use percentile-based mask for low contrast images
            margin = 0.02
            tissue_mask = (img_norm > norm_p5 + margin) & (img_norm < norm_p95 - margin)
            if logger:
                logger.debug(
                    f"Using adaptive mask for narrow range: ({norm_p5 + margin:.3f}, {norm_p95 - margin:.3f})"
                )
        else:
            # Original modality-specific masks
            if modality and "ppm" in modality.lower():
                tissue_mask = (img_norm > 0.05) & (img_norm < 0.95)
            else:
                tissue_mask = (img_norm > 0.1) & (img_norm < 0.9)
        # Calculate local texture using gradient magnitude
        gy, gx = np.gradient(img_norm)
        gradient_magnitude = np.sqrt(gx**2 + gy**2)

        # Calculate overall texture strength
        texture_strength = np.std(gradient_magnitude)

        # Identify potential tissue regions using modality-specific intensity ranges
        tissue_mask = (img_norm > tissue_mask_range[0]) & (img_norm < tissue_mask_range[1])

        # Calculate texture in tissue regions only
        if np.any(tissue_mask):
            tissue_texture = np.std(gradient_magnitude[tissue_mask])
            tissue_area_fraction = np.sum(tissue_mask) / tissue_mask.size
        else:
            tissue_texture = 0.0
            tissue_area_fraction = 0.0

        # Decision criteria
        sufficient_texture = tissue_texture > texture_threshold
        sufficient_area = tissue_area_fraction > tissue_area_threshold

        # Overall decision
        has_tissue = sufficient_texture and sufficient_area

        if logger:
            logger.debug(
                f"Tissue detection: texture={tissue_texture:.4f} (>{texture_threshold}), "
                f"area={tissue_area_fraction:.3f} (>{tissue_area_threshold}), "
                f"sufficient={has_tissue}"
            )

        if return_stats:
            stats = {
                "texture": tissue_texture,
                "texture_threshold": texture_threshold,
                "area": tissue_area_fraction,
                "area_threshold": tissue_area_threshold,
                "sufficient_texture": sufficient_texture,
                "sufficient_area": sufficient_area,
                "rgb_mean": rgb_mean.tolist() if rgb_mean is not None else None,
                "avg_brightness": float(np.mean(rgb_mean)) if rgb_mean is not None else None,
                "brightness_threshold": rgb_brightness_threshold,
                "brightness_rejected": brightness_rejected,
            }
            return has_tissue, stats
        else:
            return has_tissue

    @staticmethod
    def defer_autofocus_to_next_tile(
        current_pos_idx: int,
        original_af_positions: List[int],
        total_positions: int,
        af_min_distance: float,
        positions: List[Tuple[float, float]],
        logger=None,
    ) -> Optional[int]:
        """
        Find the next suitable tile position for autofocus when current tile lacks tissue.

        Args:
            current_pos_idx: Current tile index that was supposed to get autofocus
            original_af_positions: Original list of autofocus positions
            total_positions: Total number of tile positions
            af_min_distance: Minimum distance required between autofocus positions
            positions: List of (x, y) positions for all tiles
            logger: Optional logger instance

        Returns:
            Index of next tile to perform autofocus on, or None if no suitable tile found
        """
        if not positions or current_pos_idx >= len(positions):
            return None

        current_xy = positions[current_pos_idx]

        # Look ahead for the next tile that's far enough away and within bounds
        for candidate_idx in range(current_pos_idx + 1, total_positions):
            candidate_xy = positions[candidate_idx]

            # Check distance from current position
            distance = cdist([current_xy], [candidate_xy])[0][0]

            if distance >= af_min_distance * 0.7:  # Slightly relax distance requirement
                if logger:
                    logger.info(
                        f"Deferring autofocus from tile {current_pos_idx} to tile {candidate_idx} "
                        f"(distance: {distance:.1f} >= {af_min_distance * 0.7:.1f})"
                    )
                return candidate_idx

        # If no suitable position found nearby, try to find any position beyond minimum distance
        for candidate_idx in range(current_pos_idx + 1, min(current_pos_idx + 10, total_positions)):
            if logger:
                logger.warning(
                    f"No ideal autofocus position found, using tile {candidate_idx} as backup"
                )
            return candidate_idx

        if logger:
            logger.warning(
                f"Could not find suitable autofocus deferral position after tile {current_pos_idx}"
            )

        return None

    @staticmethod
    def test_tissue_detection(
        image: np.ndarray,
        modality: str = "unknown",
        texture_thresholds: List[float] = [0.01, 0.02, 0.03, 0.05],
        area_thresholds: List[float] = [0.10, 0.15, 0.20, 0.25],
        show_analysis: bool = True,
        logger=None,
    ) -> Dict[str, Any]:
        """
        Test tissue detection function with different threshold combinations.

        Args:
            image: Input image to analyze
            modality: Imaging modality name for reporting
            texture_thresholds: List of texture thresholds to test
            area_thresholds: List of area thresholds to test
            show_analysis: Whether to show detailed analysis
            logger: Optional logger instance

        Returns:
            Dictionary with analysis results and recommendations
        """
        import matplotlib.pyplot as plt

        # Convert to grayscale for analysis
        if len(image.shape) == 3:
            img_gray = np.mean(image, axis=2).astype(np.float32)
        elif len(image.shape) == 2:
            if image.shape[0] % 2 == 0 and image.shape[1] % 2 == 0:
                green1 = image[0::2, 0::2]
                green2 = image[1::2, 1::2]
                img_gray = ((green1 + green2) / 2.0).astype(np.float32)
            else:
                img_gray = image.astype(np.float32)
        else:
            img_gray = image.astype(np.float32)

        # Normalize image
        img_norm = (img_gray - img_gray.min()) / (img_gray.max() - img_gray.min() + 1e-10)

        # Calculate gradient and tissue metrics
        gy, gx = np.gradient(img_norm)
        gradient_magnitude = np.sqrt(gx**2 + gy**2)

        # Intensity analysis for modality-specific insights
        intensity_stats = {
            "min": float(img_norm.min()),
            "max": float(img_norm.max()),
            "mean": float(img_norm.mean()),
            "std": float(img_norm.std()),
            "median": float(np.median(img_norm)),
        }

        # Gradient analysis
        gradient_stats = {
            "mean": float(gradient_magnitude.mean()),
            "std": float(gradient_magnitude.std()),
            "max": float(gradient_magnitude.max()),
            "p95": float(np.percentile(gradient_magnitude, 95)),
        }

        # Test different tissue masks for modality analysis
        tissue_masks = {
            "conservative": (img_norm > 0.1) & (img_norm < 0.9),  # Original
            "brightfield_like": (img_norm > 0.2) & (img_norm < 0.8),  # Typical tissue range
            "polarized_inclusive": (img_norm > 0.05)
            & (img_norm < 0.95),  # Wider range for polarized
            "high_contrast": (img_norm > 0.15) & (img_norm < 0.85),  # Focus on mid-range
        }

        mask_analysis = {}
        for mask_name, mask in tissue_masks.items():
            if np.any(mask):
                mask_texture = np.std(gradient_magnitude[mask])
                mask_area = np.sum(mask) / mask.size
            else:
                mask_texture = 0.0
                mask_area = 0.0

            mask_analysis[mask_name] = {"texture": mask_texture, "area_fraction": mask_area}

        # Test threshold combinations
        results_matrix = []
        for tex_thresh in texture_thresholds:
            for area_thresh in area_thresholds:
                result = AutofocusUtils.has_sufficient_tissue(
                    image, tex_thresh, area_thresh, logger=None
                )
                results_matrix.append(
                    {
                        "texture_threshold": tex_thresh,
                        "area_threshold": area_thresh,
                        "has_tissue": result,
                    }
                )

        # Analysis summary
        analysis_summary = {
            "modality": modality,
            "image_shape": image.shape,
            "intensity_stats": intensity_stats,
            "gradient_stats": gradient_stats,
            "mask_analysis": mask_analysis,
            "threshold_results": results_matrix,
            "recommendations": {},
        }

        # Generate recommendations based on analysis
        best_mask = max(mask_analysis.keys(), key=lambda k: mask_analysis[k]["texture"])
        analysis_summary["recommendations"] = {
            "best_tissue_mask": best_mask,
            "suggested_texture_threshold": max(0.01, gradient_stats["std"] * 0.5),
            "suggested_area_threshold": max(0.1, mask_analysis[best_mask]["area_fraction"] * 0.5),
            "intensity_range": f"{intensity_stats['min']:.3f} - {intensity_stats['max']:.3f}",
            "has_good_contrast": intensity_stats["std"] > 0.15,
        }

        if show_analysis and logger:
            logger.info(f"=== TISSUE DETECTION TEST: {modality.upper()} ===")
            logger.info(f"Image shape: {image.shape}")
            logger.info(
                f"Intensity range: {intensity_stats['min']:.3f} - {intensity_stats['max']:.3f} (std: {intensity_stats['std']:.3f})"
            )
            logger.info(
                f"Gradient stats: mean={gradient_stats['mean']:.4f}, std={gradient_stats['std']:.4f}"
            )

            logger.info("Tissue mask analysis:")
            for mask_name, stats in mask_analysis.items():
                logger.info(
                    f"  {mask_name}: texture={stats['texture']:.4f}, area={stats['area_fraction']:.3f}"
                )

            logger.info("Threshold test results:")
            for result in results_matrix:
                status = "PASS" if result["has_tissue"] else "FAIL"
                logger.info(
                    f"  tex={result['texture_threshold']:.3f}, area={result['area_threshold']:.3f} -> {status}"
                )

            logger.info(f"Recommendations:")
            logger.info(f"  Best mask: {analysis_summary['recommendations']['best_tissue_mask']}")
            logger.info(
                f"  Suggested texture threshold: {analysis_summary['recommendations']['suggested_texture_threshold']:.4f}"
            )
            logger.info(
                f"  Suggested area threshold: {analysis_summary['recommendations']['suggested_area_threshold']:.3f}"
            )

        return analysis_summary

    @staticmethod
    def validate_focus_peak(z_positions: np.ndarray, scores: np.ndarray) -> Dict[str, Any]:
        """
        Validate that the focus curve has a proper peak suitable for autofocus.

        A good focus peak should have:
        1. A clear maximum that stands out from neighboring values
        2. Gradual increase leading up to the peak
        3. Gradual decrease after the peak
        4. Reasonable symmetry around the peak

        Args:
            z_positions: Array of Z positions sampled
            scores: Array of focus scores at each position

        Returns:
            Dict containing:
                - is_valid: bool - Whether peak passes quality checks
                - peak_prominence: float - How much peak stands out (0-1 normalized)
                - has_ascending: bool - Has increasing trend before peak
                - has_descending: bool - Has decreasing trend after peak
                - symmetry_score: float - Measure of left/right symmetry (0-1, 1=perfect)
                - quality_score: float - Overall quality score (0-1)
                - warnings: List[str] - List of quality issues found
                - message: str - Human-readable summary
        """
        result = {
            "is_valid": False,
            "peak_prominence": 0.0,
            "has_ascending": False,
            "has_descending": False,
            "symmetry_score": 0.0,
            "quality_score": 0.0,
            "warnings": [],
            "message": ""
        }

        if len(scores) < 5:
            result["warnings"].append("Too few samples for reliable peak detection")
            result["message"] = "Insufficient data points for peak validation"
            return result

        # Find peak position
        peak_idx = np.argmax(scores)
        peak_score = scores[peak_idx]
        mean_score = np.mean(scores)
        min_score = np.min(scores)
        max_score = np.max(scores)
        score_range = max_score - min_score
        score_std = np.std(scores)

        # 1. Check absolute score variation (detect flat/noisy curves)
        # A proper focus curve should have significant variation
        relative_range = score_range / mean_score if mean_score > 0 else 0

        # CRITICAL: Check for minimum absolute variation
        # If score range is too small, it's just noise with no real focus gradient
        # Note: These are conservative thresholds - adjust based on your microscope/metric
        MIN_ABSOLUTE_RANGE = 0.5   # Minimum score range (was 2.0, too strict)
        MIN_RELATIVE_RANGE = 0.005 # Minimum 0.5% variation (was 5%, too strict)

        if score_range < MIN_ABSOLUTE_RANGE:
            result["warnings"].append(
                f"Insufficient absolute score variation ({score_range:.2f} < {MIN_ABSOLUTE_RANGE})")
            result["message"] = f"No focus gradient detected - score range too small ({score_range:.2f})"
            return result

        if relative_range < MIN_RELATIVE_RANGE:
            result["warnings"].append(
                f"Insufficient relative score variation ({relative_range:.2%} < {MIN_RELATIVE_RANGE:.0%})")
            result["message"] = f"No focus gradient detected - scores too flat ({relative_range:.2%} variation)"
            return result

        # 2. Check peak prominence (how much it stands out within the range)
        result["peak_prominence"] = (peak_score - mean_score) / score_range

        if result["peak_prominence"] < 0.2:
            result["warnings"].append(f"Peak prominence too low ({result['peak_prominence']:.2f})")

        # 3. Check for ascending trend before peak
        if peak_idx >= 2:
            # Count how many points before peak show increasing trend
            ascending_count = 0
            for i in range(peak_idx):
                if i == 0 or scores[i] >= scores[i-1]:
                    ascending_count += 1
            result["has_ascending"] = (ascending_count / peak_idx) >= 0.5
        else:
            result["warnings"].append("Peak too close to start - cannot verify ascending trend")
            result["has_ascending"] = False

        # 4. Check for descending trend after peak
        if peak_idx < len(scores) - 2:
            # Count how many points after peak show decreasing trend
            descending_count = 0
            for i in range(peak_idx + 1, len(scores)):
                if scores[i] <= scores[i-1]:
                    descending_count += 1
            points_after = len(scores) - peak_idx - 1
            result["has_descending"] = (descending_count / points_after) >= 0.5
        else:
            result["warnings"].append("Peak too close to end - cannot verify descending trend")
            result["has_descending"] = False

        # 5. Check symmetry around peak
        # Compare left and right side score ranges
        left_scores = scores[:peak_idx] if peak_idx > 0 else np.array([])
        right_scores = scores[peak_idx+1:] if peak_idx < len(scores)-1 else np.array([])

        if len(left_scores) > 0 and len(right_scores) > 0:
            left_range = np.max(left_scores) - np.min(left_scores) if len(left_scores) > 1 else 0
            right_range = np.max(right_scores) - np.min(right_scores) if len(right_scores) > 1 else 0

            if left_range + right_range > 0:
                result["symmetry_score"] = 1.0 - abs(left_range - right_range) / (left_range + right_range)
            else:
                result["symmetry_score"] = 1.0  # Both sides flat = perfect symmetry
        else:
            result["warnings"].append("Peak at edge - cannot assess symmetry")
            result["symmetry_score"] = 0.0

        # 6. Calculate overall quality score
        weights = {
            "prominence": 0.4,
            "ascending": 0.2,
            "descending": 0.2,
            "symmetry": 0.2
        }

        result["quality_score"] = (
            weights["prominence"] * result["peak_prominence"] +
            weights["ascending"] * (1.0 if result["has_ascending"] else 0.0) +
            weights["descending"] * (1.0 if result["has_descending"] else 0.0) +
            weights["symmetry"] * result["symmetry_score"]
        )

        # 7. Determine if peak is valid (passes minimum quality threshold)
        MIN_QUALITY_THRESHOLD = 0.5
        MIN_PROMINENCE = 0.15

        result["is_valid"] = (
            result["quality_score"] >= MIN_QUALITY_THRESHOLD and
            result["peak_prominence"] >= MIN_PROMINENCE and
            (result["has_ascending"] or result["has_descending"])  # At least one side must show trend
        )

        # 8. Generate human-readable message
        if result["is_valid"]:
            result["message"] = f"Valid focus peak detected (quality: {result['quality_score']:.2f})"
        else:
            issues = []
            if result["quality_score"] < MIN_QUALITY_THRESHOLD:
                issues.append(f"low quality score ({result['quality_score']:.2f})")
            if result["peak_prominence"] < MIN_PROMINENCE:
                issues.append(f"weak peak ({result['peak_prominence']:.2f})")
            if not result["has_ascending"] and not result["has_descending"]:
                issues.append("no clear focus trend")
            result["message"] = "Invalid focus peak: " + ", ".join(issues)

        return result


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
            # JPEG compression only works with 8-bit RGB or grayscale
            # Use LZW for 16-bit data
            is_16bit = data.dtype == np.uint16
            compression = "lzw" if is_16bit else "jpeg"

            options = {
                "photometric": "rgb" if len(data.shape) == 3 else "minisblack",
                "compression": compression,
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
        pos_background: np.ndarray = None,
        neg_background: np.ndarray = None,
    ) -> np.ndarray:
        """
        Create a single birefringence image from positive and negative angle images.

        When background images are provided, per-channel scaling is applied to
        ensure that non-birefringent regions (backgrounds) produce near-zero
        birefringence values. This corrects for optical path differences between
        positive and negative polarization angles.

        Args:
            pos_image: Positive angle image (+7 or +5 degrees)
            neg_image: Negative angle image (-7 or -5 degrees)
            output_dir: Directory to save birefringence image
            filename: Output filename
            pixel_size_um: Pixel size for OME-TIFF metadata
            tile_config_source: Path to source TileConfiguration.txt to copy (optional)
            logger: Logger instance (optional)
            pos_background: Background image for positive angle (optional)
            neg_background: Background image for negative angle (optional)

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

        # Calculate birefringence (sum of absolute differences)
        output_path = output_dir / filename

        # Pass backgrounds for per-channel correction if available
        biref_img = TifWriterUtils.ppm_angle_difference(
            pos_image, neg_image, pos_background, neg_background
        )

        if logger and pos_background is not None and neg_background is not None:
            # Log the correction that was applied
            pos_bg_mean = pos_background.astype(np.float32).mean()
            neg_bg_mean = neg_background.astype(np.float32).mean()
            scale_factor = pos_bg_mean / neg_bg_mean if neg_bg_mean > 0 else 1.0
            logger.info(
                f"  Biref background correction: "
                f"pos_bg_mean={pos_bg_mean:.1f}, neg_bg_mean={neg_bg_mean:.1f}, "
                f"scale_factor={scale_factor:.4f}"
            )

        # Save as 16-bit single-channel image (no normalization)
        # Range: 0-765 (sum of absolute RGB differences)
        TifWriterUtils.ome_writer(
            filename=str(output_path),
            pixel_size_um=pixel_size_um,
            data=biref_img,  # Single channel uint16
        )

        if logger:
            logger.info(f"  Created birefringence image: {filename} (16-bit, range: {biref_img.min()}-{biref_img.max()})")

        return biref_img

    @staticmethod
    def subtraction_image(pos_image: np.ndarray, neg_image: np.ndarray) -> np.ndarray:
        """Simple subtraction of two images."""
        return pos_image.astype(np.float32) - neg_image.astype(np.float32)

    @staticmethod
    def ppm_angle_difference(
        img1: np.ndarray,
        img2: np.ndarray,
        bg1: np.ndarray = None,
        bg2: np.ndarray = None,
    ) -> np.ndarray:
        """
        Calculate angle difference for polarized microscopy images.
        Sum of absolute differences across RGB channels.

        When background images are provided, calculates a scaling factor to
        match the overall intensity of img2 to img1. This ensures that
        non-birefringent regions (backgrounds) produce near-zero birefringence.

        Note: Flat-field correction normalizes images toward the overall bg_mean,
        making per-channel values uniform. Therefore, we use overall (not per-channel)
        scaling to avoid introducing artificial per-channel differences.

        Args:
            img1: First image (RGB, uint8) - typically positive angle (+7)
            img2: Second image (RGB, uint8) - typically negative angle (-7)
            bg1: Optional background for img1 (RGB, uint8)
            bg2: Optional background for img2 (RGB, uint8)

        Returns:
            Difference image as single channel (uint16), range 0-765 (3 * 255)
        """
        # Convert to float32 for calculations
        img1_f = img1.astype(np.float32)
        img2_f = img2.astype(np.float32)

        # Apply background-based intensity correction if backgrounds provided
        if bg1 is not None and bg2 is not None:
            # Calculate overall mean intensities of backgrounds
            # (Not per-channel, because flat-field correction normalizes all channels
            # to the same overall mean, making per-channel scaling incorrect)
            bg1_mean = bg1.astype(np.float32).mean()
            bg2_mean = bg2.astype(np.float32).mean()

            if bg2_mean > 0:
                # Scale img2 to match img1's background intensity level
                # After flat-field correction:
                #   - img1 background regions -> ~bg1_mean
                #   - img2 background regions -> ~bg2_mean
                # We want: img2_scaled background -> ~bg1_mean
                scale_factor = bg1_mean / bg2_mean
                img2_f = img2_f * scale_factor

        # Calculate absolute difference per channel and sum across RGB
        # |R1-R2| + |G1-G2| + |B1-B2|
        abs_diff = np.abs(img1_f - img2_f)
        sum_abs_diff = np.sum(abs_diff, axis=2)

        # Convert to uint16 (range is 0 to 765, well within uint16)
        return np.clip(sum_abs_diff, 0, 765).astype(np.uint16)

    @staticmethod
    def ppm_angle_sum(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """
        Calculate angle sum for polarized microscopy images.
        Adds the two images to create a combined RGB image.

        Args:
            img1: First image (RGB)
            img2: Second image (RGB)

        Returns:
            Sum image as RGB
        """
        # Convert to float for calculations
        img1_f = img1.astype(np.float32) / 255.0
        img2_f = img2.astype(np.float32) / 255.0

        # Sum RGB values (average to keep values in [0,1] range)
        rgb_sum = (img1_f + img2_f) / 2.0

        return rgb_sum

    @staticmethod
    def create_sum_tile(
        pos_image: np.ndarray,
        neg_image: np.ndarray,
        output_dir: pathlib.Path,
        filename: str,
        pixel_size_um: float,
        tile_config_source: Optional[pathlib.Path] = None,
        logger=None,
    ) -> np.ndarray:
        """
        Create a single sum image from positive and negative angle images.

        Args:
            pos_image: Positive angle image
            neg_image: Negative angle image
            output_dir: Directory to save sum image
            filename: Output filename
            pixel_size_um: Pixel size for OME-TIFF metadata
            tile_config_source: Path to source TileConfiguration.txt to copy (optional)
            logger: Logger instance (optional)

        Returns:
            The sum image array
        """
        # Create output directory if it doesn't exist
        if not output_dir.exists():
            output_dir.mkdir(exist_ok=True)

            # Copy TileConfiguration.txt if source provided
            if tile_config_source and tile_config_source.exists():
                shutil.copy2(tile_config_source, output_dir / "TileConfiguration.txt")
                if logger:
                    logger.debug(f"Copied TileConfiguration.txt to {output_dir}")

        # Calculate sum
        output_path = output_dir / filename

        sum_img = TifWriterUtils.ppm_angle_sum(pos_image, neg_image)

        # Save float version for reference
        tf.imwrite(str(output_path)[:-4] + "_float.tif", sum_img.astype(np.float32))

        # Normalize to 8-bit RGB
        sum_img = sum_img * 255
        sum_img = np.clip(sum_img, 0, 255).astype(np.uint8)

        # Save as RGB (keep full color)
        TifWriterUtils.ome_writer(
            filename=str(output_path),
            pixel_size_um=pixel_size_um,
            data=sum_img,
        )

        if logger:
            logger.info(f"  Created sum image: {filename}")

        return sum_img

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
        background_dir: pathlib.Path,
        angles: List[float],
        logger=None,
        modality: Optional[str] = None,
    ) -> Tuple[Dict[float, np.ndarray], Dict[float, float], Dict[float, List[float]]]:
        """
        Load background images and calculate consistent scaling factors for each angle.

        Supports multiple directory structures:
        - Direct angle files: background_dir/angle.tif
        - Angle subdirectories: background_dir/angle/background.tif
        - Modality subdirectories: background_dir/modality/angle/background.tif

        Args:
            background_dir: Directory containing background images
            angles: List of angles to load backgrounds for
            logger: Optional logger instance
            modality: Optional modality subdirectory name

        Returns:
            Tuple of (background_images_dict, scaling_factors_dict, white_balance_dict)
        """
        backgrounds = {}
        scaling_factors = {}
        white_balance_coeffs = {}

        if logger:
            logger.info(f"Loading background images from: {background_dir}")
            if modality:
                logger.info(f"Using modality subdirectory: {modality}")

        # Determine search directory
        search_dir = background_dir / modality if modality else background_dir

        if modality and not search_dir.exists():
            if logger:
                logger.error(f"Modality directory not found: {search_dir}")
            return backgrounds, scaling_factors, white_balance_coeffs

        for angle in angles:
            background_found = False
            attempted_paths = []
            background_file = None

            # Search order: most specific to most general
            search_paths = [
                # Direct angle file
                search_dir / f"{angle}.tif",
                # Angle subdirectory
                search_dir / str(angle) / "background.tif",
            ]

            for path in search_paths:
                attempted_paths.append(str(path))
                if path.exists():
                    background_found = True
                    background_file = path
                    break

            if background_found:
                try:
                    background_img = tf.imread(str(background_file))
                    backgrounds[angle] = background_img

                    if logger:
                        logger.info(f"  [OK] Loaded background for {angle}°: {background_file}")

                    # Calculate the scaling factor for this background
                    bg_float = background_img.astype(np.float32)
                    bg_mean_all = bg_float.mean()

                    if angle == 90.0:  # Brightfield
                        # Scale to make background bright
                        target_intensity = 240.0
                        scaling_factor = target_intensity / bg_mean_all if bg_mean_all > 0 else 1.0
                        if logger:
                            logger.info(f"    Background mean intensity at 90°: {bg_mean_all:.1f}")
                    else:  # Polarized angles (-7, 0, 7)
                        # Preserve the physical intensity level - only correct spatial variation
                        scaling_factor = 1.0  # No intensity scaling - preserve polarization physics
                        if logger:
                            logger.info(
                                f"    Background mean intensity at {angle}°: {bg_mean_all:.1f}"
                            )
                            logger.info(
                                f"    No intensity scaling for {angle}° (preserves polarization physics)"
                            )

                    scaling_factors[angle] = scaling_factor

                    # Calculate white balance from background
                    bg_mean = background_img.mean(axis=(0, 1))  # Mean R,G,B
                    if len(bg_mean) >= 3 and bg_mean[1] > 0:  # Check G channel
                        # Calculate relative gains to normalize to green channel
                        white_balance_coeffs[angle] = [
                            bg_mean[1] / bg_mean[0] if bg_mean[0] > 0 else 1.0,  # R correction
                            1.0,  # G reference
                            bg_mean[1] / bg_mean[2] if bg_mean[2] > 0 else 1.0,  # B correction
                        ]
                        if logger:
                            logger.info(
                                f"    White balance coeffs for {angle}°: {white_balance_coeffs[angle]}"
                            )
                    else:
                        white_balance_coeffs[angle] = [1.0, 1.0, 1.0]

                except Exception as e:
                    if logger:
                        logger.error(f"  [ERROR] Failed to load background for {angle}°: {e}")
            else:
                if logger:
                    logger.warning(f"  [FAIL] Background not found for {angle}°")
                    logger.warning(f"    Searched paths:")
                    for path in attempted_paths:
                        logger.warning(f"      - {path}")

        if logger and backgrounds:
            logger.info(f"Successfully loaded {len(backgrounds)}/{len(angles)} background images")
        elif logger:
            logger.warning("No background images were loaded")

        # ======= BIREFRINGENCE PAIR MATCHING =======
        # For paired polarization angles (+7/-7 or +5/-5), calculate differential
        # scaling factors so their corrected backgrounds have matching intensities.
        # This is critical for birefringence calculations where we compute |pos - neg|.
        # Without this, the different bg_mean values cause non-zero biref backgrounds.
        polarization_pairs = [(7.0, -7.0), (5.0, -5.0)]

        for pos_angle, neg_angle in polarization_pairs:
            if pos_angle in backgrounds and neg_angle in backgrounds:
                pos_bg_mean = backgrounds[pos_angle].astype(np.float32).mean()
                neg_bg_mean = backgrounds[neg_angle].astype(np.float32).mean()

                if neg_bg_mean > 0:
                    # Scale the negative angle to match positive angle intensity
                    # After flat-field: corrected = image * (bg_mean / bg_pixel) * scaling_factor
                    # We want: pos_corrected_bg == neg_corrected_bg
                    # pos normalizes to pos_bg_mean, neg normalizes to neg_bg_mean
                    # So: neg_scaling_factor = pos_bg_mean / neg_bg_mean
                    differential_scale = pos_bg_mean / neg_bg_mean
                    scaling_factors[neg_angle] = differential_scale

                    if logger:
                        logger.info(
                            f"  Birefringence pair matching: {pos_angle}/{neg_angle} degrees"
                        )
                        logger.info(
                            f"    +{abs(pos_angle)} bg_mean: {pos_bg_mean:.1f}, "
                            f"-{abs(neg_angle)} bg_mean: {neg_bg_mean:.1f}"
                        )
                        logger.info(
                            f"    Differential scaling for {neg_angle}: {differential_scale:.4f}"
                        )
                        logger.info(
                            f"    Expected corrected background intensity: ~{pos_bg_mean:.1f} for both"
                        )

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

        # Track if we need to transpose the result back to original image shape
        original_shape = image.shape
        transposed_image = False

        # Handle shape mismatch - background may be (C,H,W) while image is (H,W,C)
        if img_float.shape != bg_float.shape:
            if len(bg_float.shape) == 3 and len(img_float.shape) == 3:
                # Check if we need to transpose from (C,H,W) to (H,W,C)
                if (
                    bg_float.shape[0] == img_float.shape[2]
                    and bg_float.shape[1:] == img_float.shape[:2]
                ):
                    bg_float = np.transpose(bg_float, (1, 2, 0))  # (C,H,W) -> (H,W,C)
                # Or check if we need to transpose from (H,W,C) to (C,H,W)
                elif (
                    bg_float.shape[:2] == img_float.shape[1:]
                    and bg_float.shape[2] == img_float.shape[0]
                ):
                    img_float = np.transpose(img_float, (2, 0, 1))  # (H,W,C) -> (C,H,W)
                    transposed_image = True

        # Ensure shapes match after potential transpose
        if img_float.shape != bg_float.shape:
            raise ValueError(
                f"Shape mismatch after transpose attempt: image {img_float.shape} vs background {bg_float.shape}"
            )

        # Prevent division by zero with small epsilon
        epsilon = 0.1
        bg_float = np.where(bg_float < epsilon, epsilon, bg_float)

        if method == "divide":
            # Proper flatfield correction: normalize by background mean to preserve brightness
            bg_mean = bg_float.mean()

            # The correction formula: Image * (background_mean / background_pixel)
            # This normalizes illumination while preserving overall brightness
            corrected = img_float * (bg_mean / bg_float)

            # Apply the scaling factor for consistency across tiles
            corrected = corrected * scaling_factor

            # For debugging: check if background has illumination variation
            bg_std = bg_float.std()
            bg_variation = bg_std / bg_mean
            if bg_variation < 0.05:  # Less than 5% variation
                # Background is too uniform - may not be proper flatfield reference
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(
                    f"Background image has low variation (std/mean = {bg_variation:.3f}) - may not correct illumination properly"
                )

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

        # Clip to valid range
        corrected = np.clip(corrected, 0, max_val).astype(image.dtype)

        # Transpose back to original shape if we transposed the image
        if transposed_image:
            corrected = np.transpose(corrected, (1, 2, 0))  # (C,H,W) -> (H,W,C)

        return corrected

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
                hardware.set_psg_ticks(
                    angle
                )  # , is_sequence_start=True)  # Single angle acquisition
                logger.info(f"Set angle to {angle}°")

            # Set exposure
            if angle_idx < len(exposures):
                hardware.set_exposure(exposures[angle_idx])
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


class PolarizerCalibrationUtils:
    """
    Utilities for calibrating polarized light microscopy (PPM) rotation stage.

    This class provides methods for determining crossed polarizer positions,
    which are critical for PPM imaging. These calibration functions should be
    run infrequently - only when optical components or rotation stage are
    physically repositioned or replaced.

    Note on Angle Conventions:
        - OPTICAL ANGLES: User-facing angles in degrees (0-360 deg)
        - HARDWARE POSITIONS: Motor encoder counts (device-specific units)

        For PI Stage: hardware_position = (optical_angle * 1000) + ppm_pizstage_offset
        For Thor Stage: hardware_position = -2 * optical_angle + 276

        The calibration function calculates the hardware offset needed to align
        the optical reference (0 deg) with a physical crossed polarizer position.
    """

    @staticmethod
    def find_crossed_polarizer_positions(
        hardware,
        start_angle: float = 0.0,
        end_angle: float = 360.0,
        step_size: float = 5.0,
        exposure_ms: float = 10.0,
        channel: int = 1,
        min_prominence: float = 0.1,
        logger_instance = None
    ) -> Dict[str, Any]:
        """
        Calibrate polarizer by sweeping rotation angles and finding crossed positions.

        This function rotates the polarization stage through a range of angles,
        captures images at each position, measures intensity, fits the data to
        a sinusoidal function, and identifies local minima corresponding to
        crossed polarizer orientations.

        **When to use this:**
            - After installing or repositioning polarization optics
            - After reseating or replacing the rotation stage
            - When validating polarizer alignment
            - To verify/update rotation_angles in config_PPM.yml

        **NOT needed for:**
            - Regular imaging sessions
            - Between samples
            - After software updates

        Args:
            hardware: PycromanagerHardware instance with PPM methods.
            start_angle: Starting optical angle in degrees (default: 0.0).
            end_angle: Ending optical angle in degrees (default: 360.0).
            step_size: Angular step size in degrees (default: 5.0).
            exposure_ms: Camera exposure time in milliseconds (default: 10.0).
            channel: Which RGB channel to analyze (0=R, 1=G, 2=B, None=mean).
            min_prominence: Minimum prominence for peak detection (default: 0.1).
            logger_instance: Logger for output messages.

        Returns:
            Dictionary containing:
                - 'angles': np.ndarray of optical angles tested
                - 'intensities': np.ndarray of mean intensities
                - 'minima_angles': List of crossed polarizer angles
                - 'minima_intensities': List of intensities at minima
                - 'fit_params': Fitted sine parameters
                - 'fit_curve': Fitted intensity values

        Raises:
            AttributeError: If hardware lacks PPM methods.
            RuntimeError: If image acquisition fails.
            ValueError: If no minima detected.
        """
        # Use scipy imports locally to avoid import issues if not available
        from scipy.optimize import curve_fit
        from scipy.signal import find_peaks

        if logger_instance is None:
            logger_instance = logger

        # Verify hardware has PPM methods
        if not hasattr(hardware, 'set_psg_ticks') or not hasattr(hardware, 'get_psg_ticks'):
            raise AttributeError(
                "Hardware does not have PPM methods initialized. "
                "Check that ppm_optics is set in configuration and not 'NA'."
            )

        # Generate angle array
        angles = np.arange(start_angle, end_angle + step_size, step_size)
        intensities = []

        # Set exposure
        hardware.set_exposure(exposure_ms)

        logger_instance.info(
            f"Starting polarizer calibration sweep: "
            f"{start_angle} deg to {end_angle} deg in {step_size} deg steps"
        )
        logger_instance.info(f"Exposure: {exposure_ms} ms, Channel: {channel if channel is not None else 'mean'}")
        logger_instance.info(f"Expected duration: ~{len(angles) * 0.5:.0f} seconds")

        # Sweep through optical angles
        for i, angle in enumerate(angles):
            # Set rotation angle
            hardware.set_psg_ticks(angle)

            # Capture image
            try:
                img, tags = hardware.snap_image(debayering=True)
            except Exception as e:
                raise RuntimeError(f"Image acquisition failed at angle {angle} deg: {e}")

            # Calculate mean intensity
            if channel is not None and len(img.shape) == 3:
                intensity = float(np.mean(img[:, :, channel]))
            else:
                intensity = float(np.mean(img))

            intensities.append(intensity)

            # Progress indicator
            if (i + 1) % 10 == 0 or i == 0:
                logger_instance.info(f"  Angle {angle:.1f} deg: intensity = {intensity:.1f}")

        intensities = np.array(intensities)
        logger_instance.info(f"Intensity range: {intensities.min():.1f} to {intensities.max():.1f}")

        # Normalize intensities
        intensities_norm = (intensities - intensities.min()) / (intensities.max() - intensities.min())

        # Define sine function
        def sine_func(x, amplitude, frequency, phase, offset):
            return amplitude * np.sin(2 * np.pi * frequency * x + phase) + offset

        # Initial guess (180 deg period for polarizers)
        initial_guess = [0.5, 1.0/180.0, 0.0, 0.5]

        # Fit sine function
        try:
            popt, pcov = curve_fit(sine_func, angles, intensities_norm, p0=initial_guess)
            fit_curve_norm = sine_func(angles, *popt)
            fit_curve = fit_curve_norm * (intensities.max() - intensities.min()) + intensities.min()

            logger_instance.info("Sine fit successful:")
            logger_instance.info(f"  Amplitude: {popt[0]:.4f}")
            logger_instance.info(f"  Frequency: {popt[1]:.6f} (period: {1/popt[1]:.1f} deg)")
            logger_instance.info(f"  Phase: {popt[2]:.4f} rad ({np.degrees(popt[2]):.1f} deg)")
            logger_instance.info(f"  Offset: {popt[3]:.4f}")
        except Exception as e:
            logger_instance.warning(f"Sine fit failed: {e}. Using initial guess.")
            popt = initial_guess
            fit_curve_norm = sine_func(angles, *popt)
            fit_curve = fit_curve_norm * (intensities.max() - intensities.min()) + intensities.min()

        # Find local minima
        inverted = -intensities_norm
        peaks, properties = find_peaks(inverted, prominence=min_prominence, distance=len(angles)/10)

        if len(peaks) == 0:
            raise ValueError(
                "No minima detected. Try adjusting: "
                "exposure_ms, step_size, min_prominence, or angular_range"
            )

        minima_angles = angles[peaks].tolist()
        minima_intensities = intensities[peaks].tolist()

        logger_instance.info(f"Found {len(minima_angles)} crossed polarizer positions:")
        for angle, intensity in zip(minima_angles, minima_intensities):
            logger_instance.info(f"  {angle:.1f} deg: intensity = {intensity:.1f}")

        return {
            'angles': angles,
            'intensities': intensities,
            'minima_angles': minima_angles,
            'minima_intensities': minima_intensities,
            'fit_params': popt,
            'fit_curve': fit_curve
        }

    @staticmethod
    def calibrate_hardware_offset_two_stage(
        hardware,
        coarse_range_deg: float = 360.0,
        coarse_step_deg: float = 5.0,
        fine_range_deg: float = 10.0,
        fine_step_deg: float = 0.1,
        exposure_ms: float = 10.0,
        channel: int = 1,
        logger_instance = None
    ) -> Dict[str, Any]:
        """
        Two-stage calibration to determine exact hardware offset for PPM rotation stage.

        This function performs precise hardware position calibration in two stages:
        1. Coarse sweep: Find approximate locations of crossed polarizer minima
        2. Fine sweep: Determine exact hardware encoder positions for each minimum

        The result provides the exact hardware position that should be set as
        ppm_pizstage_offset in config_PPM.yml.

        **CRITICAL**: This calculates the hardware offset itself, not optical angles.
        Run this ONLY when:
            - Installing or repositioning rotation stage hardware
            - After optical component changes
            - When ppm_pizstage_offset needs recalibration

        Args:
            hardware: PycromanagerHardware instance with PPM methods.
            coarse_range_deg: Full range to sweep in coarse stage (default: 360.0 deg).
            coarse_step_deg: Step size for coarse sweep (default: 5.0 deg).
            fine_range_deg: Range around each minimum for fine sweep (default: 10.0 deg, increased for optical stability).
            fine_step_deg: Step size for fine sweep (default: 0.1 deg).
            exposure_ms: Camera exposure time in milliseconds (default: 10.0).
            channel: Which RGB channel to analyze (0=R, 1=G, 2=B, None=mean).
            logger_instance: Logger for output messages.

        Returns:
            Dictionary containing:
                - 'rotation_device': Name of rotation device (PIZStage or Thor)
                - 'coarse_hardware_positions': Hardware positions tested in coarse sweep
                - 'coarse_intensities': Intensities from coarse sweep
                - 'approximate_minima': Approximate hardware positions of minima
                - 'fine_results': List of dicts with fine sweep data for each minimum
                - 'exact_minima': List of exact hardware positions at intensity minima
                - 'recommended_offset': Hardware position to use as ppm_pizstage_offset
                - 'optical_angles': Optical angles corresponding to exact minima

        Raises:
            AttributeError: If hardware lacks PPM methods or rotation device.
            RuntimeError: If image acquisition fails.
            ValueError: If fewer than 2 minima detected (expected for 360 deg sweep).
        """
        from scipy.optimize import curve_fit
        from scipy.signal import find_peaks

        if logger_instance is None:
            logger_instance = logger

        # Verify hardware has PPM methods
        if not hasattr(hardware, 'rotation_device'):
            raise AttributeError(
                "Hardware does not have rotation_device attribute. "
                "Check that PPM is properly initialized."
            )

        rotation_device = hardware.rotation_device
        logger_instance.info(f"=== TWO-STAGE HARDWARE OFFSET CALIBRATION ===")
        logger_instance.info(f"Rotation device: {rotation_device}")

        # Get current hardware position as reference
        current_hw_pos = hardware.core.get_position(rotation_device)
        logger_instance.info(f"Current hardware position: {current_hw_pos:.1f}")

        # Determine conversion factor based on device type
        if rotation_device == "PIZStage":
            # For PI: 1 deg optical = 1000 encoder counts
            hw_per_deg = 1000.0
            logger_instance.info("PI Stage detected: 1 deg = 1000 encoder counts")
        elif rotation_device == "KBD101_Thor_Rotation":
            # For Thor: Uses ppm_psgticks_to_thor conversion (-2x + 276)
            # For sweep purposes, we treat it as 2 counts per degree
            hw_per_deg = 2.0
            logger_instance.info("Thor Stage detected: 1 deg = 2 encoder counts (approx)")
        else:
            raise ValueError(f"Unknown rotation device: {rotation_device}")

        # ===== STAGE 1: COARSE SWEEP =====
        logger_instance.info("\n--- STAGE 1: COARSE SWEEP ---")

        # Calculate hardware range for coarse sweep
        coarse_hw_range = coarse_range_deg * hw_per_deg
        coarse_hw_step = coarse_step_deg * hw_per_deg

        # Center sweep on current position
        coarse_start_hw = current_hw_pos - (coarse_hw_range / 2)
        coarse_end_hw = current_hw_pos + (coarse_hw_range / 2)

        coarse_hw_positions = np.arange(coarse_start_hw, coarse_end_hw + coarse_hw_step, coarse_hw_step)
        coarse_intensities = []

        hardware.set_exposure(exposure_ms)

        logger_instance.info(
            f"Sweeping {coarse_start_hw:.0f} to {coarse_end_hw:.0f} "
            f"in steps of {coarse_hw_step:.0f} ({len(coarse_hw_positions)} positions)"
        )
        logger_instance.info(f"Expected duration: ~{len(coarse_hw_positions) * 0.5:.0f} seconds")

        for i, hw_pos in enumerate(coarse_hw_positions):
            # Set hardware position directly
            hardware.core.set_position(rotation_device, hw_pos)
            hardware.core.wait_for_device(rotation_device)

            # Capture image
            try:
                img, tags = hardware.snap_image(debayering=True)
            except Exception as e:
                raise RuntimeError(f"Image acquisition failed at hardware position {hw_pos:.0f}: {e}")

            # Calculate mean intensity
            if channel is not None and len(img.shape) == 3:
                intensity = float(np.mean(img[:, :, channel]))
            else:
                intensity = float(np.mean(img))

            coarse_intensities.append(intensity)

            # Progress indicator
            if (i + 1) % 10 == 0 or i == 0:
                logger_instance.info(f"  Position {hw_pos:.0f}: intensity = {intensity:.1f}")

        coarse_intensities = np.array(coarse_intensities)
        logger_instance.info(
            f"Coarse sweep complete. Intensity range: {coarse_intensities.min():.1f} to "
            f"{coarse_intensities.max():.1f}"
        )

        # Fit sine curve to coarse data
        intensities_norm = (coarse_intensities - coarse_intensities.min()) / (
            coarse_intensities.max() - coarse_intensities.min()
        )

        def sine_func(x, amplitude, frequency, phase, offset):
            return amplitude * np.sin(2 * np.pi * frequency * x + phase) + offset

        # Initial guess for sine fit (180 deg period)
        period_hw = 180.0 * hw_per_deg
        initial_guess = [0.5, 1.0/period_hw, 0.0, 0.5]

        try:
            popt, _ = curve_fit(sine_func, coarse_hw_positions, intensities_norm, p0=initial_guess)
            logger_instance.info("Sine fit successful")
        except Exception as e:
            logger_instance.warning(f"Sine fit failed: {e}. Using initial guess.")
            popt = initial_guess

        # Find local minima in coarse sweep
        inverted = -intensities_norm
        min_distance = int(len(coarse_hw_positions) / 4)  # At least 90 deg apart

        # Try with default prominence first
        peaks, properties = find_peaks(inverted, prominence=0.1, distance=min_distance)

        # If we didn't find 2 minima, try with lower prominence
        if len(peaks) < 2:
            logger_instance.warning(
                f"Only found {len(peaks)} minimum with prominence=0.1. "
                "Retrying with lower prominence threshold..."
            )
            peaks, properties = find_peaks(inverted, prominence=0.05, distance=min_distance)

        # If still only one, try finding the global minimum in opposite half
        if len(peaks) < 2:
            logger_instance.warning(
                f"Still only found {len(peaks)} minimum. "
                "Searching for second minimum in opposite 180deg region..."
            )
            # If we have one peak, look for minimum in opposite half of sweep
            if len(peaks) == 1:
                first_min_idx = peaks[0]
                half_len = len(coarse_intensities) // 2

                # Search in opposite half
                if first_min_idx < half_len:
                    # First minimum in first half, search second half
                    second_half_min_idx = half_len + np.argmin(coarse_intensities[half_len:])
                else:
                    # First minimum in second half, search first half
                    second_half_min_idx = np.argmin(coarse_intensities[:half_len])

                peaks = np.array([peaks[0], second_half_min_idx])
                logger_instance.info(f"  Found second minimum at index {second_half_min_idx}")

        approximate_minima = coarse_hw_positions[peaks].tolist()
        logger_instance.info(f"Found {len(approximate_minima)} approximate minima:")
        for hw_pos in approximate_minima:
            logger_instance.info(f"  Hardware position: {hw_pos:.0f}")

        # ===== STAGE 2: FINE SWEEP AROUND EACH MINIMUM =====
        logger_instance.info("\n--- STAGE 2: FINE SWEEP ---")

        fine_hw_range = fine_range_deg * hw_per_deg
        fine_hw_step = fine_step_deg * hw_per_deg

        fine_results = []
        exact_minima = []

        for min_idx, approx_hw_pos in enumerate(approximate_minima):
            logger_instance.info(f"\nFine sweep {min_idx + 1}/{len(approximate_minima)}:")
            logger_instance.info(f"  Centered on hardware position: {approx_hw_pos:.0f}")

            # Calculate fine sweep range
            fine_start_hw = approx_hw_pos - (fine_hw_range / 2)
            fine_end_hw = approx_hw_pos + (fine_hw_range / 2)

            fine_hw_positions = np.arange(fine_start_hw, fine_end_hw + fine_hw_step, fine_hw_step)
            fine_intensities = []

            logger_instance.info(
                f"  Sweeping {fine_start_hw:.1f} to {fine_end_hw:.1f} "
                f"in steps of {fine_hw_step:.1f} ({len(fine_hw_positions)} positions)"
            )

            for hw_pos in fine_hw_positions:
                hardware.core.set_position(rotation_device, hw_pos)
                hardware.core.wait_for_device(rotation_device)

                try:
                    img, tags = hardware.snap_image(debayering=True)
                except Exception as e:
                    raise RuntimeError(f"Fine sweep image acquisition failed at {hw_pos:.1f}: {e}")

                if channel is not None and len(img.shape) == 3:
                    intensity = float(np.mean(img[:, :, channel]))
                else:
                    intensity = float(np.mean(img))

                fine_intensities.append(intensity)

            fine_intensities = np.array(fine_intensities)

            # Find exact minimum
            min_idx_local = np.argmin(fine_intensities)
            exact_hw_pos = fine_hw_positions[min_idx_local]
            exact_intensity = fine_intensities[min_idx_local]

            exact_minima.append(exact_hw_pos)

            logger_instance.info(f"  Exact minimum found:")
            logger_instance.info(f"    Hardware position: {exact_hw_pos:.1f}")
            logger_instance.info(f"    Intensity: {exact_intensity:.1f}")

            fine_results.append({
                'approximate_position': approx_hw_pos,
                'fine_hw_positions': fine_hw_positions,
                'fine_intensities': fine_intensities,
                'exact_position': exact_hw_pos,
                'exact_intensity': exact_intensity
            })

        # ===== CALCULATE RECOMMENDATIONS =====
        logger_instance.info("\n--- CALIBRATION RESULTS ---")

        # Sort minima by hardware position
        exact_minima_sorted = sorted(exact_minima)

        # Recommend the minimum closest to current offset as reference (0 deg)
        # For PI stage, current offset is approximately current_hw_pos
        recommended_offset = exact_minima_sorted[0]

        # Calculate optical angles for all minima relative to recommended offset
        optical_angles = []
        for hw_pos in exact_minima_sorted:
            if rotation_device == "PIZStage":
                optical_angle = (hw_pos - recommended_offset) / hw_per_deg
            elif rotation_device == "KBD101_Thor_Rotation":
                # Thor uses: hw_pos = -2 * angle + 276
                # Need to account for this in angle calculation
                optical_angle = (hw_pos - recommended_offset) / hw_per_deg
            else:
                optical_angle = 0.0
            optical_angles.append(optical_angle)

        logger_instance.info(f"Recommended ppm_pizstage_offset: {recommended_offset:.1f}")
        logger_instance.info(f"Exact minima positions (hardware):")
        for i, (hw_pos, opt_angle) in enumerate(zip(exact_minima_sorted, optical_angles)):
            logger_instance.info(f"  Minimum {i+1}: {hw_pos:.1f} ({opt_angle:.2f} deg optical)")

        return {
            'rotation_device': rotation_device,
            'coarse_hardware_positions': coarse_hw_positions,
            'coarse_intensities': coarse_intensities,
            'approximate_minima': approximate_minima,
            'fine_results': fine_results,
            'exact_minima': exact_minima_sorted,
            'recommended_offset': recommended_offset,
            'optical_angles': optical_angles,
            'hw_per_deg': hw_per_deg
        }

    @staticmethod
    def calibrate_hardware_offset_with_stability_check(
        hardware,
        num_runs: int = 3,
        stability_threshold_counts: float = 50.0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run hardware offset calibration multiple times to check optical stability.

        Performs the two-stage calibration multiple times in succession and validates
        that results are consistent. This helps identify optical instability issues
        such as loose mounts, thermal drift, or mechanical backlash.

        Args:
            hardware: PycromanagerHardware instance
            num_runs: Number of calibration runs to perform (default: 3)
            stability_threshold_counts: Maximum acceptable variation in encoder counts (default: 50.0 = 0.05deg)
            **kwargs: Additional arguments passed to calibrate_hardware_offset_two_stage

        Returns:
            Dictionary with calibration results plus stability metrics:
                - 'all_runs': List of all calibration results
                - 'recommended_offset': Average offset from all runs
                - 'offset_std': Standard deviation of offsets (stability metric)
                - 'offset_range': Max - min offsets (stability metric)
                - 'is_stable': Boolean indicating if variation < threshold
                - 'stability_warning': Warning message if unstable

        Raises:
            RuntimeError: If optical instability exceeds threshold
        """
        logger_instance = kwargs.get('logger_instance', logger)

        logger_instance.info("="*70)
        logger_instance.info("POLARIZER CALIBRATION WITH STABILITY CHECK")
        logger_instance.info("="*70)
        logger_instance.info(f"Running {num_runs} calibrations to validate optical stability")
        logger_instance.info(f"Stability threshold: +/-{stability_threshold_counts:.1f} encoder counts")

        all_results = []
        all_offsets = []

        for run_num in range(1, num_runs + 1):
            logger_instance.info(f"\n{'='*70}")
            logger_instance.info(f"CALIBRATION RUN {run_num}/{num_runs}")
            logger_instance.info(f"{'='*70}")

            result = PolarizerCalibrationUtils.calibrate_hardware_offset_two_stage(
                hardware, **kwargs
            )

            all_results.append(result)
            all_offsets.append(result['recommended_offset'])

            logger_instance.info(f"Run {run_num} completed: offset = {result['recommended_offset']:.1f}")

            # Brief pause between runs to allow hardware to settle
            if run_num < num_runs:
                import time
                time.sleep(2.0)

        # Calculate stability metrics
        all_offsets = np.array(all_offsets)

        # Get hardware conversion factor from first run
        hw_per_deg = all_results[0]['hw_per_deg']
        full_rotation_counts = 360.0 * hw_per_deg  # e.g., 360000 for PI stage

        # Normalize offsets to a single 360-degree range
        # The stage may have rotated multiple full turns between runs
        normalized_offsets = all_offsets % full_rotation_counts

        # Calculate statistics on normalized offsets
        mean_offset = np.mean(all_offsets)  # Keep original mean for recommended offset
        normalized_mean = np.mean(normalized_offsets)
        std_offset = np.std(normalized_offsets)  # Use normalized for std
        range_offset = np.max(normalized_offsets) - np.min(normalized_offsets)  # Use normalized for range

        logger_instance.info(f"\n{'='*70}")
        logger_instance.info("STABILITY ANALYSIS")
        logger_instance.info(f"{'='*70}")
        logger_instance.info(f"Raw offsets from {num_runs} runs: {all_offsets}")
        logger_instance.info(f"Normalized offsets (within 0-360 deg): {normalized_offsets}")
        logger_instance.info(f"Mean offset (for config): {mean_offset:.1f}")
        logger_instance.info(f"Normalized mean: {normalized_mean:.1f}")
        logger_instance.info(f"Std deviation: {std_offset:.2f} counts ({std_offset/1000:.4f} deg)")
        logger_instance.info(f"Range (max-min): {range_offset:.1f} counts ({range_offset/1000:.4f} deg)")

        is_stable = range_offset <= stability_threshold_counts

        if is_stable:
            logger_instance.info(f"RESULT: STABLE - Variation {range_offset:.1f} counts within threshold {stability_threshold_counts:.1f}")
        else:
            warning_msg = (
                f"WARNING: OPTICAL INSTABILITY DETECTED!\n"
                f"  Variation: {range_offset:.1f} counts ({range_offset/1000:.3f} deg)\n"
                f"  Threshold: {stability_threshold_counts:.1f} counts ({stability_threshold_counts/1000:.3f} deg)\n"
                f"  Possible causes:\n"
                f"    - Loose polarizer/analyzer mounts\n"
                f"    - Thermal drift in optical components\n"
                f"    - Mechanical backlash in rotation stage\n"
                f"    - Vibration or external disturbances\n"
                f"  Recommendation: Check hardware before proceeding with acquisitions"
            )
            logger_instance.warning(warning_msg)

        return {
            'all_runs': all_results,
            'recommended_offset': float(normalized_mean),  # Use normalized mean for recommended offset
            'offset_std': float(std_offset),
            'offset_range': float(range_offset),
            'is_stable': is_stable,
            'stability_warning': None if is_stable else warning_msg,
            'individual_offsets': all_offsets.tolist(),
            'normalized_offsets': normalized_offsets.tolist(),  # Add normalized offsets
            'rotation_device': all_results[0]['rotation_device'],
            'hw_per_deg': all_results[0]['hw_per_deg']
        }
