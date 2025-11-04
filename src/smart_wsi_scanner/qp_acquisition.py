"""Acquisition workflow and microscope-side operations for QuPath server.

This module contains the acquisition logic and helpers that interact with the
microscope hardware, separated from the socket server/transport logic.
"""

from __future__ import annotations

import time
from typing import Callable, List, Tuple, Optional, Dict, Any
from pathlib import Path
import shutil
import logging
import yaml

import numpy as np

from .hardware import Position
from .hardware_pycromanager import PycromanagerHardware
from .qp_utils import BackgroundCorrectionUtils, TileConfigUtils, TifWriterUtils, AutofocusUtils
import shlex
import skimage.filters

logger = logging.getLogger(__name__)


def calculate_luminance_gain(r, g, b):
    """Calculate luminance-based gain from RGB values."""
    return 0.299 * r + 0.587 * g + 0.114 * b


def parse_angles_exposures(angles_str, exposures_str=None) -> Tuple[List[float], List[int]]:
    """Parse angle and exposure strings from various formats."""
    angles: List[float] = []
    exposures: List[int] = []

    # Parse angles
    if isinstance(angles_str, list):
        angles = angles_str
    elif isinstance(angles_str, str):
        angles_str = angles_str.strip("()")
        if "," in angles_str:
            angles = [float(x.strip()) for x in angles_str.split(",")]
        elif angles_str:
            angles = [float(x) for x in angles_str.split()]

    # Parse exposures if provided
    if exposures_str:
        if isinstance(exposures_str, list):
            exposures = exposures_str
        elif isinstance(exposures_str, str):
            exposures_str = exposures_str.strip("()")
            if "," in exposures_str:
                exposures = [float(x.strip()) for x in exposures_str.split(",")]
            elif exposures_str:
                exposures = [float(x) for x in exposures_str.split()]

    # Default exposures if not provided
    if not exposures and angles:
        for angle in angles:
            if angle == 90.0:
                exposures.append(10.0)
            elif angle == 0.0:
                exposures.append(800.0)
            else:
                exposures.append(500.0)

    return angles, exposures


def parse_acquisition_message(message: str) -> dict:
    """Parse acquisition message supporting both legacy and new flag-based formats."""
    # Remove END_MARKER if present
    message = message.replace(" END_MARKER", "").replace("END_MARKER", "").strip()

    # Check if it's flag-based format
    if "--" in message:
        # Parse flag-based format
        params = {}

        # Split by spaces but preserve quoted strings
        try:
            # For Windows compatibility, temporarily replace backslashes
            temp_message = message.replace("\\", "|||BACKSLASH|||")
            parts = shlex.split(temp_message)
            # Restore backslashes
            parts = [part.replace("|||BACKSLASH|||", "\\") for part in parts]
        except Exception:
            # Fallback to simple split if shlex fails
            parts = message.split()

        i = 0
        while i < len(parts):
            if parts[i] == "--yaml" and i + 1 < len(parts):
                params["yaml_file_path"] = parts[i + 1]
                i += 2
            elif parts[i] == "--projects" and i + 1 < len(parts):
                params["projects_folder_path"] = parts[i + 1]
                i += 2
            elif parts[i] == "--sample" and i + 1 < len(parts):
                params["sample_label"] = parts[i + 1]
                i += 2
            elif parts[i] == "--scan-type" and i + 1 < len(parts):
                params["scan_type"] = parts[i + 1]
                i += 2
            elif parts[i] == "--region" and i + 1 < len(parts):
                params["region_name"] = parts[i + 1]
                i += 2
            elif parts[i] == "--angles" and i + 1 < len(parts):
                params["angles_str"] = parts[i + 1]
                i += 2
            elif parts[i] == "--exposures" and i + 1 < len(parts):
                params["exposures_str"] = parts[i + 1]
                i += 2
            elif parts[i] == "--bg-correction" and i + 1 < len(parts):
                params["background_correction_enabled"] = parts[i + 1].lower() == "true"
                i += 2
            elif parts[i] == "--bg-method" and i + 1 < len(parts):
                params["background_correction_method"] = parts[i + 1]
                i += 2
            elif parts[i] == "--bg-folder" and i + 1 < len(parts):
                params["background_folder"] = parts[i + 1]
                i += 2
            elif parts[i] == "--bg-disabled-angles" and i + 1 < len(parts):
                params["background_disabled_angles_str"] = parts[i + 1]
                i += 2
            elif parts[i] == "--white-balance" and i + 1 < len(parts):
                params["white_balance_enabled"] = parts[i + 1].lower() == "true"
                i += 2
            elif parts[i] == "--objective" and i + 1 < len(parts):
                params["objective"] = parts[i + 1]
                i += 2
            elif parts[i] == "--detector" and i + 1 < len(parts):
                params["detector"] = parts[i + 1]
                i += 2
            elif parts[i] == "--pixel-size" and i + 1 < len(parts):
                params["pixel_size"] = float(parts[i + 1])
                i += 2
            elif parts[i] == "--af-tiles" and i + 1 < len(parts):
                params["autofocus_tiles"] = int(parts[i + 1])
                i += 2
            elif parts[i] == "--af-steps" and i + 1 < len(parts):
                params["autofocus_steps"] = int(parts[i + 1])
                i += 2
            elif parts[i] == "--af-range" and i + 1 < len(parts):
                params["autofocus_range"] = float(parts[i + 1])
                i += 2
            elif parts[i] == "--processing" and i + 1 < len(parts):
                params["processing_pipeline"] = parts[i + 1]
                i += 2
            else:
                i += 1

        # Parse angles and exposures if present
        angles, exposures = parse_angles_exposures(
            params.get("angles_str", "()"), params.get("exposures_str", None)
        )
        params["angles"] = angles
        params["exposures"] = exposures

        # Parse disabled angles for background correction
        disabled_angles = []
        disabled_angles_str = params.get("background_disabled_angles_str", "()")
        if disabled_angles_str and disabled_angles_str != "()":
            disabled_angles_str = disabled_angles_str.strip("()")
            if "," in disabled_angles_str:
                disabled_angles = [float(x.strip()) for x in disabled_angles_str.split(",")]
            elif disabled_angles_str:
                disabled_angles = [float(x) for x in disabled_angles_str.split()]
        params["background_disabled_angles"] = disabled_angles

        # Validate required parameters
        required = [
            "yaml_file_path",
            "projects_folder_path",
            "sample_label",
            "scan_type",
            "region_name",
        ]
        missing = [key for key in required if key not in params]
        if missing:
            raise ValueError(f"Missing required parameters: {missing}")

        return params

    # Legacy format not implemented here
    raise ValueError("Unsupported acquisition message format")


def get_angles_wb_from_settings(settings: Dict[str, Any]) -> Dict[float, List[float]]:
    """Extract white balance values for different angles from settings."""
    angles_wb = {}

    # Try to find white balance settings
    wb_settings = settings.get("white_balance", {})
    ppm_wb = wb_settings.get("ppm", {})

    # Map standard angle names to numeric values
    angle_mapping = {"crossed": 0.0, "uncrossed": 90.0, "positive": 5.0, "negative": -5.0}

    for angle_name, angle_value in angle_mapping.items():
        if angle_name in ppm_wb:
            wb_values = ppm_wb[angle_name]
            # Handle different formats
            if isinstance(wb_values, list):
                if len(wb_values) > 0 and isinstance(wb_values[0], str):
                    # Format: ["1.0 1.0 1.0"]
                    angles_wb[angle_value] = [float(x) for x in wb_values[0].split()]
                else:
                    # Format: [1.0, 1.0, 1.0]
                    angles_wb[angle_value] = wb_values
            elif isinstance(wb_values, str):
                # Format: "1.0 1.0 1.0"
                angles_wb[angle_value] = [float(x) for x in wb_values.split()]

    # Default fallback values if not found
    if not angles_wb:
        logger.warning("No white balance settings found, using defaults")
        angles_wb = {
            0.0: [1.0, 1.0, 1.0],  # crossed
            90.0: [1.2, 1.0, 1.1],  # uncrossed
            5.0: [1.0, 1.0, 1.0],  # positive
            -5.0: [1.0, 1.0, 1.0],  # negative
        }

    return angles_wb


def _acquisition_workflow(
    message: str,
    client_addr,
    hardware: PycromanagerHardware,
    config_manager,
    logger,
    update_progress: Callable[[int, int], None],
    set_state: Callable[[str], None],
    is_cancelled: Callable[[], bool],
):
    """Execute the main image acquisition workflow with progress and cancellation."""

    logger.info(f"=== ACQUISITION WORKFLOW STARTED for client {client_addr} ===")

    try:
        # Parse the acquisition parameters
        params = parse_acquisition_message(message)

        modality = "_".join(params["scan_type"].split("_")[:2])

        logger.info("Acquisition parameters:")
        logger.info(f"  Client: {client_addr}")
        logger.info(f"  Modality: {modality}")
        logger.info(f"  Sample label: {params['sample_label']}")
        logger.info(f"  Scan type: {params['scan_type']}")
        logger.info(f"  Region: {params['region_name']}")
        logger.info(f"  Angles: {params['angles']} degrees")
        logger.info(f"  Exposures: {params['exposures']} ms")

        # Load the yaml file
        if not params["yaml_file_path"]:
            raise ValueError("YAML file path is required")
        if not Path(params["yaml_file_path"]).exists():
            raise FileNotFoundError(f"YAML file {params['yaml_file_path']} does not exist")

        # Load configuration using the config manager
        ppm_settings = config_manager.load_config_file(params["yaml_file_path"])
        loci_rsc_file = str(
            Path(params["yaml_file_path"]).parent / "resources" / "resources_LOCI.yml"
        )
        loci_resources = config_manager.load_config_file(loci_rsc_file)
        ppm_settings.update(loci_resources)
        hardware.settings = ppm_settings

        # Home rot-stage
        # hardware.home_psg()

        # Extract modality from scan type
        modality = BackgroundCorrectionUtils.get_modality_from_scan_type(params["scan_type"])
        logger.info(f"Using modality: {modality}")

        # Get processing settings from parameters
        background_correction_enabled = params.get("background_correction_enabled", False)
        background_correction_method = params.get("background_correction_method", "divide")
        background_disabled_angles = params.get("background_disabled_angles", [])
        white_balance_enabled = params.get("white_balance_enabled", True)

        # Log background correction configuration
        if background_correction_enabled:
            logger.info(
                f"Background correction enabled with method: {background_correction_method}"
            )
            if background_disabled_angles:
                logger.info(
                    f"Background correction will be disabled for angles: {background_disabled_angles}"
                )
        else:
            logger.info("Background correction disabled")

        # ======= WARNING FOR BOTH CORRECTIONS ENABLED =======
        if background_correction_enabled and white_balance_enabled:
            logger.warning("=" * 70)
            logger.warning("WARNING: Both background correction and white balance are enabled!")
            logger.warning("This may lead to over-correction of the images.")
            logger.warning("Consider using only one correction method.")
            logger.warning("=" * 70)

        # ======= BACKGROUND CORRECTION SETUP =======
        background_images = {}
        background_scaling_factors = {}

        if background_correction_enabled:
            background_dir = None

            # Priority 1: Message parameter
            if "background_folder" in params:
                background_dir = Path(params["background_folder"])
                logger.info(f"Using background folder from message: {background_dir}")
            else:
                # Priority 2: YAML configuration
                modalities = ppm_settings.get("modalities", {})
                ppm_config = modalities.get("ppm", {})
                bc_settings = ppm_config.get("background_correction", {})

                if bc_settings.get("enabled") and bc_settings.get("base_folder"):
                    # For YAML config, construct path with modality subdirectory
                    background_dir = Path(bc_settings["base_folder"]) / modality
                    logger.info(f"Using background folder from YAML config: {background_dir}")

            # Load background images if directory is valid
            if background_dir and background_dir.exists():
                logger.info(f"Loading background images from: {background_dir}")
                background_images, background_scaling_factors, _ = (
                    BackgroundCorrectionUtils.load_background_images(
                        background_dir, params["angles"], logger
                    )
                )

                if background_images:
                    logger.info(f"Loaded {len(background_images)} background images")
                else:
                    logger.warning("No background images found - disabling background correction")
                    background_correction_enabled = False
            else:
                logger.warning(f"Background directory not found: {background_dir}")
                logger.warning("Disabling background correction")
                background_correction_enabled = False

        # ======= WHITE BALANCE SETUP =======
        angles_wb = {}

        if white_balance_enabled:
            # Load white balance settings from configuration
            angles_wb = get_angles_wb_from_settings(ppm_settings)
            logger.info(f"Loaded white balance settings for {len(angles_wb)} angles")

        # Set up output paths
        project_path = Path(params["projects_folder_path"]) / params["sample_label"]
        output_path = project_path / params["scan_type"] / params["region_name"]
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created output directory: {output_path}")

        # Read tile positions
        tile_config_path = output_path / "TileConfiguration.txt"
        positions = TileConfigUtils.read_tile_config(tile_config_path, hardware.core)

        if not positions:
            logger.error(f"No positions found in {tile_config_path}")
            set_state("FAILED")
            return

        xy_positions = [(pos.x, pos.y) for pos, filename in positions]
        # except Exception as e:
        #   logger.warning("Falling back to older tileconfig reader: %s", e)
        #   xy_positions = TileConfigUtils.read_TileConfiguration_coordinates(tile_config_path)

        # Create angle subdirectories
        if params["angles"]:
            for angle in params["angles"]:
                angle_dir = output_path / str(angle)
                angle_dir.mkdir(exist_ok=True)
                shutil.copy2(tile_config_path, angle_dir / "TileConfiguration.txt")

        # Calculate total images and update progress
        total_images = (
            len(positions) * len(params["angles"]) if params["angles"] else len(positions)
        )

        # check if total image is 720_psg_degs (should be 360, MSN tested 720) x number_of_tiles < MM2 limit
        # for DDR25 limit is 536870.9 (thorlabs)degs or 268573 (ticks)
        # that is 372 tiles
        total_rotation = 720 * len(positions)  #
        if total_rotation > 2**18:  # 262144
            logger.error(
                f"Total rotation steps {total_rotation} exceed Micro-Manager limit of 536870. Acquisition aborted."
            )

        starting_position = hardware.get_current_position()

        update_progress(0, total_images)
        logger.info(
            f"Starting acquisition of {total_images} total images "
            f"({len(positions)} positions × {len(params['angles'])} angles)"
        )

        image_count = 0

        # Find autofocus positions
        fov = hardware.get_fov()

        # Load autofocus settings from separate autofocus_{microscope}.yml file
        af_n_tiles = 5  # default
        af_search_range = 50  # default
        af_n_steps = 11  # default
        af_interp_strength = 100  # default
        af_interp_kind = "quadratic"  # default
        af_score_metric_name = "laplacian_variance"  # default

        # Try to get current objective from hardware
        microscope = ppm_settings.get("microscope", {})
        current_objective = microscope.get("objective_in_use", "")

        try:
            # Derive autofocus config path from main config path
            # e.g., "config_PPM.yml" -> "autofocus_PPM.yml"
            config_path = Path(params["yaml_file_path"])
            config_name = config_path.stem  # "config_PPM"
            microscope_name = config_name.replace("config_", "")  # "PPM"
            autofocus_file = config_path.parent / f"autofocus_{microscope_name}.yml"

            if autofocus_file.exists():
                with open(autofocus_file, "r") as f:
                    autofocus_config = yaml.safe_load(f)

                # Find settings for current objective
                af_settings_list = autofocus_config.get("autofocus_settings", [])
                for af_setting in af_settings_list:
                    if af_setting.get("objective") == current_objective:
                        af_n_tiles = af_setting.get("n_tiles", af_n_tiles)
                        af_search_range = af_setting.get("search_range_um", af_search_range)
                        af_n_steps = af_setting.get("n_steps", af_n_steps)
                        af_interp_strength = af_setting.get("interp_strength", af_interp_strength)
                        af_interp_kind = af_setting.get("interp_kind", af_interp_kind)
                        af_score_metric_name = af_setting.get("score_metric", af_score_metric_name)
                        logger.info(
                            f"Loaded autofocus settings for {current_objective}: "
                            f"n_steps={af_n_steps}, search_range={af_search_range}um, n_tiles={af_n_tiles}, "
                            f"interp_strength={af_interp_strength}, interp_kind={af_interp_kind}, "
                            f"score_metric={af_score_metric_name}"
                        )
                        break
            else:
                logger.warning(
                    f"Autofocus config file not found: {autofocus_file}. Using defaults."
                )
        except Exception as e:
            logger.error(f"Error loading autofocus settings: {e}. Using defaults.")

        # Map score metric name to function
        score_metric_map = {
            "laplacian_variance": AutofocusUtils.autofocus_profile_laplacian_variance,
            "sobel": AutofocusUtils.autofocus_profile_sobel,
            "brenner_gradient": AutofocusUtils.autofocus_profile_brenner_gradient,
            "robust_sharpness": AutofocusUtils.autofocus_profile_robust_sharpness_metric,
            "hybrid_sharpness": AutofocusUtils.autofocus_profile_hybrid_sharpness_metric,
        }
        af_score_metric = score_metric_map.get(
            af_score_metric_name, AutofocusUtils.autofocus_profile_laplacian_variance
        )

        af_positions, af_min_distance = AutofocusUtils.get_autofocus_positions(
            fov, xy_positions, n_tiles=af_n_tiles
        )

        logger.info(f"Autofocus positions: {af_positions}")

        # Create dynamic autofocus positions set (can be modified during acquisition)
        dynamic_af_positions = set(af_positions)
        deferred_af_positions = set()  # Track positions where AF was deferred

        metadata_txt_for_positions = output_path / "image_positions_metadata.txt"

        # Main acquisition loop
        for pos_idx, (pos, filename) in enumerate(positions):
            # Check for cancellation
            if is_cancelled():
                logger.warning(f"Acquisition cancelled by client {client_addr}")
                set_state("CANCELLED")
                return

            logger.info(f"Position {pos_idx + 1}/{len(positions)}: {filename}")

            # Ensure Z is current autofocus value
            pos.z = hardware.get_current_position().z

            # Move to position
            logger.info(f"Moving to position: X={pos.x}, Y={pos.y}, Z={pos.z}")
            hardware.move_to_position(pos)

            # Perform autofocus if needed (with tissue detection)
            if pos_idx in dynamic_af_positions:
                logger.info(
                    f"Checking for autofocus at position {pos_idx}: X={pos.x}, Y={pos.y}, Z={pos.z}"
                )

                # For PPM, always autofocus at 90° (uncrossed polarizers - brightest, fastest)
                # This ensures consistent, fast autofocus regardless of angle sequence
                if "ppm" in modality.lower():
                    hardware.set_psg_ticks(90.0)
                    logger.info("Set rotation to 90° (uncrossed) for PPM autofocus")
                    # CRITICAL: Set appropriate exposure for 90° before tissue detection
                    # Find the 90° exposure from acquisition parameters
                    exposure_90 = 2.0  # Default fallback

                    if 90.0 in params["angles"]:
                        angle_idx = params["angles"].index(90.0)
                        if angle_idx < len(params["exposures"]):
                            exposure_90 = params["exposures"][angle_idx]

                    hardware.set_exposure(exposure_90)
                    logger.info(f"Set exposure to {exposure_90}ms for 90° tissue detection")
                # Take a quick image to assess tissue content
                test_img, _ = hardware.snap_image()

                # Ensure consistent format for tissue detection
                if test_img.dtype in [np.float32, np.float64]:
                    # Check if already normalized (0-1 range)
                    if test_img.max() <= 1.0 and test_img.min() >= 0.0:
                        # Convert to uint8 to match expected format
                        test_img = (test_img * 255).astype(np.uint8)
                        logger.info(
                            "Converted normalized float image to uint8 for tissue detection"
                        )
                    else:
                        # Float but not normalized - clip and convert
                        test_img = np.clip(test_img, 0, 255).astype(np.uint8)
                        logger.info("Converted float image to uint8 for tissue detection")

                # Check if there's sufficient tissue for reliable autofocus
                # Pass modality for modality-specific thresholds
                has_tissue, tissue_stats = AutofocusUtils.has_sufficient_tissue(
                    test_img,
                    texture_threshold=0.010,
                    tissue_area_threshold=0.2,
                    modality=modality,
                    logger=logger,
                    return_stats=True,
                )

                if has_tissue:
                    logger.info(f"Sufficient tissue detected - performing autofocus")
                    logger.info(
                        f"  Tissue stats: texture={tissue_stats['texture']:.4f} (threshold={tissue_stats['texture_threshold']:.4f}), "
                        f"area={tissue_stats['area']:.3f} (threshold={tissue_stats['area_threshold']:.3f})"
                    )

                    new_z = hardware.autofocus(
                        move_stage_to_estimate=True,
                        n_steps=af_n_steps,
                        search_range=af_search_range,
                        interp_strength=af_interp_strength,
                        interp_kind=af_interp_kind,
                        score_metric=af_score_metric,
                    )
                    logger.info(f"  Autofocus :: New Z {new_z}")
                else:
                    logger.warning(
                        f"Insufficient tissue at position {pos_idx} - deferring autofocus"
                    )
                    logger.warning(
                        f"  Tissue stats: texture={tissue_stats['texture']:.4f} (threshold={tissue_stats['texture_threshold']:.4f}), "
                        f"area={tissue_stats['area']:.3f} (threshold={tissue_stats['area_threshold']:.3f})"
                    )

                    # Remove this position from autofocus list
                    dynamic_af_positions.discard(pos_idx)
                    deferred_af_positions.add(pos_idx)

                    # Try to find next suitable position for autofocus
                    next_af_pos = AutofocusUtils.defer_autofocus_to_next_tile(
                        current_pos_idx=pos_idx,
                        original_af_positions=af_positions,
                        total_positions=len(positions),
                        af_min_distance=af_min_distance,
                        positions=xy_positions,
                        logger=logger,
                    )

                    if next_af_pos is not None and next_af_pos < len(positions):
                        dynamic_af_positions.add(next_af_pos)
                        logger.info(f"Added position {next_af_pos} to autofocus queue")
                    else:
                        logger.warning(f"Could not find suitable position to defer autofocus to")

            if params["angles"]:
                # Storage for birefringence image calculation
                angle_images = {}

                # Multi-angle acquisition
                for angle_idx, angle in enumerate(params["angles"]):
                    # Check for cancellation
                    if is_cancelled():
                        logger.warning(f"Acquisition cancelled by client {client_addr}")
                        set_state("CANCELLED")
                        return

                    # Set rotation angle
                    # First angle of each position should reset to "a" polarization state
                    # is_sequence_start = angle_idx == 0
                    hardware.set_psg_ticks(angle)  # , is_sequence_start=is_sequence_start)

                    # Backup check of angle - seem to be having hardware issues sometimes
                    # actual_angle = hardware.get_psg_ticks()
                    # angle_diff = min(abs(actual_angle - angle), 360 - abs(actual_angle - angle))
                    # if angle_diff > 5.0:
                    #     logger.warning(f"  Angle mismatch: requested {angle:.1f}°, got {actual_angle:.1f}°, retrying...")
                    #     hardware.set_psg_ticks(angle, is_sequence_start=False)
                    #     time.sleep(0.15)
                    #     actual_angle = hardware.get_psg_ticks()
                    # logger.info(f"  Angle set to {hardware.get_psg_ticks():.1f}")

                    # Set exposure time if specified
                    if angle_idx < len(params["exposures"]):
                        exposure_ms = params["exposures"][angle_idx]
                        hardware.set_exposure(exposure_ms)
                    logger.info(f"  Exposure set to {hardware.core.get_exposure()}")

                    # Acquire image
                    image, metadata = hardware.snap_image(debayering=False)

                    if image is None:
                        logger.error(f"Failed to acquire image at angle {angle}")
                        continue

                    logger.info(f"  Image shape: {image.shape}, mean: {image.mean((0,1))}")

                    # Save raw (unprocessed) image for comparison
                    raw_output_path = output_path.parent / "Raw" / output_path.name
                    raw_image_path = raw_output_path / str(angle) / filename
                    if not raw_image_path.parent.exists():
                        raw_image_path.parent.mkdir(parents=True, exist_ok=True)

                    try:
                        TifWriterUtils.ome_writer(  # raw
                            filename=str(raw_image_path),
                            pixel_size_um=hardware.core.get_pixel_size_um(),
                            data=image,
                        )
                        logger.info(f"  Saved raw image: {raw_image_path}")
                        write_position_metadata(
                            metadata_txt_for_positions, raw_image_path, hardware, modality
                        )
                    except Exception as e:
                        logger.warning(f"  Failed to save raw image: {e}")

                    # ======= APPLY BACKGROUND CORRECTION (STEP 1) =======
                    # Check if background correction is enabled, background exists, and angle is not disabled
                    if (
                        background_correction_enabled
                        and angle in background_images
                        and angle not in background_disabled_angles
                    ):
                        bg_img = background_images[angle]
                        logger.info(f"  Applying background correction for {angle} degrees")
                        logger.info(
                            f"    Background stats: mean={bg_img.mean():.1f}, std={bg_img.std():.1f}"
                        )

                        image = BackgroundCorrectionUtils.apply_flat_field_correction(
                            image,
                            background_images[angle],
                            background_scaling_factors[angle],
                            method=background_correction_method,
                        )
                        logger.info(
                            f"    Correction applied with method: {background_correction_method}"
                        )
                        logger.info(f"    Post-correction RGB means: {image.mean(axis=(0,1))}")
                    elif background_correction_enabled and angle in background_disabled_angles:
                        logger.info(
                            f"  Background correction SKIPPED for {angle}° (validation failed - exposure mismatch or missing background)"
                        )
                    elif background_correction_enabled and angle not in background_images:
                        logger.info(
                            f"  Background correction SKIPPED for {angle}° (no background image available)"
                        )

                    # ======= APPLY WHITE BALANCE (STEP 2) =======
                    if white_balance_enabled:
                        # Use pre-configured white balance values
                        if angle in angles_wb:
                            wb_profile = angles_wb[angle]
                        else:
                            # Default neutral if angle not found
                            wb_profile = [1.0, 1.0, 1.0]
                            logger.warning(
                                f"    No white balance profile for {angle}°, using neutral"
                            )

                        gain = calculate_luminance_gain(*wb_profile)
                        image = hardware.white_balance(
                            image, white_balance_profile=wb_profile, gain=gain
                        )
                        logger.info(
                            f"  Applied white balance: R={wb_profile[0]:.2f}, G={wb_profile[1]:.2f}, B={wb_profile[2]:.2f}"
                        )

                    # Save processed image
                    image_path = output_path / str(angle) / filename

                    if image_path.parent.exists():
                        TifWriterUtils.ome_writer(  # processed
                            filename=str(image_path),
                            pixel_size_um=hardware.core.get_pixel_size_um(),
                            data=image,
                        )
                        image_count += 1
                        update_progress(image_count, total_images)

                        # Store image for birefringence calculation
                        angle_images[angle] = image
                    else:
                        logger.error(f"Failed to save {image_path} - parent directory missing")

                # Create birefringence image for this tile after all angles acquired
                positive_angles = [a for a in angle_images.keys() if a > 0 and a != 90]
                negative_angles = [a for a in angle_images.keys() if a < 0]

                if positive_angles and negative_angles:
                    pos_angle = min(positive_angles)
                    neg_angle = max(negative_angles)

                    # Set up birefringence directory and tile config source
                    biref_dir = output_path / f"{pos_angle}.biref"
                    tile_config_source = output_path / str(pos_angle) / "TileConfiguration.txt"

                    # Create birefringence image
                    TifWriterUtils.create_birefringence_tile(  # biref
                        pos_image=angle_images[pos_angle],
                        neg_image=angle_images[neg_angle],
                        output_dir=biref_dir,
                        filename=filename,
                        pixel_size_um=hardware.core.get_pixel_size_um(),
                        tile_config_source=tile_config_source,
                        logger=logger,
                    )

                    # # Create sum image alongside birefringence image
                    # sum_dir = output_path / f"{pos_angle}.sum"
                    # TifWriterUtils.create_sum_tile(
                    #     pos_image=angle_images[pos_angle],
                    #     neg_image=angle_images[neg_angle],
                    #     output_dir=sum_dir,
                    #     filename=filename,
                    #     pixel_size_um=hardware.core.get_pixel_size_um(),
                    #     tile_config_source=tile_config_source,
                    #     logger=logger,
                    # )

            else:
                # Single image acquisition: no angles specified
                image, metadata = hardware.snap_image()
                image_path = output_path / filename

                if image_path.parent.exists():
                    TifWriterUtils.ome_writer(  # brightfield
                        filename=str(image_path),
                        pixel_size_um=hardware.core.get_pixel_size_um(),
                        data=image,
                    )
                    image_count += 1
                    update_progress(image_count, total_images)

                try:
                    write_position_metadata(
                        metadata_txt_for_positions, image_path, hardware, modality
                    )
                except Exception as e:
                    logger.warning(
                        f"  Failed to write position text {metadata_txt_for_positions}: {e}"
                    )

        # Save device properties
        current_props = hardware.get_device_properties()
        props_path = output_path / "MMproperties.txt"
        with open(props_path, "w") as fid:
            from pprint import pprint as dict_printer

            dict_printer(current_props, stream=fid)

        set_state("COMPLETED")
        logger.info("=== ACQUISITION COMPLETED SUCCESSFULLY ===")
        logger.info(f"Total images saved: {image_count}/{total_images}")
        logger.info(f"Output directory: {output_path}")

        # Report autofocus activity
        if deferred_af_positions:
            logger.info(
                f"Autofocus deferred at {len(deferred_af_positions)} positions due to insufficient tissue: {sorted(deferred_af_positions)}"
            )
        # else:
        #    logger.info(
        #    f"Autofocus completed at {len([p for p in af_positions if p not in deferred_af_positions])} positions")

    except Exception as e:
        logger.error("=== ACQUISITION FAILED ===")
        logger.error(f"Error: {str(e)}", exc_info=True)
        set_state("FAILED")
    finally:
        # Return to starting position
        logger.info("Returning to starting position")
        hardware.move_to_position(starting_position)


def write_position_metadata(metadata_txt_for_positions, raw_image_path, hardware, modality):
    pos_read = hardware.get_current_position()
    line = (
        f"filename = {raw_image_path} ; "
        f"(x,y,z) = ({pos_read.x},{pos_read.y},{round(pos_read.z, 3)}); "
    )

    # TODO: modality is chosen on the config , but then overwrittebn by the message parameter
    # here we should use the modality used for acquisition?
    # if modality.lower.count("ppm") > 0:  # user-set value passed from acquisition parameters
    # if hardware.settings.get("modality", "ppm") == "ppm":  # config set value from yaml

    if "ppm" in modality.lower():
        angle = (
            hardware.get_psg_ticks()
            if hardware.settings.get("ppm_optics", "ZCutQuartz") != "NA"
            else "NA"
        )
        line += f"r = {angle} ; "

    line += f"exposure (ms) = {hardware.core.get_exposure()}\n"

    with open(metadata_txt_for_positions, "a") as f:
        f.write(line)


def get_target_intensity_for_background(modality: str, angle: float) -> float:
    """
    Get target intensity for background acquisition based on modality and angle.

    Args:
        modality: Modality identifier (e.g., "ppm", "brightfield")
        angle: Rotation angle in degrees (for PPM)

    Returns:
        Target grayscale intensity (0-255)

    Examples:
        >>> get_target_intensity_for_background("brightfield", 0)
        250.0
        >>> get_target_intensity_for_background("ppm", 90)
        245.0
        >>> get_target_intensity_for_background("ppm", 5)
        150.0
        >>> get_target_intensity_for_background("ppm", -5)
        150.0
        >>> get_target_intensity_for_background("ppm", 0)
        125.0
    """
    # Normalize modality to lowercase for comparison
    modality_lower = modality.lower()

    # Brightfield modality
    if "brightfield" in modality_lower or "bf" in modality_lower:
        return 250.0

    # PPM modality - angle-specific targets
    if "ppm" in modality_lower:
        # Normalize angle to absolute value for symmetric angles
        abs_angle = abs(angle)

        if abs_angle == 90:
            return 245.0
        elif abs_angle in [7, 5]:
            # Same target for positive and negative angles
            return 150.0  # ±7 or ±5 degrees
        elif abs_angle == 0:
            return 125.0
        else:
            # Default for unknown PPM angles
            logger.warning(f"Unknown PPM angle {angle}, using default target 150")
            return 150.0

    # Default fallback
    logger.warning(f"Unknown modality {modality}, using default target 200")
    return 200.0


def acquire_background_with_target_intensity(
    hardware: PycromanagerHardware,
    target_intensity: float,
    tolerance: float = 2.5,
    initial_exposure_ms: float = 100.0,
    max_iterations: int = 10,
    logger=None,
) -> Tuple[np.ndarray, float]:
    """
    Acquire background image with adaptive exposure to reach target intensity.

    Uses proportional control to iteratively adjust exposure time until the
    median image intensity is within tolerance of the target value. Median is
    used instead of mean as it is more robust to outliers and hot pixels.

    Args:
        hardware: Microscope hardware interface
        target_intensity: Target median grayscale value (0-255)
        tolerance: Acceptable deviation from target (default ±2.5)
        initial_exposure_ms: Starting exposure time in milliseconds
        max_iterations: Maximum adjustment iterations
        logger: Logger instance for tracking convergence

    Returns:
        Tuple of (image, final_exposure_ms)
            image: Acquired image at target intensity
            final_exposure_ms: Final exposure time used

    Raises:
        RuntimeError: If image acquisition fails
    """
    # Exposure bounds to prevent extreme values
    MIN_EXPOSURE_MS = 0.0001
    MAX_EXPOSURE_MS = 5000.0

    # Set initial exposure
    current_exposure = max(MIN_EXPOSURE_MS, min(MAX_EXPOSURE_MS, initial_exposure_ms))
    hardware.set_exposure(current_exposure)

    if logger:
        logger.info(
            f"Starting adaptive exposure: target={target_intensity:.1f}, "
            f"tolerance={tolerance:.1f}, initial_exposure={current_exposure:.1f}ms"
        )

    last_image = None
    last_exposure = current_exposure

    for iteration in range(max_iterations):
        # Snap image with debayering
        image, metadata = hardware.snap_image(debayering=True)

        if image is None:
            raise RuntimeError(f"Failed to acquire image at iteration {iteration}")

        # Calculate median intensity across all channels (more robust than mean)
        mean_intensity = float(np.median(image))

        # Store for potential use if we don't converge
        last_image = image
        last_exposure = current_exposure

        if logger:
            logger.info(
                f"  Iteration {iteration + 1}/{max_iterations}: "
                f"median={mean_intensity:.1f}, exposure={current_exposure:.1f}ms"
            )

        # Check convergence
        intensity_error = abs(mean_intensity - target_intensity)
        if intensity_error <= tolerance:
            if logger:
                logger.info(
                    f"Converged! Final: median={mean_intensity:.1f}, "
                    f"exposure={current_exposure:.1f}ms, iterations={iteration + 1}"
                )
            return image, current_exposure

        # Calculate proportional adjustment
        # If image is too dark, increase exposure; if too bright, decrease
        if mean_intensity >= 254.0:
            # Image is saturated - decrease exposure aggressively
            # Proportional control alone is too slow when saturated
            new_exposure = max(current_exposure * 0.5, MIN_EXPOSURE_MS)
            if logger:
                logger.warning(
                    f"    Image saturated (median={mean_intensity:.1f}), halving exposure to {new_exposure:.1f}ms"
                )
            current_exposure = new_exposure
            hardware.set_exposure(current_exposure)
        elif mean_intensity > 0:
            adjustment_ratio = target_intensity / mean_intensity
            new_exposure = current_exposure * adjustment_ratio

            # Clamp to bounds
            new_exposure = max(MIN_EXPOSURE_MS, min(MAX_EXPOSURE_MS, new_exposure))

            if logger:
                logger.info(
                    f"    Adjusting exposure: {current_exposure:.1f}ms -> {new_exposure:.1f}ms "
                    f"(ratio={adjustment_ratio:.2f})"
                )

            current_exposure = new_exposure
            hardware.set_exposure(current_exposure)
        else:
            # Image is completely black, increase exposure significantly
            new_exposure = min(current_exposure * 2.0, MAX_EXPOSURE_MS)
            if logger:
                logger.warning(
                    f"    Image completely black, doubling exposure to {new_exposure:.1f}ms"
                )
            current_exposure = new_exposure
            hardware.set_exposure(current_exposure)

    # Max iterations reached without convergence
    if logger:
        logger.warning(
            f"Did not converge after {max_iterations} iterations. "
            f"Using last image: median={float(np.median(last_image)):.1f}, exposure={last_exposure:.1f}ms"
        )

    return last_image, last_exposure


def simple_background_collection(
    yaml_file_path: str,
    output_folder_path: str,
    modality: str,
    angles_str: str,
    exposures_str: str,
    hardware: PycromanagerHardware,
    config_manager,
    logger,
    update_progress: Callable[[int, int], None],
):
    """
    Simplified background collection for BackgroundCollectionWorkflow.

    Acquires background images at current position using adaptive exposure
    to reach target intensities. Saves directly to correct folder structure
    for flat field correction.

    Args:
        yaml_file_path: Path to microscope configuration YAML
        output_folder_path: Base folder for backgrounds
        modality: Modality identifier (e.g., "ppm")
        angles_str: String of angles like "(0,90,5,-5)"
        exposures_str: String of initial exposure times like "(1.5,100,50,50)".
                      These are used as starting points for adaptive exposure.
        hardware: Microscope hardware interface
        config_manager: Configuration manager
        logger: Logger instance
        update_progress: Progress callback function

    Returns:
        Dict[float, float]: Dictionary mapping angles to final exposure times (ms)
                           e.g., {90.0: 1.2, 5.0: 45.8, ...}
    """
    logger.info("=== SIMPLE BACKGROUND COLLECTION STARTED ===")

    try:
        # Parse angles and exposures from QuPath
        # Use QuPath's exposures as initial values for adaptive exposure
        angles, exposures = parse_angles_exposures(angles_str, exposures_str)
        logger.info(f"Collecting backgrounds for angles: {angles} using adaptive exposure")
        logger.info(f"Initial exposures from QuPath: {exposures}")

        # Load microscope configuration
        if not Path(yaml_file_path).exists():
            raise FileNotFoundError(f"YAML file {yaml_file_path} does not exist")

        # Load main configuration file
        settings = config_manager.load_config_file(yaml_file_path)

        # Load and merge LOCI resources (same pattern as regular acquisition workflow)
        loci_rsc_file = str(
            Path(__file__).parent / "configurations" / "resources" / "resources_LOCI.yml"
        )
        try:
            loci_resources = config_manager.load_config_file(loci_rsc_file)
            settings.update(loci_resources)
            logger.info("Loaded and merged LOCI resources for background collection")
        except FileNotFoundError:
            logger.warning(
                f"LOCI resources file not found at {loci_rsc_file}, continuing without device mappings"
            )

        hardware.settings = settings

        # Re-initialize microscope-specific methods with updated settings
        # This is critical for PPM rotation to work correctly
        if hasattr(hardware, "_initialize_microscope_methods"):
            hardware._initialize_microscope_methods()
            logger.info("Re-initialized hardware methods with updated settings")

        # Get current position for reference
        current_pos = hardware.get_current_position()
        logger.info(
            f"Acquiring backgrounds at position: X={current_pos.x:.1f}, Y={current_pos.y:.1f}, Z={current_pos.z:.1f}"
        )

        # Create output directory structure
        output_path = Path(output_folder_path)
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving backgrounds to: {output_path}")

        # Initialize progress
        total_images = len(angles)
        update_progress(0, total_images)

        # Track final exposures for each angle
        final_exposures = {}

        # Acquire background for each angle
        for angle_idx, angle in enumerate(angles):
            logger.info(f"Acquiring background {angle_idx + 1}/{total_images} for angle {angle}°")

            # Set rotation angle if supported
            if hasattr(hardware, "set_psg_ticks"):
                hardware.set_psg_ticks(
                    angle  # , is_sequence_start=True
                )  # Each background is independent
                logger.info(f"Set angle to {angle}°")

            # Get target intensity for this modality/angle
            target_intensity = get_target_intensity_for_background(modality, angle)
            logger.info(f"Target intensity: {target_intensity:.1f}")

            # Use exposure from QuPath as initial value for adaptive exposure
            # This is typically based on modality defaults and provides a good starting point
            initial_exposure_ms = exposures[angle_idx] if angle_idx < len(exposures) else 100.0
            logger.info(f"Initial exposure from QuPath: {initial_exposure_ms:.2f}ms")

            # Acquire with adaptive exposure to reach target intensity
            try:
                image, final_exposure = acquire_background_with_target_intensity(
                    hardware=hardware,
                    target_intensity=target_intensity,
                    tolerance=2.5,
                    initial_exposure_ms=initial_exposure_ms,
                    max_iterations=10,
                    logger=logger,
                )
                logger.info(
                    f"Acquired background: shape={image.shape}, median={float(np.median(image)):.1f}, "
                    f"final_exposure={final_exposure:.1f}ms"
                )
                # Store final exposure for this angle
                final_exposures[angle] = final_exposure
            except RuntimeError as e:
                logger.error(f"Failed to acquire background at angle {angle}°: {e}")
                continue

            # Save background image using new format: angle.tif (not in subdirectory)
            background_path = output_path / f"{angle}.tif"
            TifWriterUtils.ome_writer(  # background -single
                filename=str(background_path),
                pixel_size_um=hardware.core.get_pixel_size_um(),
                data=image,
            )

            logger.info(f"Saved background for {angle}° to {background_path}")

            # Update progress
            update_progress(angle_idx + 1, total_images)

        logger.info("=== SIMPLE BACKGROUND COLLECTION COMPLETE ===")
        logger.info(f"Successfully collected {len(angles)} background images")

        # Return final exposures for metadata writing
        return final_exposures

    except Exception as e:
        logger.error(f"Simple background collection failed: {str(e)}", exc_info=True)
        raise


def background_acquisition_workflow(
    yaml_file_path: str,
    output_folder_path: str,
    modality: str,
    angles_str: str,
    exposures_str: Optional[str],
    hardware: PycromanagerHardware,
    config_manager,
    logger,
):
    """
    Acquire background images for flat-field correction.

    IMPORTANT: Position the microscope at a blank area before calling this function.
    The system will acquire images at the current position using adaptive exposure
    to reach target intensities.

    Args:
        yaml_file_path: Path to microscope configuration YAML
        output_folder_path: Base folder for backgrounds (will create modality subfolder)
        modality: Modality identifier (e.g., "PPM_20x")
        angles_str: String of angles like "(0,90,5,-5)"
        exposures_str: String of initial exposure times like "(1.5,100,50,50)".
                      These are used as starting points for adaptive exposure.
        hardware: Microscope hardware interface
        config_manager: Configuration manager
        logger: Logger instance

    Returns:
        Tuple[str, Dict[float, float]]: (output_path, final_exposures)
            output_path: Path where backgrounds were saved
            final_exposures: Dictionary mapping angles to final exposure times (ms)
    """
    logger.info("=== BACKGROUND ACQUISITION WORKFLOW STARTED ===")
    logger.warning("Ensure microscope is positioned at a clean, blank area!")

    # Get and log current position for reference
    current_pos = hardware.get_current_position()
    logger.info(
        f"Acquiring backgrounds at position: X={current_pos.x:.1f}, "
        f"Y={current_pos.y:.1f}, Z={current_pos.z:.1f}"
    )

    try:
        # Parse angles and exposures from QuPath
        # Use QuPath's exposures as initial values for adaptive exposure
        angles, exposures = parse_angles_exposures(angles_str, exposures_str)
        logger.info(f"Initial exposures from QuPath: {exposures}")

        # Load the microscope configuration
        if not Path(yaml_file_path).exists():
            raise FileNotFoundError(f"YAML file {yaml_file_path} does not exist")

        settings = config_manager.load_config_file(yaml_file_path)
        hardware.settings = settings

        # Re-initialize microscope-specific methods with updated settings
        # This is critical for PPM rotation to work correctly
        if hasattr(hardware, "_initialize_microscope_methods"):
            hardware._initialize_microscope_methods()
            logger.info("Re-initialized hardware methods with updated settings")

        # Create output directory structure with modality
        output_path = Path(output_folder_path) / "backgrounds" / modality
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving backgrounds to: {output_path}")

        # Track final exposures for each angle
        final_exposures = {}

        # Acquire background for each angle
        for angle_idx, angle in enumerate(angles):
            # Create angle subdirectory
            angle_dir = output_path / str(angle)
            angle_dir.mkdir(exist_ok=True)

            # Set rotation angle if PPM
            if hasattr(hardware, "set_psg_ticks"):
                hardware.set_psg_ticks(
                    angle  # , is_sequence_start=True
                )  # Each background is independent
                logger.info(f"Set angle to {angle}°")

            # Get target intensity for this modality/angle
            target_intensity = get_target_intensity_for_background(modality, angle)
            logger.info(f"Target intensity: {target_intensity:.1f}")

            # Use exposure from QuPath as initial value for adaptive exposure
            # This is typically based on modality defaults and provides a good starting point
            initial_exposure_ms = exposures[angle_idx] if angle_idx < len(exposures) else 100.0
            logger.info(f"Initial exposure from QuPath: {initial_exposure_ms:.2f}ms")

            # Acquire with adaptive exposure to reach target intensity
            try:
                image, final_exposure = acquire_background_with_target_intensity(
                    hardware=hardware,
                    target_intensity=target_intensity,
                    tolerance=2.5,
                    initial_exposure_ms=initial_exposure_ms,
                    max_iterations=10,
                    logger=logger,
                )
                logger.info(
                    f"Acquired background: median={float(np.median(image)):.1f}, "
                    f"final_exposure={final_exposure:.1f}ms"
                )
                # Store final exposure for this angle
                final_exposures[angle] = final_exposure
            except RuntimeError as e:
                logger.error(f"Failed to acquire background at angle {angle}°: {e}")
                continue

            # Save background image
            background_path = angle_dir / "background.tif"
            TifWriterUtils.ome_writer(  # background 2 with bkg-workflow
                filename=str(background_path),
                pixel_size_um=hardware.core.get_pixel_size_um(),
                data=image,
            )

            logger.info(f"Saved background for {angle}° to {background_path}")

        logger.info("=== BACKGROUND ACQUISITION COMPLETE ===")
        return str(output_path), final_exposures

    except Exception as e:
        logger.error(f"Background acquisition failed: {str(e)}", exc_info=True)
        raise


def polarizer_calibration_workflow(
    yaml_file_path: str,
    output_folder_path: str,
    start_angle: float,
    end_angle: float,
    step_size: float,
    exposure_ms: float,
    hardware: PycromanagerHardware,
    config_manager,
    logger,
) -> str:
    """
    Calibrate PPM polarizer rotation stage to find crossed polarizer positions.

    IMPORTANT: Position microscope at uniform, bright background before calling.
    This workflow sweeps the rotation stage through angles, measures intensity,
    and determines optimal crossed polarizer positions for config_PPM.yml.

    Args:
        yaml_file_path: Path to microscope configuration YAML
        output_folder_path: Base folder for backgrounds (will write report at top level)
        start_angle: Starting angle for sweep (degrees)
        end_angle: Ending angle for sweep (degrees)
        step_size: Step size for sweep (degrees)
        exposure_ms: Exposure time (milliseconds)
        hardware: Microscope hardware interface
        config_manager: Configuration manager
        logger: Logger instance

    Returns:
        str: Path to the calibration report text file
    """
    logger.info("=== POLARIZER CALIBRATION WORKFLOW STARTED ===")
    logger.warning("Ensure microscope is positioned at a uniform, bright background!")

    # Get and log current position for reference
    current_pos = hardware.get_current_position()
    logger.info(
        f"Running calibration at position: X={current_pos.x:.1f}, "
        f"Y={current_pos.y:.1f}, Z={current_pos.z:.1f}"
    )

    try:
        # Load the microscope configuration
        if not Path(yaml_file_path).exists():
            raise FileNotFoundError(f"YAML file {yaml_file_path} does not exist")

        settings = config_manager.load_config_file(yaml_file_path)

        # Load and merge LOCI resources (required for rotation stage device lookup)
        loci_rsc_file = str(
            Path(yaml_file_path).parent / "resources" / "resources_LOCI.yml"
        )
        if Path(loci_rsc_file).exists():
            loci_resources = config_manager.load_config_file(loci_rsc_file)
            settings.update(loci_resources)
            logger.info("Loaded and merged LOCI resources")
        else:
            logger.warning(f"LOCI resources file not found: {loci_rsc_file}")

        hardware.settings = settings

        # Re-initialize microscope-specific methods
        if hasattr(hardware, "_initialize_microscope_methods"):
            hardware._initialize_microscope_methods()
            logger.info("Re-initialized hardware methods with updated settings")

        # Verify PPM is available
        if not hasattr(hardware, "set_psg_ticks"):
            raise RuntimeError(
                "PPM rotation stage methods not available. "
                "Check ppm_optics setting in configuration."
            )

        # Import the calibration utility
        from smart_wsi_scanner.qp_utils import PolarizerCalibrationUtils

        # Run two-stage calibration to determine exact hardware offset
        logger.info(
            f"Starting two-stage hardware calibration: "
            f"Coarse: 0-360 deg in {step_size} deg steps, "
            f"Fine: +/-{step_size} deg in 0.1 deg steps"
        )
        logger.info(f"Exposure: {exposure_ms} ms")

        result = PolarizerCalibrationUtils.calibrate_hardware_offset_two_stage(
            hardware=hardware,
            coarse_range_deg=360.0,  # Full rotation
            coarse_step_deg=step_size,  # Use user-specified step size for coarse
            fine_range_deg=step_size,  # Fine sweep range = coarse step size
            fine_step_deg=0.1,  # Fine step for precise positioning
            exposure_ms=exposure_ms,
            channel=1,  # Green channel
            logger_instance=logger,
        )

        # Write calibration report
        import datetime

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"polarizer_calibration_{timestamp}.txt"
        report_path = Path(output_folder_path) / report_filename

        # Ensure output directory exists
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("HARDWARE OFFSET CALIBRATION REPORT (TWO-STAGE)\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Calibration Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Configuration File: {yaml_file_path}\n")
            f.write(
                f"Microscope Position: X={current_pos.x:.1f}, Y={current_pos.y:.1f}, Z={current_pos.z:.1f}\n"
            )
            f.write(f"Rotation Device: {result['rotation_device']}\n\n")

            f.write("CALIBRATION METHOD:\n")
            f.write("  Stage 1: Coarse sweep to locate approximate minima\n")
            f.write("  Stage 2: Fine sweep around each minimum for exact position\n\n")

            f.write("CALIBRATION PARAMETERS:\n")
            f.write(f"  Coarse Range: 360.0 deg\n")
            f.write(f"  Coarse Step Size: {step_size} deg\n")
            f.write(f"  Fine Range: +/-{step_size} deg around each minimum\n")
            f.write(f"  Fine Step Size: 0.1 deg\n")
            f.write(f"  Exposure: {exposure_ms} ms\n")
            f.write(f"  Channel: Green (1)\n\n")

            f.write("INTENSITY STATISTICS (COARSE SWEEP):\n")
            coarse_intensities = result["coarse_intensities"]
            f.write(f"  Minimum Intensity: {coarse_intensities.min():.1f}\n")
            f.write(f"  Maximum Intensity: {coarse_intensities.max():.1f}\n")
            f.write(
                f"  Dynamic Range: {coarse_intensities.max() / coarse_intensities.min():.2f}x\n\n"
            )

            f.write("=" * 80 + "\n")
            f.write("CALIBRATION RESULTS\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Found {len(result['exact_minima'])} crossed polarizer positions:\n\n")

            for i, (hw_pos, opt_angle) in enumerate(
                zip(result["exact_minima"], result["optical_angles"])
            ):
                f.write(f"  Minimum {i+1}:\n")
                f.write(f"    Hardware Position: {hw_pos:.1f} encoder counts\n")
                f.write(
                    f"    Optical Angle: {opt_angle:.2f} deg (relative to recommended offset)\n"
                )

                # Find corresponding fine sweep result
                for fine_result in result["fine_results"]:
                    if abs(fine_result["exact_position"] - hw_pos) < 0.1:
                        f.write(f"    Intensity: {fine_result['exact_intensity']:.1f}\n")
                        break
                f.write("\n")

            # Calculate separation between minima
            if len(result["exact_minima"]) >= 2:
                separation = abs(result["exact_minima"][1] - result["exact_minima"][0])
                separation_deg = separation / result["hw_per_deg"]
                f.write(
                    f"Separation between minima: {separation:.1f} counts ({separation_deg:.1f} deg)\n"
                )
                f.write(f"Expected: {180.0 * result['hw_per_deg']:.1f} counts (180.0 deg)\n\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("CONFIG_PPM.YML UPDATE RECOMMENDATIONS\n")
            f.write("=" * 80 + "\n\n")

            f.write("CRITICAL: Update ppm_pizstage_offset to the recommended value below.\n")
            f.write("This sets the hardware reference position for optical angle 0 deg.\n\n")

            f.write(f"ppm_pizstage_offset: {result['recommended_offset']:.1f}\n\n")

            f.write("After updating the offset, you can use the following optical angles:\n\n")
            f.write("rotation_angles:\n")
            f.write("  - name: 'crossed'\n")
            f.write(
                "    tick: 0   # Reference position (hardware: {:.1f})\n".format(
                    result["recommended_offset"]
                )
            )

            # If there's a second minimum, suggest it as the other crossed position
            if len(result["exact_minima"]) >= 2:
                other_angle = result["optical_angles"][1]
                other_hw = result["exact_minima"][1]
                f.write(
                    "    # OR tick: {:.0f}   # Alternate crossed (hardware: {:.1f})\n".format(
                        other_angle, other_hw
                    )
                )

            f.write("  - name: 'uncrossed'\n")
            f.write("    tick: 90  # 90 deg from crossed (perpendicular)\n\n")

            f.write("Note: The 'tick' values use simplified angle convention.\n")
            f.write("Hardware automatically applies offset: hw_pos = (tick * 1000) + offset\n\n")

            # ===== METADATA AND PARAMETERS - REFERENCE INFORMATION =====
            f.write("=" * 80 + "\n")
            f.write("CALIBRATION METADATA\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Calibration Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Configuration File: {yaml_file_path}\n")
            f.write(f"Microscope Position: X={current_pos.x:.1f}, Y={current_pos.y:.1f}, Z={current_pos.z:.1f}\n")
            f.write(f"Rotation Device: {result['rotation_device']}\n\n")

            f.write("CALIBRATION METHOD:\n")
            f.write("  Stage 1: Coarse sweep to locate approximate minima\n")
            f.write("  Stage 2: Fine sweep around each minimum for exact position\n\n")

            f.write("CALIBRATION PARAMETERS:\n")
            f.write(f"  Coarse Range: 360.0 deg\n")
            f.write(f"  Coarse Step Size: {step_size} deg\n")
            f.write(f"  Fine Range: +/-{step_size} deg around each minimum\n")
            f.write(f"  Fine Step Size: 0.1 deg\n")
            f.write(f"  Exposure: {exposure_ms} ms\n")
            f.write(f"  Channel: Green (1)\n\n")

            f.write("INTENSITY STATISTICS (COARSE SWEEP):\n")
            coarse_intensities = result['coarse_intensities']
            f.write(f"  Minimum Intensity: {coarse_intensities.min():.1f}\n")
            f.write(f"  Maximum Intensity: {coarse_intensities.max():.1f}\n")
            f.write(f"  Dynamic Range: {coarse_intensities.max() / coarse_intensities.min():.2f}x\n\n")

            f.write("=" * 80 + "\n")
            f.write("RAW DATA - COARSE SWEEP\n")
            f.write("=" * 80 + "\n\n")

            f.write("Hardware Position (counts), Intensity\n")
            for hw_pos, intensity in zip(
                result["coarse_hardware_positions"], result["coarse_intensities"]
            ):
                f.write(f"{hw_pos:.1f}, {intensity:.2f}\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("RAW DATA - FINE SWEEPS\n")
            f.write("=" * 80 + "\n\n")

            for i, fine_result in enumerate(result["fine_results"]):
                f.write(
                    f"\nFine Sweep {i+1} (centered on {fine_result['approximate_position']:.1f}):\n"
                )
                f.write("Hardware Position (counts), Intensity\n")
                for hw_pos, intensity in zip(
                    fine_result["fine_hw_positions"], fine_result["fine_intensities"]
                ):
                    f.write(f"{hw_pos:.1f}, {intensity:.2f}\n")

        logger.info(f"Calibration report saved to: {report_path}")
        logger.info("=== POLARIZER CALIBRATION WORKFLOW COMPLETE ===")

        return str(report_path)

    except Exception as e:
        logger.error(f"Polarizer calibration failed: {str(e)}", exc_info=True)
        raise
