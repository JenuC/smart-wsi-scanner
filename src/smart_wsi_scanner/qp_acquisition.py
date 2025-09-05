"""Acquisition workflow and microscope-side operations for QuPath server.

This module contains the acquisition logic and helpers that interact with the
microscope hardware, separated from the socket server/transport logic.
"""

from __future__ import annotations

from typing import Callable, List, Tuple, Optional, Dict, Any
import pathlib
import shutil
import logging

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
                exposures = [int(x.strip()) for x in exposures_str.split(",")]
            elif exposures_str:
                exposures = [int(x) for x in exposures_str.split()]

    # Default exposures if not provided
    if not exposures and angles:
        for angle in angles:
            if angle == 90.0:
                exposures.append(10)
            elif angle == 0.0:
                exposures.append(800)
            else:
                exposures.append(500)
                
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
            else:
                i += 1

        # Parse angles and exposures if present
        angles, exposures = parse_angles_exposures(
            params.get("angles_str", "()"), params.get("exposures_str", None)
        )
        params["angles"] = angles
        params["exposures"] = exposures

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
    wb_settings = settings.get('white_balance', {})
    ppm_wb = wb_settings.get('ppm', {})
    
    # Map standard angle names to numeric values
    angle_mapping = {
        'crossed': 0.0,
        'uncrossed': 90.0,
        'positive': 5.0,
        'negative': -5.0
    }
    
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
            0.0: [1.0, 1.0, 1.0],    # crossed
            90.0: [1.2, 1.0, 1.1],   # uncrossed
            5.0: [1.0, 1.0, 1.0],    # positive
            -5.0: [1.0, 1.0, 1.0]    # negative
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
        if not pathlib.Path(params["yaml_file_path"]).exists():
            raise FileNotFoundError(f"YAML file {params['yaml_file_path']} does not exist")
        
        # Load configuration using the config manager
        ppm_settings = config_manager.load_config_file(params["yaml_file_path"])
        loci_rsc_file = str(
            pathlib.Path(__file__).parent / "configurations" / "resources" / "resources_LOCI.yml"
        )
        loci_resources = config_manager.load_config_file(loci_rsc_file)
        ppm_settings.update(loci_resources)
        hardware.settings = ppm_settings

        # Extract modality from scan type
        modality = BackgroundCorrectionUtils.get_modality_from_scan_type(params["scan_type"])
        logger.info(f"Using modality: {modality}")

        # Load background images if correction is enabled
        background_images = {}
        background_correction_enabled = False
        angle_wb_corrections = {}

        # Check for background correction settings
        modalities = ppm_settings.get('modalities', {})
        ppm_config = modalities.get('ppm', {})
        bc_settings = ppm_config.get('background_correction', {})

        if bc_settings.get('enabled') and bc_settings.get('base_folder'):
            # Load background images for this modality
            background_dir = pathlib.Path(bc_settings['base_folder'])

            if background_dir.exists():
                is_valid, missing = BackgroundCorrectionUtils.validate_background_images(
                    background_dir, modality, params["angles"], logger
                )

                if is_valid:
                    background_images, background_scaling_factors, white_balance_coeffs = (
                        BackgroundCorrectionUtils.load_background_images(
                            background_dir, modality, params["angles"], logger
                        )
                    )

                    if background_images and background_scaling_factors:
                        background_correction_enabled = True
                        logger.info(
                            f"Background correction enabled with {len(background_images)} images"
                        )
                        logger.info(f"Method: {bc_settings.get('method', 'divide')}")
                        logger.info(f"Scaling factors: {background_scaling_factors}")
                        logger.info(f"White balance coefficients: {white_balance_coeffs}")
                else:
                    logger.error(
                        f"Cannot proceed with background correction - missing images for angles: {missing}"
                    )
                    logger.warning("Continuing without background correction")
            else:
                logger.warning(f"Background directory not found: {background_dir}")
                
        # Set up output paths
        project_path = pathlib.Path(params["projects_folder_path"]) / params["sample_label"]
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
        update_progress(0, total_images)
        logger.info(
            f"Starting acquisition of {total_images} total images "
            f"({len(positions)} positions × {len(params['angles'])} angles)"
        )

        image_count = 0

        # Get white balance values for angles from settings
        angles_wb = get_angles_wb_from_settings(ppm_settings)

        # Find autofocus positions
        fov = hardware.get_fov()
        
        # Get autofocus settings from config
        acq_profiles = ppm_settings.get('acq_profiles_new', {})
        defaults = acq_profiles.get('defaults', [])
        
        # Find autofocus parameters for current objective
        af_n_tiles = 5  # default
        af_search_range = 50  # default
        af_n_steps = 11  # default
        
        # Try to get current objective from hardware
        microscope = ppm_settings.get('microscope', {})
        current_objective = microscope.get('objective_in_use', '')
        
        # Look for autofocus settings in defaults
        for default in defaults:
            if default.get('objective') == current_objective:
                af_settings = default.get('settings', {}).get('autofocus', {})
                af_n_tiles = af_settings.get('n_tiles', af_n_tiles)
                af_search_range = af_settings.get('search_range_um', af_search_range)
                af_n_steps = af_settings.get('n_steps', af_n_steps)
                break
        
        try:
            xy_positions = [(pos.x, pos.y) for pos, filename in positions]
            af_positions, af_min_distance = AutofocusUtils.get_autofocus_positions(
                fov, xy_positions, n_tiles=af_n_tiles
            )
        except Exception as e:
            logger.warning("Falling back to older tileconfig reader: %s", e)
            xy_positions = TileConfigUtils.read_TileConfiguration_coordinates(tile_config_path)
            af_positions, af_min_distance = AutofocusUtils.get_autofocus_positions(
                fov, xy_positions, n_tiles=af_n_tiles
            )
        
        logger.info(f"Autofocus positions: {af_positions}")

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

            # Perform autofocus if needed
            if pos_idx in af_positions:
                logger.info(f"Performing Autofocus at X={pos.x}, Y={pos.y}, Z={pos.z}")
                new_z = hardware.autofocus(
                    move_stage_to_estimate=True,
                    n_steps=af_n_steps,
                    search_range=af_search_range,
                    interp_strength=100,
                    score_metric=AutofocusUtils.autofocus_profile_laplacian_variance,
                )
                logger.info(f"  Autofocus :: New Z {new_z}")
                
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
                    hardware.set_psg_ticks(angle)
                    logger.info(f"  Angle set to {hardware.get_psg_ticks():.1f}")
                    
                    # Set exposure time if specified
                    if angle_idx < len(params["exposures"]):
                        exposure_ms = params["exposures"][angle_idx]
                        hardware.core.set_exposure(exposure_ms)
                    logger.info(f"  Exposure set to {hardware.core.get_exposure()}")
                    
                    # Acquire image
                    image, metadata = hardware.snap_image(debayering=False)
                    
                    if image is None:
                        logger.error(f"Failed to acquire image at angle {angle}")
                        continue
                        
                    logger.info(f"  Image shape: {image.shape}, mean: {image.mean((0,1))}")

                    # Apply background correction if available
                    if background_correction_enabled and angle in background_images:
                        bc_method = bc_settings.get('method', 'divide')

                        image = BackgroundCorrectionUtils.apply_flat_field_correction(
                            image,
                            background_images[angle],
                            background_scaling_factors[angle],
                            method=bc_method,
                        )
                        logger.info(
                            f"  Applied {bc_method} flat-field correction with factor "
                            f"{background_scaling_factors[angle]:.3f}"
                        )
                        logger.info(f"  Post-correction RGB means: {image.mean(axis=(0,1))}")
                        logger.info(f"  Post-correction RGB max: {image.max(axis=(0,1))}")

                        # Calculate WB from first tile only
                        if pos_idx == 0:
                            # Calculate what correction is needed
                            bg_color, confidence = (
                                BackgroundCorrectionUtils.calculate_background_color_from_mode(
                                    image
                                )
                            )

                            if confidence > 0.2:  # Reasonable amount of background found
                                logger.info(
                                    f"  Detected background color: RGB={bg_color}, "
                                    f"confidence={confidence:.1%}"
                                )

                                # Calculate correction needed
                                target_white = bg_color.mean()
                                wb_for_division = bg_color / target_white
                                angle_wb_corrections[angle] = wb_for_division.tolist()

                                logger.info(f"  WB coefficients from background: {wb_for_division}")
                            else:
                                # Low confidence - try alternative approach
                                logger.warning(
                                    "  Low background confidence, using percentile approach"
                                )

                                # Use bright percentile as fallback
                                percentile_95 = np.percentile(image, 95, axis=(0, 1))
                                target = percentile_95.mean()
                                wb_for_division = percentile_95 / target
                                angle_wb_corrections[angle] = wb_for_division.tolist()

                        # Apply the stored correction
                        if angle in angle_wb_corrections:
                            wb_profile = angle_wb_corrections[angle]
                            gain = calculate_luminance_gain(*wb_profile)
                            image = hardware.white_balance(
                                image, white_balance_profile=wb_profile, gain=gain
                            )
                            logger.info(f"  Applied first-tile WB: {wb_profile}")
                    else:
                        # No background correction - use settings-based WB
                        if angle in angles_wb:
                            wb_profile = angles_wb[angle]
                        else:
                            # Default if angle not found
                            wb_profile = [1.0, 1.0, 1.0]
                            
                        gain = calculate_luminance_gain(*wb_profile)
                        image = hardware.white_balance(
                            image, white_balance_profile=wb_profile, gain=gain
                        )
                        logger.info(f"  Applied white balance: {wb_profile}")
                        
                    # Save image
                    image_path = output_path / str(angle) / filename
                    if image_path.parent.exists():
                        TifWriterUtils.ome_writer(
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
                    TifWriterUtils.create_birefringence_tile(
                        pos_image=angle_images[pos_angle],
                        neg_image=angle_images[neg_angle],
                        output_dir=biref_dir,
                        filename=filename,
                        pixel_size_um=hardware.core.get_pixel_size_um(),
                        tile_config_source=tile_config_source,
                        logger=logger,
                    )

            else:
                # Single image acquisition: no angles specified
                image, metadata = hardware.snap_image()
                image_path = output_path / filename

                if image_path.parent.exists():
                    TifWriterUtils.ome_writer(
                        filename=str(image_path),
                        pixel_size_um=hardware.core.get_pixel_size_um(),
                        data=image,
                    )
                    image_count += 1
                    update_progress(image_count, total_images)

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

    except Exception as e:
        logger.error("=== ACQUISITION FAILED ===")
        logger.error(f"Error: {str(e)}", exc_info=True)
        set_state("FAILED")


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
    The system will acquire images at the current position.

    Args:
        yaml_file_path: Path to microscope configuration YAML
        output_folder_path: Base folder for backgrounds (will create modality subfolder)
        modality: Modality identifier (e.g., "PPM_20x")
        angles_str: String of angles like "(0,90,5,-5)"
        exposures_str: Optional string of exposures like "(800,10,500,500)".
                If None, defaults will be used based on angles.
        hardware: Microscope hardware interface
        config_manager: Configuration manager
        logger: Logger instance
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
        # Parse angles and exposures
        angles, exposures = parse_angles_exposures(angles_str, exposures_str)

        # Load the microscope configuration
        if not pathlib.Path(yaml_file_path).exists():
            raise FileNotFoundError(f"YAML file {yaml_file_path} does not exist")

        settings = config_manager.load_config_file(yaml_file_path)
        hardware.settings = settings

        # Create output directory structure with modality
        output_path = pathlib.Path(output_folder_path) / "backgrounds" / modality
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving backgrounds to: {output_path}")

        # Acquire background for each angle
        for angle_idx, angle in enumerate(angles):
            # Create angle subdirectory
            angle_dir = output_path / str(angle)
            angle_dir.mkdir(exist_ok=True)

            # Set rotation angle if PPM
            if hasattr(hardware, "set_psg_ticks"):
                hardware.set_psg_ticks(angle)
                logger.info(f"Set angle to {angle}°")

            # Set exposure time
            if angle_idx < len(exposures):
                exposure_ms = exposures[angle_idx]
                hardware.core.set_exposure(exposure_ms)
                logger.info(f"Set exposure to {exposure_ms}ms")

            # Acquire image with debayering
            image, metadata = hardware.snap_image(debayering=True)

            # Save background image
            background_path = angle_dir / "background.tif"
            TifWriterUtils.ome_writer(
                filename=str(background_path),
                pixel_size_um=hardware.core.get_pixel_size_um(),
                data=image,
            )

            logger.info(f"Saved background for {angle}° to {background_path}")

        logger.info("=== BACKGROUND ACQUISITION COMPLETE ===")
        return str(output_path)

    except Exception as e:
        logger.error(f"Background acquisition failed: {str(e)}", exc_info=True)
        raise
