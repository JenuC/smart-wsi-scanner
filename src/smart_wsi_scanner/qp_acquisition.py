"""Acquisition workflow and microscope-side operations for QuPath server.

This module contains the acquisition logic and helpers that interact with the
microscope hardware, separated from the socket server/transport logic.
"""

from __future__ import annotations

from typing import Callable, List, Tuple, Optional
import pathlib
import shutil

from .config import sp_position
from .hardware_pycromanager import PycromanagerHardware
from .qp_utils import TileConfigUtils, TifWriterUtils, AutofocusUtils
import shlex
import skimage.filters


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
    """Execute the main image acquisition workflow with progress and cancellation.
    The server provides callbacks for status/progress and cancellation checks.
    # Parse the acquisition parameters
        ## load the yaml file
        ## Set up output paths
    # Read tile positions
        ## Create subdirectories for each angle
        ## Calculate total images and update progress
        ## load white balance values for angles from the yaml
        ## find autofocus positions for range from the yaml
    # Main acquisition loop
        ## TODO ideally the pos-list needs to be parsed/ estimated before acq
    ## Move to position- set Angle - set Expsoure - Debayer  - WhiteBalance - Save

    """

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

        # load the yaml file
        if not params["yaml_file_path"]:
            raise ValueError("YAML file path is required")
        if not pathlib.Path(params["yaml_file_path"]).exists():
            raise FileNotFoundError(f"YAML file {params['yaml_file_path']} does not exist")
        ppm_settings = config_manager.load_config(params["yaml_file_path"])  # type: ignore[attr-defined]
        hardware.settings = ppm_settings
        # Set up output paths
        project_path = pathlib.Path(params["projects_folder_path"]) / params["sample_label"]
        output_path = project_path / params["scan_type"] / params["region_name"]
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created output directory: {output_path}")

        # Read tile positions
        tile_config_path = output_path / "TileConfiguration.txt"
        positions = TileConfigUtils.read_tile_config(tile_config_path, hardware.core)  # type:ignore

        if not positions:
            logger.error(f"No positions found in {tile_config_path}")
            set_state("FAILED")
            return

        # Create subdirectories for each angle
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
            f"Starting acquisition of {total_images} total images ({len(positions)} positions × {len(params['angles'])} angles)"
        )

        image_count = 0

        ## white balance values for angles

        [float(x) for x in ppm_settings.white_balance.ppm.crossed[0].split(" ")]
        angles_wb_ = {
            0: hardware.settings.white_balance.ppm.crossed,
            90: hardware.settings.white_balance.ppm.uncrossed,
            5.0: hardware.settings.white_balance.ppm.positive,
            -5.0: hardware.settings.white_balance.ppm.negative,
        }
        angles_wb = {
            angle: [float(x) for x in wb_list[0].split()] for angle, wb_list in angles_wb_.items()
        }

        # find autofocus positions:
        fov = hardware.get_fov()
        try:
            xy_positions = [(pos.x, pos.y) for pos, filename in positions]
            af_positions, af_min_distance = AutofocusUtils.get_autofocus_positions(
                fov, xy_positions, n_tiles=ppm_settings.autofocus.n_tiles  # type:ignore
            )
        except Exception as e:
            print("! falling back to older tileconfig-reader")
            xy_positions = TileConfigUtils.read_TileConfiguration_coordinates(tile_config_path)
            af_positions, af_min_distance = AutofocusUtils.get_autofocus_positions(
                fov, xy_positions, n_tiles=ppm_settings.autofocus.n_tiles  # type:ignore
            )
        print(af_positions)

        # Main acquisition loop
        for pos_idx, (pos, filename) in enumerate(positions):
            # Check for cancellation
            if is_cancelled():
                logger.warning(f"Acquisition cancelled by client {client_addr}")
                set_state("CANCELLED")
                return

            logger.info(f"Position {pos_idx + 1}/{len(positions)}: {filename}")

            # tile config is not handling Z, so we need the z as the last set autofocus z value
            # ensure Z is not loaded from the pos-list (that uses a single z value when tile config was passed)
            # TODO ideally the pos-list needs to be parsed and autofocus positions and z needs to estimated separately
            pos.z = hardware.get_current_position().z
            # Move to position
            logger.info(f"Moving to position: X={pos.x}, Y={pos.y}, Z={pos.z}")
            hardware.move_to_position(pos)

            if pos_idx in af_positions:
                logger.info(f"Performing Autofocus at X={pos.x}, Y={pos.y}, Z={pos.z}")
                new_z = hardware.autofocus(
                    move_stage_to_estimate=True,
                    n_steps=ppm_settings.autofocus.n_steps,
                    search_range=ppm_settings.autofocus.search_range,
                    interp_strength=100,
                    score_metric=AutofocusUtils.autofocus_profile_laplacian_variance,
                )
                logger.info(f"New Z {new_z}")
            if params["angles"]:
                # Multi-angle acquisition
                for angle_idx, angle in enumerate(params["angles"]):

                    # Check for cancellation
                    if is_cancelled():
                        logger.warning(f"Acquisition cancelled by client {client_addr}")
                        set_state("CANCELLED")
                        return

                    # Set rotation angle
                    hardware.set_psg_ticks(angle)  # type:Ignore
                    # set_angle(hardware, brushless_device, angle)

                    # Set exposure time if specified
                    if angle_idx < len(params["exposures"]):
                        exposure_ms = params["exposures"][angle_idx]
                        hardware.core.set_exposure(exposure_ms)  # type:ignore

                    ## FORCE debayering for mm2:
                    # Acquire image
                    image, metadata = hardware.snap_image(debayering=True)  # type: ignore[attr-defined]
                    logger.info(f"Debayer on ndim{image.ndim} mean {image.mean((0,1))}")
                    # white balance

                    image = hardware.white_balance(image, white_balance_profile=angles_wb[angle])
                    logger.info(f"whitebalance applied {angles_wb[angle]}")
                    # Save image
                    image_path = output_path / str(angle) / filename
                    if image_path.parent.exists():
                        TifWriterUtils.ome_writer(
                            filename=str(image_path),
                            pixel_size_um=hardware.core.get_pixel_size_um(),  # type: ignore[attr-defined]
                            data=image,
                        )
                        image_count += 1
                        update_progress(image_count, total_images)
                    else:
                        logger.error(f"Failed to save {image_path} - parent directory missing")
            else:
                # Single image acquisition: no angles specified
                image, metadata = hardware.snap_image()  # type: ignore[attr-defined]
                image_path = output_path / filename

                if image_path.parent.exists():
                    TifWriterUtils.ome_writer(
                        filename=str(image_path),
                        pixel_size_um=hardware.core.get_pixel_size_um(),  # type: ignore[attr-defined]
                        data=image,  # type:ignore
                    )
                    image_count += 1
                    update_progress(image_count, total_images)

        # Save device properties
        current_props = hardware.get_device_properties()  # type:ignore
        props_path = output_path / "MMproperties.txt"
        with open(props_path, "w") as fid:
            from pprint import pprint as dict_printer

            dict_printer(current_props, stream=fid)

        set_state("COMPLETED")
        logger.info(f"=== ACQUISITION COMPLETED SUCCESSFULLY ===")
        logger.info(f"Total images saved: {image_count}/{total_images}")
        logger.info(f"Output directory: {output_path}")

    except Exception as e:  # noqa: BLE001
        logger.error(f"=== ACQUISITION FAILED ===")
        logger.error(f"Error: {str(e)}", exc_info=True)
        set_state("FAILED")
