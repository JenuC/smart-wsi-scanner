"""Acquisition workflow and microscope-side operations for QuPath server.

This module contains the acquisition logic and helpers that interact with the
microscope hardware, separated from the socket server/transport logic.
"""

from __future__ import annotations

from typing import Callable, List, Tuple, Optional
import pathlib
import re
import shutil
import time

from .config import sp_position
from .hardware import MicroscopeHardware
from .smartpath import smartpath, smartpath_qpscope


def ppm_to_thor(angle: float) -> float:
    """Convert PPM angle (in degrees) to Thor rotation stage position."""
    return -2 * angle + 276


def thor_to_ppm(kinesis_pos: float) -> float:
    """Convert Thor rotation stage position to PPM angle (in degrees)."""
    return (276 - kinesis_pos) / 2


def set_angle(core, brushless_device: str, theta: float) -> None:
    """Set the rotation stage to a specific angle and wait for completion."""
    theta_thor = ppm_to_thor(theta)
    core.set_position(brushless_device, theta_thor)
    core.wait_for_device(brushless_device)


def read_tile_config(tile_config_path: pathlib.Path, core) -> List[Tuple[sp_position, str]]:
    """Read tile positions from a QuPath-generated TileConfiguration.txt file."""
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


def parse_angles_exposures(
    angles_str, exposures_str=None
) -> Tuple[List[float], List[int]]:
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
        import shlex

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
            elif parts[i] == "--laser-power" and i + 1 < len(parts):
                params["laser_power"] = float(parts[i + 1])
                i += 2
            elif parts[i] == "--laser-wavelength" and i + 1 < len(parts):
                params["laser_wavelength"] = int(parts[i + 1])
                i += 2
            elif parts[i] == "--dwell-time" and i + 1 < len(parts):
                params["dwell_time"] = float(parts[i + 1])
                i += 2
            elif parts[i] == "--averaging" and i + 1 < len(parts):
                params["averaging"] = int(parts[i + 1])
                i += 2
            elif parts[i] == "--z-stack":
                params["z_stack_enabled"] = True
                i += 1
            elif parts[i] == "--z-start" and i + 1 < len(parts):
                params["z_start"] = float(parts[i + 1])
                i += 2
            elif parts[i] == "--z-end" and i + 1 < len(parts):
                params["z_end"] = float(parts[i + 1])
                i += 2
            elif parts[i] == "--z-step" and i + 1 < len(parts):
                params["z_step"] = float(parts[i + 1])
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


def acquisition_workflow(
    message: str,
    client_addr,
    *,
    core,
    hardware: MicroscopeHardware,
    config_manager,
    logger,
    update_progress: Callable[[int, int], None],
    set_state: Callable[[str], None],
    is_cancelled: Callable[[], bool],
    brushless_device: Optional[str] = None,
):
    """Execute the main image acquisition workflow with progress and cancellation.

    The server provides callbacks for status/progress and cancellation checks.
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
        ppm_settings = config_manager.load_yaml(params["yaml_file_path"])  # type: ignore[attr-defined]

        # Set up output paths
        project_path = pathlib.Path(params["projects_folder_path"]) / params["sample_label"]
        output_path = project_path / params["scan_type"] / params["region_name"]
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created output directory: {output_path}")

        # Read tile positions
        tile_config_path = output_path / "TileConfiguration.txt"
        positions = read_tile_config(tile_config_path, core)

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

        # Initialize smartpath
        sp = smartpath(core)

        # Calculate total images and update progress
        total_images = (
            len(positions) * len(params["angles"]) if params["angles"] else len(positions)
        )
        update_progress(0, total_images)
        logger.info(
            f"Starting acquisition of {total_images} total images ({len(positions)} positions × {len(params['angles'])} angles)"
        )

        image_count = 0

        # Main acquisition loop
        for pos_idx, (pos, filename) in enumerate(positions):
            # Check for cancellation
            if is_cancelled():
                logger.warning(f"Acquisition cancelled by client {client_addr}")
                set_state("CANCELLED")
                return

            logger.info(f"Position {pos_idx + 1}/{len(positions)}: {filename}")

            # Move to position
            logger.debug(f"Moving to position: X={pos.x}, Y={pos.y}, Z={pos.z}")
            hardware.move_to_position(pos)

            if params["angles"]:
                # Multi-angle acquisition
                for angle_idx, angle in enumerate(params["angles"]):
                    # Check for cancellation
                    if is_cancelled():
                        logger.warning(f"Acquisition cancelled by client {client_addr}")
                        set_state("CANCELLED")
                        return

                    # Set rotation angle
                    if brushless_device is not None:
                        set_angle(core, brushless_device, angle)

                    # Set exposure time if specified
                    if angle_idx < len(params["exposures"]):
                        exposure_ms = params["exposures"][angle_idx]
                        core.set_exposure(exposure_ms)

                    # Acquire image
                    image, metadata = hardware.snap_image()  # type: ignore[attr-defined]

                    # Save image
                    image_path = output_path / str(angle) / filename
                    if image_path.parent.exists():
                        smartpath_qpscope.ome_writer(
                            filename=str(image_path),
                            pixel_size_um=ppm_settings.imagingMode.BF_10x.pixelSize_um,  # type: ignore[attr-defined]
                            data=image,
                        )
                        image_count += 1
                        update_progress(image_count, total_images)
                    else:
                        logger.error(f"Failed to save {image_path} - parent directory missing")
            else:
                # Single image acquisition
                image, metadata = hardware.snap_image()  # type: ignore[attr-defined]
                image_path = output_path / filename

                if image_path.parent.exists():
                    smartpath_qpscope.ome_writer(
                        filename=str(image_path),
                        pixel_size_um=ppm_settings.imagingMode.BF_10x.pixelSize_um,  # type: ignore[attr-defined]
                        data=image,
                    )
                    image_count += 1
                    update_progress(image_count, total_images)

        # Save device properties
        current_props = sp.get_device_properties(core)
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


