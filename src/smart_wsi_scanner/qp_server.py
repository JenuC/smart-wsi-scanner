"""
QuPath Microscope Server
========================

A socket-based server that provides remote control of a microscope through Micro-Manager.
Handles stage movement, image acquisition, and multi-angle imaging workflows.

Key Features:
- Socket communication with QuPath extension
- Support for both legacy and new flag-based command formats
- Multi-angle PPM (Polarized Light Microscopy) acquisition
- Exposure time control per angle
- Thread-safe client handling
- Graceful shutdown support
"""

import socket
import threading
import struct
from smart_wsi_scanner.smartpath import init_pycromanager
from smart_wsi_scanner.config import ConfigManager, sp_position
from smart_wsi_scanner.hardware import PycromanagerHardware
import sys
from smart_wsi_scanner.qp_server_config import Command, TCP_PORT, END_MARKER
import pathlib
import re
from smart_wsi_scanner.smartpath import smartpath
from smart_wsi_scanner.smartpath_qpscope import smartpath_qpscope
from pprint import pprint as dict_printer
import shutil
import time


# Server configuration
HOST = "0.0.0.0"  # Listen on all interfaces
PORT = TCP_PORT  # Default: 5000

# Threading events for coordination
shutdown_event = threading.Event()
wait_for_function_event = threading.Event()


def _pycromanager():
    """Initialize Pycro-Manager connection to Micro-Manager."""
    core, studio = init_pycromanager()
    if not core:
        print("Failed to initialize Micro-Manager connection")
        sys.exit(1)
    return core, studio


# Initialize hardware connections
config_manager = ConfigManager()
ppm_settings = config_manager.get_config("config_PPM")
core, studio = _pycromanager()
hardware = PycromanagerHardware(core, ppm_settings, studio)
brushless = "KBD101_Thor_Rotation"  # Rotation stage device name


# ============================================================================
# Rotation Stage Control Functions
# ============================================================================


def ppm_to_thor(angle):
    """
    Convert PPM angle (in degrees) to Thor rotation stage position.

    The Thor stage uses a different coordinate system than PPM convention.
    This function applies the necessary transformation.

    Args:
        angle: PPM angle in degrees

    Returns:
        Thor stage position
    """
    return -2 * angle + 276


def thor_to_ppm(kinesis_pos):
    """
    Convert Thor rotation stage position to PPM angle (in degrees).

    Args:
        kinesis_pos: Thor stage position

    Returns:
        PPM angle in degrees
    """
    return (276 - kinesis_pos) / 2


def set_angle(theta):
    """
    Set the rotation stage to a specific angle and wait for completion.

    Args:
        theta: Target angle in PPM degrees
    """
    theta_thor = ppm_to_thor(theta)
    core.set_position(brushless, theta_thor)
    core.wait_for_device(brushless)


# ============================================================================
# Tile Configuration Processing
# ============================================================================


def read_tile_config(tile_config_path):
    """
    Read tile positions from a QuPath-generated TileConfiguration.txt file.

    The file format is:
    filename.tif; ; (x_position, y_position)

    Args:
        tile_config_path: Path to TileConfiguration.txt

    Returns:
        List of [position, filename] pairs where position is sp_position object
    """
    positions = []
    if tile_config_path.exists():
        with open(tile_config_path, "r") as f:
            for line in f:
                # Pattern matches: filename.tif; ; (x, y)
                pattern = r"^([\w\-\.]+); ; \(\s*([-\d.]+),\s*([-\d.]+)"
                m = re.match(pattern, line)
                if m:
                    filename = m.group(1)
                    x = float(m.group(2))
                    y = float(m.group(3))
                    # Z position from current stage position if not specified
                    z = core.get_position()
                    positions.append([sp_position(x, y, z), filename])
    else:
        print(f"Tile configuration file {tile_config_path} does not exist.")
    return positions


# ============================================================================
# Argument Parsing for Flexible Command Support
# ============================================================================


def parse_angles_exposures(angles_str, exposures_str=None):
    """
    Parse angle and exposure strings from various formats.

    Supports:
    1. Legacy format: "(angle1 angle2 angle3)"  - space-separated in parentheses
    2. New format: "(angle1,angle2,angle3)"     - comma-separated in parentheses
    3. Direct list: [angle1, angle2, angle3]    - for socket communication

    Args:
        angles_str: String or list containing angles
        exposures_str: Optional string containing exposure times

    Returns:
        Tuple of (angles_list, exposures_list)
    """
    angles = []
    exposures = []

    # Parse angles
    if isinstance(angles_str, list):
        # Direct list from socket
        angles = angles_str
    elif isinstance(angles_str, str):
        # Remove parentheses and parse
        angles_str = angles_str.strip("()")
        # Try comma-separated first (new format)
        if "," in angles_str:
            angles = [float(x.strip()) for x in angles_str.split(",")]
        else:
            # Fall back to space-separated (legacy format)
            angles = [float(x) for x in angles_str.split()]

    # Parse exposures if provided
    if exposures_str:
        if isinstance(exposures_str, list):
            exposures = exposures_str
        elif isinstance(exposures_str, str):
            exposures_str = exposures_str.strip("()")
            if "," in exposures_str:
                exposures = [int(x.strip()) for x in exposures_str.split(",")]
            else:
                exposures = [int(x) for x in exposures_str.split()]

    # If no exposures provided, use default based on angle
    if not exposures and angles:
        exposures = []
        for angle in angles:
            if angle == 90.0:
                exposures.append(10)  # Default brightfield exposure
            elif angle == 0.0:
                exposures.append(800)  # Default 0° exposure
            else:
                exposures.append(500)  # Default for other angles

    return angles, exposures


def parse_acquisition_message(message):
    """
    Parse acquisition message supporting both legacy and new formats.

    Legacy format: yaml,projects,sample,scanType,region,(angles)
    New format: --yaml path --projects path --sample name --scan-type type --region name --angles (a1,a2) --exposures (e1,e2)

    Args:
        message: Command string from QuPath

    Returns:
        Dictionary with parsed parameters
    """
    # Check if this is the new flag-based format
    if "--yaml" in message:
        # Parse flag-based format
        params = {}
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

        # Parse angles and exposures
        angles, exposures = parse_angles_exposures(
            params.get("angles_str", "()"), params.get("exposures_str", None)
        )
        params["angles"] = angles
        params["exposures"] = exposures

        return params

    else:
        # Legacy comma-separated format
        parts = message.split(",")
        if len(parts) < 6:
            raise ValueError(
                f"Legacy format requires at least 6 comma-separated values, got {len(parts)}"
            )

        angles, exposures = parse_angles_exposures(parts[5])

        return {
            "yaml_file_path": parts[0],
            "projects_folder_path": parts[1],
            "sample_label": parts[2],
            "scan_type": parts[3],
            "region_name": parts[4],
            "angles": angles,
            "exposures": exposures,
        }


# ============================================================================
# Main Acquisition Workflow
# ============================================================================


def acquisitionWorkflow(message, test=None):
    """
    Execute the main image acquisition workflow.

    This function:
    1. Parses acquisition parameters
    2. Creates output directories
    3. Reads tile positions from configuration
    4. Acquires images at each position and angle
    5. Saves images with proper metadata

    Args:
        message: Command string containing acquisition parameters
        test: Optional test parameter (unused)
    """
    try:
        # Parse the acquisition parameters
        params = parse_acquisition_message(message)

        # Extract modality from scan type (e.g., "PPM_10x_1" -> "PPM_10x")
        modality = "_".join(params["scan_type"].split("_")[:2])

        print("\n" + "=" * 60)
        print("ACQUISITION WORKFLOW STARTED")
        print("=" * 60)
        print(f"  Modality: {modality}")
        print(f"  YAML file: {params['yaml_file_path']}")
        print(f"  Projects folder: {params['projects_folder_path']}")
        print(f"  Sample label: {params['sample_label']}")
        print(f"  Scan type: {params['scan_type']}")
        print(f"  Region: {params['region_name']}")

        if params["angles"]:
            print(f"  Angles: {params['angles']} degrees")
            print(f"  Exposures: {params['exposures']} ms")

        # Set up output paths
        project_path = pathlib.Path(params["projects_folder_path"]) / params["sample_label"]
        output_path = project_path / params["scan_type"] / params["region_name"]
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)

        # Read tile positions
        tile_config_path = output_path / "TileConfiguration.txt"
        positions = read_tile_config(tile_config_path)

        if not positions:
            print(f"ERROR: No positions found in {tile_config_path}")
            return

        # Create subdirectories for each angle and copy tile configuration
        if params["angles"]:
            for angle in params["angles"]:
                angle_dir = output_path / str(angle)
                angle_dir.mkdir(exist_ok=True)
                # Copy tile configuration to each angle directory
                shutil.copy2(tile_config_path, angle_dir / "TileConfiguration.txt")

        # Initialize smartpath for device control
        sp = smartpath(core)

        # Main acquisition loop
        print(f"\nStarting acquisition of {len(positions)} positions...")
        total_images = (
            len(positions) * len(params["angles"]) if params["angles"] else len(positions)
        )
        image_count = 0

        for pos_idx, (pos, filename) in enumerate(positions):
            print(f"\nPosition {pos_idx + 1}/{len(positions)}: {filename}")

            # Move to position
            hardware.move_to_position(pos)

            if params["angles"]:
                # Multi-angle acquisition (e.g., PPM)
                for angle_idx, angle in enumerate(params["angles"]):
                    # Set rotation angle
                    set_angle(angle)
                    print(f"  - Angle {angle}° ", end="")

                    # Set exposure time if specified
                    if angle_idx < len(params["exposures"]):
                        exposure_ms = params["exposures"][angle_idx]
                        core.set_exposure(exposure_ms)
                        print(f"(exposure: {exposure_ms}ms)", end="")

                    # Acquire image
                    image, metadata = hardware.snap_image()

                    # Save image
                    image_path = output_path / str(angle) / filename
                    if image_path.parent.exists():
                        smartpath_qpscope.ome_writer(
                            filename=str(image_path),
                            pixel_size_um=ppm_settings.imagingMode.BF_10x.pixelSize_um,
                            data=image,
                        )
                        image_count += 1
                        # This print is REQUIRED for QuPath progress bar
                        print(f" -> Tile saved: {image_path}", flush=True)
                    else:
                        print(f" -> ERROR: Failed to save {image_path}")
            else:
                # Single image acquisition (standard brightfield)
                image, metadata = hardware.snap_image()
                image_path = output_path / filename

                if image_path.parent.exists():
                    smartpath_qpscope.ome_writer(
                        filename=str(image_path),
                        pixel_size_um=ppm_settings.imagingMode.BF_10x.pixelSize_um,
                        data=image,
                    )
                    image_count += 1
                    print(f"Tile saved: {image_path}", flush=True)

        # Save device properties for debugging
        current_props = sp.get_device_properties(core)
        with open(output_path / "MMproperties.txt", "w") as fid:
            dict_printer(current_props, stream=fid)

        print(f"\n{'='*60}")
        print(f"ACQUISITION COMPLETED SUCCESSFULLY")
        print(f"Total images saved: {image_count}/{total_images}")
        print(f"Output directory: {output_path}")
        print(f"{'='*60}\n")

    except Exception as e:
        print(f"\n{'='*60}")
        print(f"ERROR IN ACQUISITION WORKFLOW")
        print(f"{'='*60}")
        print(f"Error: {str(e)}")
        import traceback

        traceback.print_exc()
        print(f"{'='*60}\n")

    finally:
        # Signal completion regardless of success/failure
        wait_for_function_event.set()


# ============================================================================
# Client Connection Handler
# ============================================================================


def handle_client(conn, addr):
    """
    Handle commands from a connected QuPath client.

    Supports various commands:
    - Stage movement (XY, Z, rotation)
    - Position queries
    - Acquisition workflows
    - Connection management

    Args:
        conn: Socket connection object
        addr: Client address tuple (host, port)
    """
    print(f"Connected by client at {addr}")
    try:
        while True:
            # All commands are 8 bytes
            data = conn.recv(8)
            if not data:
                break

            # Connection management commands
            if data == Command.DISCONNECT.value:
                print(f"Client {addr} requested to quit.")
                break

            if data == Command.SHUTDOWN.value:
                print(f"Client {addr} requested server shutdown.")
                shutdown_event.set()
                break

            # Position query commands
            if data == Command.GETXY.value:
                current_position_xyz = hardware.get_current_position()
                response = struct.pack("!ff", current_position_xyz.x, current_position_xyz.y)
                conn.sendall(response)
                continue

            if data == Command.GETZ.value:
                current_position_xyz = hardware.get_current_position()
                response = struct.pack("!f", current_position_xyz.z)
                conn.sendall(response)
                continue

            if data == Command.GETR.value:
                kinesis_pos = core.get_position(brushless)
                response = struct.pack("!f", thor_to_ppm(kinesis_pos))
                conn.sendall(response)
                continue

            # Movement commands
            if data == Command.MOVE.value:
                coords = conn.recv(8)
                if len(coords) == 8:
                    x, y = struct.unpack("!ff", coords)
                    print(f"Client {addr} requested move to: x={x}, y={y}")
                    hardware.move_to_position(sp_position(x, y))
                else:
                    print(f"Client {addr} sent incomplete move coordinates")
                continue

            if data == Command.MOVEZ.value:
                z = conn.recv(4)
                z_position = struct.unpack("!f", z)[0]
                print(f"Client {addr} requested move to Z position: {z_position}")
                hardware.move_to_position(sp_position(z=z_position))
                continue

            if data == Command.MOVER.value:
                coords = conn.recv(4)
                angle = struct.unpack("!f", coords)[0]
                print(f"Client {addr} requested move to rotation angle: {angle}")
                set_angle(angle)
                continue

            # Acquisition command
            if data == Command.ACQUIRE.value:
                print(f"Client {addr} requested acquisition workflow.")
                message = ""
                # Read until END_MARKER
                while True:
                    chunk = conn.recv(1024)
                    if not chunk:
                        break
                    message += chunk.decode()
                    if str(END_MARKER) in message:
                        message = message.replace(END_MARKER, "")
                        break

                # Clear the event before starting
                wait_for_function_event.clear()

                # Run acquisition in separate thread to avoid blocking
                function_thread = threading.Thread(
                    target=acquisitionWorkflow, args=(message, "test"), daemon=True
                )
                function_thread.start()

                # Wait for acquisition to complete
                wait_for_function_event.wait()
                continue

            # TODO: Implement GET and SET property commands
            if data == Command.GET.value:
                print("GET property not yet implemented")
                continue

            if data == Command.SET.value:
                print("SET property not yet implemented")
                continue

    except Exception as e:
        print(f"Error handling client {addr}: {str(e)}")
    finally:
        conn.close()
        print(f"Connection closed for {addr}")


# ============================================================================
# Main Server Loop
# ============================================================================


def main():
    """
    Main server loop that accepts client connections and spawns handler threads.

    Features:
    - Multi-client support via threading
    - Graceful shutdown on SHUTDOWN command
    - Automatic cleanup of client threads
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # Allow socket reuse to avoid "Address already in use" errors
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen()
        print(f"\n{'='*60}")
        print(f"QuPath Microscope Server")
        print(f"{'='*60}")
        print(f"Listening on {HOST}:{PORT}")
        print(f"Micro-Manager core initialized: {core is not None}")
        print(f"Hardware initialized: {hardware is not None}")
        print(f"Ready for connections...")
        print(f"{'='*60}\n")

        threads = []

        while not shutdown_event.is_set():
            try:
                # Use timeout to check shutdown event periodically
                s.settimeout(1.0)
                conn, addr = s.accept()
                # Create daemon thread for each client
                thread = threading.Thread(target=handle_client, args=(conn, addr), daemon=True)
                thread.start()
                threads.append(thread)
            except socket.timeout:
                continue
            except OSError:
                break

        print("\nServer shutting down. Waiting for client threads to finish...")
        shutdown_event.set()

        # Wait for all client threads to complete
        for t in threads:
            t.join(timeout=5.0)

        print("Server has shut down.")


if __name__ == "__main__":
    main()
