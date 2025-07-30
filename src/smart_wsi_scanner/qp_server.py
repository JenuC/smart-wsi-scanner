"""
QuPath Microscope Server - Enhanced Version
===========================================

A socket-based server that provides remote control of a microscope through Micro-Manager.
Handles stage movement, image acquisition, and multi-angle imaging workflows.

Enhanced Features:
- Acquisition status monitoring
- Real-time progress updates
- Acquisition cancellation support
- Non-blocking socket communication during acquisition
- Improved state management and logging
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
import enum
from threading import Lock
import logging
from datetime import datetime


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'qp_server_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Server configuration
HOST = "0.0.0.0"  # Listen on all interfaces
PORT = TCP_PORT   # Default: 5000

# Threading events for coordination
shutdown_event = threading.Event()

# Global acquisition state management
class AcquisitionState(enum.Enum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    CANCELLING = "CANCELLING"
    CANCELLED = "CANCELLED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

# Global acquisition tracking
acquisition_states = {}  # addr -> AcquisitionState
acquisition_progress = {}  # addr -> (current, total)
acquisition_locks = {}  # addr -> Lock
acquisition_cancel_events = {}  # addr -> Event


def _pycromanager():
    """Initialize Pycro-Manager connection to Micro-Manager."""
    logger.info("Initializing Pycro-Manager connection...")
    core, studio = init_pycromanager()
    if not core:
        logger.error("Failed to initialize Micro-Manager connection")
        sys.exit(1)
    logger.info("Pycro-Manager initialized successfully")
    return core, studio


# Initialize hardware connections
logger.info("Loading configuration...")
config_manager = ConfigManager()
ppm_settings = config_manager.get_config("config_PPM")
core, studio = _pycromanager()
hardware = PycromanagerHardware(core, ppm_settings, studio)
brushless = "KBD101_Thor_Rotation"  # Rotation stage device name
logger.info("Hardware initialization complete")


# ============================================================================
# Enhanced Command Enumeration
# ============================================================================

# Extend the Command enum with new commands
class ExtendedCommand:
    """Extended commands for enhanced acquisition control."""
    # Existing commands from Command enum
    GETXY = Command.GETXY.value
    GETZ = Command.GETZ.value
    MOVEZ = Command.MOVEZ.value
    MOVE = Command.MOVE.value
    GETR = Command.GETR.value
    MOVER = Command.MOVER.value
    SHUTDOWN = Command.SHUTDOWN.value
    DISCONNECT = Command.DISCONNECT.value
    ACQUIRE = Command.ACQUIRE.value
    GET = Command.GET.value
    SET = Command.SET.value
    
    # New commands (8 bytes each)
    STATUS = b"status__"    # Get acquisition status
    PROGRESS = b"progress"  # Get acquisition progress
    CANCEL = b"cancel__"    # Cancel acquisition


# ============================================================================
# Rotation Stage Control Functions
# ============================================================================

def ppm_to_thor(angle):
    """Convert PPM angle (in degrees) to Thor rotation stage position."""
    return -2 * angle + 276


def thor_to_ppm(kinesis_pos):
    """Convert Thor rotation stage position to PPM angle (in degrees)."""
    return (276 - kinesis_pos) / 2


def set_angle(theta):
    """Set the rotation stage to a specific angle and wait for completion."""
    theta_thor = ppm_to_thor(theta)
    logger.debug(f"Setting rotation angle to {theta}° (Thor position: {theta_thor})")
    core.set_position(brushless, theta_thor)
    core.wait_for_device(brushless)
    logger.debug(f"Rotation complete at {theta}°")


# ============================================================================
# Tile Configuration Processing
# ============================================================================

def read_tile_config(tile_config_path):
    """Read tile positions from a QuPath-generated TileConfiguration.txt file."""
    positions = []
    if tile_config_path.exists():
        logger.debug(f"Reading tile configuration from {tile_config_path}")
        with open(tile_config_path, "r") as f:
            for line in f:
                pattern = r"^([\w\-\.]+); ; \(\s*([-\d.]+),\s*([-\d.]+)"
                m = re.match(pattern, line)
                if m:
                    filename = m.group(1)
                    x = float(m.group(2))
                    y = float(m.group(3))
                    z = core.get_position()
                    positions.append([sp_position(x, y, z), filename])
        logger.info(f"Loaded {len(positions)} tile positions from configuration")
    else:
        logger.error(f"Tile configuration file {tile_config_path} does not exist.")
    return positions


# ============================================================================
# Argument Parsing Functions
# ============================================================================

def parse_angles_exposures(angles_str, exposures_str=None):
    """Parse angle and exposure strings from various formats."""
    angles = []
    exposures = []
    
    # Parse angles
    if isinstance(angles_str, list):
        angles = angles_str
    elif isinstance(angles_str, str):
        angles_str = angles_str.strip("()")
        if ',' in angles_str:
            angles = [float(x.strip()) for x in angles_str.split(',')]
        else:
            angles = [float(x) for x in angles_str.split()]
    
    # Parse exposures if provided
    if exposures_str:
        if isinstance(exposures_str, list):
            exposures = exposures_str
        elif isinstance(exposures_str, str):
            exposures_str = exposures_str.strip("()")
            if ',' in exposures_str:
                exposures = [int(x.strip()) for x in exposures_str.split(',')]
            else:
                exposures = [int(x) for x in exposures_str.split()]
    
    # Default exposures if not provided
    if not exposures and angles:
        exposures = []
        for angle in angles:
            if angle == 90.0:
                exposures.append(10)
            elif angle == 0.0:
                exposures.append(800)
            else:
                exposures.append(500)
    
    logger.debug(f"Parsed angles: {angles}, exposures: {exposures}")
    return angles, exposures


def parse_acquisition_message(message):
    """Parse acquisition message supporting both legacy and new flag-based formats."""
    logger.debug(f"Parsing acquisition message: {message[:200]}...")
    
    # Remove END_MARKER if present
    message = message.replace(' END_MARKER', '').replace('END_MARKER', '').strip()
    
    # Check if it's flag-based format
    if '--' in message:
        # Parse flag-based format
        logger.debug("Detected flag-based format")
        params = {}
        
        # Split by spaces but preserve quoted strings
        import shlex
        try:
            # For Windows compatibility, temporarily replace backslashes
            # This prevents shlex from treating them as escape characters
            temp_message = message.replace('\\', '|||BACKSLASH|||')
            parts = shlex.split(temp_message)
            # Restore backslashes
            parts = [part.replace('|||BACKSLASH|||', '\\') for part in parts]
        except:
            # Fallback to simple split if shlex fails
            parts = message.split()
        
        i = 0
        while i < len(parts):
            if parts[i] == '--yaml' and i + 1 < len(parts):
                params['yaml_file_path'] = parts[i + 1]
                i += 2
            elif parts[i] == '--projects' and i + 1 < len(parts):
                params['projects_folder_path'] = parts[i + 1]
                i += 2
            elif parts[i] == '--sample' and i + 1 < len(parts):
                params['sample_label'] = parts[i + 1]
                i += 2
            elif parts[i] == '--scan-type' and i + 1 < len(parts):
                params['scan_type'] = parts[i + 1]
                i += 2
            elif parts[i] == '--region' and i + 1 < len(parts):
                params['region_name'] = parts[i + 1]
                i += 2
            elif parts[i] == '--angles' and i + 1 < len(parts):
                params['angles_str'] = parts[i + 1]
                i += 2
            elif parts[i] == '--exposures' and i + 1 < len(parts):
                params['exposures_str'] = parts[i + 1]
                i += 2
            elif parts[i] == '--laser-power' and i + 1 < len(parts):
                params['laser_power'] = float(parts[i + 1])
                i += 2
            elif parts[i] == '--laser-wavelength' and i + 1 < len(parts):
                params['laser_wavelength'] = int(parts[i + 1])
                i += 2
            elif parts[i] == '--dwell-time' and i + 1 < len(parts):
                params['dwell_time'] = float(parts[i + 1])
                i += 2
            elif parts[i] == '--averaging' and i + 1 < len(parts):
                params['averaging'] = int(parts[i + 1])
                i += 2
            elif parts[i] == '--z-stack':
                params['z_stack_enabled'] = True
                i += 1
            elif parts[i] == '--z-start' and i + 1 < len(parts):
                params['z_start'] = float(parts[i + 1])
                i += 2
            elif parts[i] == '--z-end' and i + 1 < len(parts):
                params['z_end'] = float(parts[i + 1])
                i += 2
            elif parts[i] == '--z-step' and i + 1 < len(parts):
                params['z_step'] = float(parts[i + 1])
                i += 2
            else:
                logger.debug(f"Unknown flag or argument: {parts[i]}")
                i += 1
        
        # Parse angles and exposures if present
        angles, exposures = parse_angles_exposures(
            params.get('angles_str', '()'),
            params.get('exposures_str', None)
        )
        params['angles'] = angles
        params['exposures'] = exposures
        
        # Validate required parameters
        required = ['yaml_file_path', 'projects_folder_path', 'sample_label', 'scan_type', 'region_name']
        missing = [key for key in required if key not in params]
        if missing:
            raise ValueError(f"Missing required parameters: {missing}")
        
        logger.info(f"Parsed flag-based parameters: {params}")
        return params



# ============================================================================
# Enhanced Acquisition Workflow with Progress and Cancellation
# ============================================================================

def acquisitionWorkflow(message, client_addr):
    """
    Execute the main image acquisition workflow with progress tracking and cancellation support.
    
    Args:
        message: Command string containing acquisition parameters
        client_addr: Client address tuple for tracking state
    """
    logger.info(f"=== ACQUISITION WORKFLOW STARTED for client {client_addr} ===")
    
    try:
        # Parse the acquisition parameters
        params = parse_acquisition_message(message)
        
        modality = "_".join(params['scan_type'].split("_")[:2])
        
        logger.info(f"Acquisition parameters:")
        logger.info(f"  Client: {client_addr}")
        logger.info(f"  Modality: {modality}")
        logger.info(f"  Sample label: {params['sample_label']}")
        logger.info(f"  Scan type: {params['scan_type']}")
        logger.info(f"  Region: {params['region_name']}")
        logger.info(f"  Angles: {params['angles']} degrees")
        logger.info(f"  Exposures: {params['exposures']} ms")
        
        # Set up output paths
        project_path = pathlib.Path(params['projects_folder_path']) / params['sample_label']
        output_path = project_path / params['scan_type'] / params['region_name']
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created output directory: {output_path}")
        
        # Read tile positions
        tile_config_path = output_path / "TileConfiguration.txt"
        positions = read_tile_config(tile_config_path)
        
        if not positions:
            logger.error(f"No positions found in {tile_config_path}")
            with acquisition_locks[client_addr]:
                acquisition_states[client_addr] = AcquisitionState.FAILED
            return
        
        # Create subdirectories for each angle
        if params['angles']:
            for angle in params['angles']:
                angle_dir = output_path / str(angle)
                angle_dir.mkdir(exist_ok=True)
                shutil.copy2(tile_config_path, angle_dir / "TileConfiguration.txt")
                logger.debug(f"Created angle directory: {angle_dir}")
        
        # Initialize smartpath
        sp = smartpath(core)
        
        # Calculate total images and update progress
        total_images = len(positions) * len(params['angles']) if params['angles'] else len(positions)
        with acquisition_locks[client_addr]:
            acquisition_progress[client_addr] = (0, total_images)
        logger.info(f"Starting acquisition of {total_images} total images ({len(positions)} positions × {len(params['angles'])} angles)")
        
        image_count = 0
        
        # Main acquisition loop
        for pos_idx, (pos, filename) in enumerate(positions):
            # Check for cancellation
            if acquisition_cancel_events[client_addr].is_set():
                logger.warning(f"Acquisition cancelled by client {client_addr}")
                with acquisition_locks[client_addr]:
                    acquisition_states[client_addr] = AcquisitionState.CANCELLED
                return
            
            logger.info(f"Position {pos_idx + 1}/{len(positions)}: {filename}")
            
            # Move to position
            logger.debug(f"Moving to position: X={pos.x}, Y={pos.y}, Z={pos.z}")
            hardware.move_to_position(pos)
            
            if params['angles']:
                # Multi-angle acquisition
                for angle_idx, angle in enumerate(params['angles']):
                    # Check for cancellation
                    if acquisition_cancel_events[client_addr].is_set():
                        logger.warning(f"Acquisition cancelled by client {client_addr}")
                        with acquisition_locks[client_addr]:
                            acquisition_states[client_addr] = AcquisitionState.CANCELLED
                        return
                    
                    # Set rotation angle
                    set_angle(angle)
                    
                    # Set exposure time if specified
                    if angle_idx < len(params['exposures']):
                        exposure_ms = params['exposures'][angle_idx]
                        core.set_exposure(exposure_ms)
                        logger.debug(f"Set exposure to {exposure_ms}ms for angle {angle}°")
                    
                    # Acquire image
                    logger.debug(f"Acquiring image at angle {angle}°")
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
                        logger.info(f"Saved tile {image_count}/{total_images}: {image_path}")
                        
                        # Update progress
                        with acquisition_locks[client_addr]:
                            acquisition_progress[client_addr] = (image_count, total_images)
                    else:
                        logger.error(f"Failed to save {image_path} - parent directory missing")
            else:
                # Single image acquisition
                logger.debug("Acquiring single image (no rotation)")
                image, metadata = hardware.snap_image()
                image_path = output_path / filename
                
                if image_path.parent.exists():
                    smartpath_qpscope.ome_writer(
                        filename=str(image_path),
                        pixel_size_um=ppm_settings.imagingMode.BF_10x.pixelSize_um,
                        data=image,
                    )
                    image_count += 1
                    logger.info(f"Saved tile {image_count}/{total_images}: {image_path}")
                    
                    # Update progress
                    with acquisition_locks[client_addr]:
                        acquisition_progress[client_addr] = (image_count, total_images)
        
        # Save device properties
        current_props = sp.get_device_properties(core)
        props_path = output_path / "MMproperties.txt"
        with open(props_path, "w") as fid:
            dict_printer(current_props, stream=fid)
        logger.info(f"Saved device properties to {props_path}")
        
        logger.info(f"=== ACQUISITION COMPLETED SUCCESSFULLY ===")
        logger.info(f"Total images saved: {image_count}/{total_images}")
        logger.info(f"Output directory: {output_path}")
        
        # Update final state
        with acquisition_locks[client_addr]:
            acquisition_states[client_addr] = AcquisitionState.COMPLETED
        
    except Exception as e:
        logger.error(f"=== ACQUISITION FAILED ===")
        logger.error(f"Error: {str(e)}", exc_info=True)
        
        # Update state to failed
        with acquisition_locks[client_addr]:
            acquisition_states[client_addr] = AcquisitionState.FAILED


# ============================================================================
# Enhanced Client Connection Handler
# ============================================================================

def handle_client(conn, addr):
    """
    Handle commands from a connected QuPath client with enhanced acquisition control.
    """
    logger.info(f">>> New client connected from {addr}")
    
    # Initialize client state
    acquisition_locks[addr] = Lock()
    acquisition_states[addr] = AcquisitionState.IDLE
    acquisition_progress[addr] = (0, 0)
    acquisition_cancel_events[addr] = threading.Event()
    
    acquisition_thread = None
    
    try:
        while True:
            # All commands are 8 bytes
            data = conn.recv(8)
            if not data:
                logger.info(f"Client {addr} disconnected (no data)")
                break
            
            logger.debug(f"Received command from {addr}: {data}")
            
            # Connection management commands
            if data == ExtendedCommand.DISCONNECT:
                logger.info(f"Client {addr} requested to disconnect")
                break
            
            if data == ExtendedCommand.SHUTDOWN:
                logger.warning(f"Client {addr} requested server shutdown")
                shutdown_event.set()
                break
            
            # Position query commands
            if data == ExtendedCommand.GETXY:
                logger.debug(f"Client {addr} requested XY position")
                current_position_xyz = hardware.get_current_position()
                response = struct.pack("!ff", current_position_xyz.x, current_position_xyz.y)
                conn.sendall(response)
                logger.debug(f"Sent XY position to {addr}: ({current_position_xyz.x}, {current_position_xyz.y})")
                continue
            
            if data == ExtendedCommand.GETZ:
                logger.debug(f"Client {addr} requested Z position")
                current_position_xyz = hardware.get_current_position()
                response = struct.pack("!f", current_position_xyz.z)
                conn.sendall(response)
                logger.debug(f"Sent Z position to {addr}: {current_position_xyz.z}")
                continue
            
            if data == ExtendedCommand.GETR:
                logger.debug(f"Client {addr} requested rotation angle")
                kinesis_pos = core.get_position(brushless)
                angle = thor_to_ppm(kinesis_pos)
                response = struct.pack("!f", angle)
                conn.sendall(response)
                logger.debug(f"Sent rotation angle to {addr}: {angle}°")
                continue
            
            # Movement commands
            if data == ExtendedCommand.MOVE:
                coords = conn.recv(8)
                if len(coords) == 8:
                    x, y = struct.unpack("!ff", coords)
                    logger.info(f"Client {addr} requested move to: X={x}, Y={y}")
                    hardware.move_to_position(sp_position(x, y))
                    logger.info(f"Move completed to X={x}, Y={y}")
                else:
                    logger.error(f"Client {addr} sent incomplete move coordinates")
                continue
            
            if data == ExtendedCommand.MOVEZ:
                z = conn.recv(4)
                z_position = struct.unpack("!f", z)[0]
                logger.info(f"Client {addr} requested move to Z={z_position}")
                hardware.move_to_position(sp_position(z=z_position))
                logger.info(f"Move completed to Z={z_position}")
                continue
            
            if data == ExtendedCommand.MOVER:
                coords = conn.recv(4)
                angle = struct.unpack("!f", coords)[0]
                logger.info(f"Client {addr} requested rotation to {angle}°")
                set_angle(angle)
                logger.info(f"Rotation completed to {angle}°")
                continue
            
            # ============ ACQUISITION STATUS COMMANDS ============
            
            # Status query command
            if data == ExtendedCommand.STATUS:
                with acquisition_locks[addr]:
                    state = acquisition_states[addr]
                # Send state as 16-byte string (padded)
                state_str = state.value.ljust(16)[:16]
                conn.sendall(state_str.encode())
                logger.debug(f"Sent acquisition status to {addr}: {state.value}")
                continue
            
            # Progress query command
            if data == ExtendedCommand.PROGRESS:
                with acquisition_locks[addr]:
                    current, total = acquisition_progress[addr]
                # Send as two integers
                response = struct.pack("!II", current, total)
                conn.sendall(response)
                logger.debug(f"Sent progress to {addr}: {current}/{total}")
                continue
            
            # Cancel acquisition command
            if data == ExtendedCommand.CANCEL:
                logger.warning(f"Client {addr} requested acquisition cancellation")
                with acquisition_locks[addr]:
                    if acquisition_states[addr] == AcquisitionState.RUNNING:
                        acquisition_states[addr] = AcquisitionState.CANCELLING
                        acquisition_cancel_events[addr].set()
                        logger.info(f"Cancellation initiated for {addr}")
                # Send acknowledgment
                conn.sendall(b"ACK")
                continue
            
            # ============ ACQUISITION COMMAND ============
            
            if data == ExtendedCommand.ACQUIRE:
                logger.info(f"Client {addr} requested acquisition workflow")
                
                # Check if already running
                with acquisition_locks[addr]:
                    if acquisition_states[addr] == AcquisitionState.RUNNING:
                        logger.warning(f"Acquisition already running for {addr}")
                        continue
                    # Set state to RUNNING immediately
                    acquisition_states[addr] = AcquisitionState.RUNNING
                    acquisition_progress[addr] = (0, 0)
                
                # Read the full message immediately
                message_parts = []
                total_bytes = 0
                start_time = time.time()
                
                # Set a timeout for reading
                conn.settimeout(5.0)
                
                try:
                    while True:
                        # Read in chunks
                        chunk = conn.recv(1024)
                        if not chunk:
                            logger.error(f"Connection closed while reading acquisition message from {addr}")
                            with acquisition_locks[addr]:
                                acquisition_states[addr] = AcquisitionState.FAILED
                            break
                        
                        message_parts.append(chunk.decode('utf-8'))
                        total_bytes += len(chunk)
                        
                        # Check if we have the end marker
                        full_message = ''.join(message_parts)
                        if 'END_MARKER' in full_message:
                            # Remove the end marker
                            message = full_message.replace(',END_MARKER', '').replace('END_MARKER', '')
                            logger.debug(f"Received complete acquisition message ({total_bytes} bytes) in {time.time() - start_time:.2f}s")
                            
                            # Clear cancellation event
                            acquisition_cancel_events[addr].clear()
                            
                            # Start acquisition in separate thread
                            acquisition_thread = threading.Thread(
                                target=acquisitionWorkflow, 
                                args=(message, addr), 
                                daemon=True,
                                name=f"Acquisition-{addr}"
                            )
                            acquisition_thread.start()
                            
                            logger.info(f"Acquisition thread started for {addr}")
                            break
                        
                        # Safety check for message size
                        if total_bytes > 10000:  # 10KB max
                            logger.error(f"Acquisition message too large from {addr}: {total_bytes} bytes")
                            with acquisition_locks[addr]:
                                acquisition_states[addr] = AcquisitionState.FAILED
                            break
                            
                        # Timeout check
                        if time.time() - start_time > 10:
                            logger.error(f"Timeout reading acquisition message from {addr}")
                            with acquisition_locks[addr]:
                                acquisition_states[addr] = AcquisitionState.FAILED
                            break
                    
                except socket.timeout:
                    logger.error(f"Socket timeout reading acquisition message from {addr}")
                    with acquisition_locks[addr]:
                        acquisition_states[addr] = AcquisitionState.FAILED
                except Exception as e:
                    logger.error(f"Error reading acquisition message from {addr}: {e}")
                    with acquisition_locks[addr]:
                        acquisition_states[addr] = AcquisitionState.FAILED
                finally:
                    # Reset socket to blocking mode
                    conn.settimeout(None)
                
                continue
            
            # Legacy GET/SET commands (not implemented)
            if data == ExtendedCommand.GET:
                logger.debug("GET property not yet implemented")
                continue
            
            if data == ExtendedCommand.SET:
                logger.debug("SET property not yet implemented")
                continue
            
            # Unknown command
            logger.warning(f"Unknown command from {addr}: {data}")
    
    except Exception as e:
        logger.error(f"Error handling client {addr}: {str(e)}", exc_info=True)
    finally:
        # Cleanup
        if acquisition_thread and acquisition_thread.is_alive():
            logger.info(f"Cancelling acquisition for disconnected client {addr}")
            acquisition_cancel_events[addr].set()
            acquisition_thread.join(timeout=10)
        
        # Remove client state
        if addr in acquisition_locks:
            del acquisition_locks[addr]
        if addr in acquisition_states:
            del acquisition_states[addr]
        if addr in acquisition_progress:
            del acquisition_progress[addr]
        if addr in acquisition_cancel_events:
            del acquisition_cancel_events[addr]
        
        conn.close()
        logger.info(f"<<< Client {addr} disconnected and cleaned up")


# ============================================================================
# Main Server Loop
# ============================================================================

def main():
    """Main server loop that accepts client connections and spawns handler threads."""
    logger.info("="*60)
    logger.info("QuPath Microscope Server - Enhanced Version")
    logger.info("="*60)
    logger.info(f"Server configuration:")
    logger.info(f"  Host: {HOST}")
    logger.info(f"  Port: {PORT}")
    logger.info(f"  Micro-Manager core: {'Connected' if core else 'Not connected'}")
    logger.info(f"  Hardware: {'Initialized' if hardware else 'Not initialized'}")
    logger.info(f"Features:")
    logger.info(f"  - Status monitoring")
    logger.info(f"  - Progress tracking")
    logger.info(f"  - Cancellation support")
    logger.info(f"  - Enhanced logging")
    logger.info("="*60)
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen()
        logger.info(f"Server listening on {HOST}:{PORT}")
        logger.info("Ready for connections...")
        
        threads = []
        
        while not shutdown_event.is_set():
            try:
                s.settimeout(1.0)
                conn, addr = s.accept()
                thread = threading.Thread(
                    target=handle_client, 
                    args=(conn, addr), 
                    daemon=True
                )
                thread.start()
                threads.append(thread)
            except socket.timeout:
                continue
            except OSError:
                break
        
        logger.info("Server shutting down. Waiting for client threads to finish...")
        shutdown_event.set()
        
        for t in threads:
            t.join(timeout=5.0)
        
        logger.info("Server has shut down.")


if __name__ == "__main__":
    main()