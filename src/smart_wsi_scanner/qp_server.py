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
import sys
import pathlib
import time
import enum
from threading import Lock
import logging
from datetime import datetime

from smart_wsi_scanner.config import ConfigManager, sp_position
from smart_wsi_scanner.hardware import PycromanagerHardware, init_pycromanager
from smart_wsi_scanner.qp_server_config import Command, TCP_PORT, END_MARKER

from smart_wsi_scanner.qp_acquisition import (
    acquisition_workflow,
    # thor_to_ppm,
    # set_angle as acquisition_set_angle,
)


# Configure logging
current_file_path = pathlib.Path(__file__).resolve()
base_dir = current_file_path.parent  # e.g., smart-wsi-scanner/src/smart_wsi_scanner
log_dir = base_dir / "server_logfiles"
log_dir.mkdir(parents=True, exist_ok=True)  # Create it if it doesn't exist
filename = log_dir / f'qp_server_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(filename), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# Server configuration
HOST = "0.0.0.0"  # Listen on all interfaces
PORT = TCP_PORT  # Default: 5000

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


def int_pycromanager_with_logger():
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
core, studio = int_pycromanager_with_logger()
hardware = PycromanagerHardware(core, ppm_settings, studio)
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
    STATUS = b"status__"  # Get acquisition status
    PROGRESS = b"progress"  # Get acquisition progress
    CANCEL = b"cancel__"  # Cancel acquisition


def acquisitionWorkflow(message, client_addr):
    """Deprecated: acquisition moved to qp_acquisition.acquisition_workflow"""

    def _update_progress(current: int, total: int):
        with acquisition_locks[client_addr]:
            acquisition_progress[client_addr] = (current, total)

    def _set_state(state_str: str):
        with acquisition_locks[client_addr]:
            try:
                acquisition_states[client_addr] = AcquisitionState[state_str]
            except KeyError:
                acquisition_states[client_addr] = AcquisitionState.FAILED

    def _is_cancelled() -> bool:
        return acquisition_cancel_events[client_addr].is_set()

    return acquisition_workflow(
        message,
        client_addr,
        core=core,
        hardware=hardware,
        config_manager=config_manager,
        logger=logger,
        update_progress=_update_progress,
        set_state=_set_state,
        is_cancelled=_is_cancelled,
        brushless_device=brushless,
    )


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
                logger.debug(
                    f"Sent XY position to {addr}: ({current_position_xyz.x}, {current_position_xyz.y})"
                )
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
                acquisition_set_angle(core, brushless, angle)
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
                            logger.error(
                                f"Connection closed while reading acquisition message from {addr}"
                            )
                            with acquisition_locks[addr]:
                                acquisition_states[addr] = AcquisitionState.FAILED
                            break

                        message_parts.append(chunk.decode("utf-8"))
                        total_bytes += len(chunk)

                        # Check if we have the end marker
                        full_message = "".join(message_parts)
                        if "END_MARKER" in full_message:
                            # Remove the end marker
                            message = full_message.replace(",END_MARKER", "").replace(
                                "END_MARKER", ""
                            )
                            logger.debug(
                                f"Received complete acquisition message ({total_bytes} bytes) in {time.time() - start_time:.2f}s"
                            )

                            # Clear cancellation event
                            acquisition_cancel_events[addr].clear()

                            # Start acquisition in separate thread
                            acquisition_thread = threading.Thread(
                                target=acquisitionWorkflow,
                                args=(message, addr),
                                daemon=True,
                                name=f"Acquisition-{addr}",
                            )
                            acquisition_thread.start()

                            logger.info(f"Acquisition thread started for {addr}")
                            break

                        # Safety check for message size
                        if total_bytes > 10000:  # 10KB max
                            logger.error(
                                f"Acquisition message too large from {addr}: {total_bytes} bytes"
                            )
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


def main():
    """Main server loop that accepts client connections and spawns handler threads."""
    logger.info("=" * 60)
    logger.info("QuPath Microscope Server - Enhanced Version")
    logger.info("=" * 60)
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
    logger.info("=" * 60)

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
                thread = threading.Thread(target=handle_client, args=(conn, addr), daemon=True)
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
