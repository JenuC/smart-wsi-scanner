"""
Test Client for QuPath Microscope Server
========================================

This client tests all available server commands including:
- Position queries (GETXY, GETZ, GETFOV, GETR)
- Movement commands (MOVE, MOVEZ, MOVER)
- Acquisition commands (ACQUIRE, BGACQUIRE)
- Status monitoring (STATUS, PROGRESS, CANCEL)
- Connection management (DISCONNECT, SHUTDOWN)

Usage:
    python test_client.py [--host HOST] [--port PORT] [--test TEST_NAME]
"""

import socket
import struct
import time
import argparse
import logging
from datetime import datetime
from typing import Optional, Tuple, List
import sys
from pathlib import Path

# Import the command definitions
from smart_wsi_scanner.qp_server_config import ExtendedCommand, TCP_PORT, END_MARKER


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class QuPathTestClient:
    """Test client for QuPath microscope server."""

    def __init__(self, host: str = "127.0.0.1", port: int = TCP_PORT):
        """Initialize test client with server connection parameters."""
        self.host = host
        self.port = port
        self.socket: Optional[socket.socket] = None

    def connect(self) -> bool:
        """Connect to the server."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(10.0)  # 10 second timeout
            self.socket.connect((self.host, self.port))
            logger.info(f"Connected to server at {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False

    def disconnect(self):
        """Disconnect from server."""
        if self.socket:
            try:
                self.socket.send(ExtendedCommand.DISCONNECT)
                time.sleep(0.1)
            except:
                pass
            finally:
                self.socket.close()
                self.socket = None
                logger.info("Disconnected from server")

    def test_get_xy(self) -> Tuple[float, float]:
        """Test GETXY command."""
        logger.info("Testing GETXY...")
        try:
            self.socket.send(ExtendedCommand.GETXY)
            response = self.socket.recv(8)
            x, y = struct.unpack("!ff", response)
            logger.info(f"  Current XY position: X={x:.2f}, Y={y:.2f}")
            return x, y
        except Exception as e:
            logger.error(f"  GETXY failed: {e}")
            raise

    def test_get_z(self) -> float:
        """Test GETZ command."""
        logger.info("Testing GETZ...")
        try:
            self.socket.send(ExtendedCommand.GETZ)
            response = self.socket.recv(4)
            z = struct.unpack("!f", response)[0]
            logger.info(f"  Current Z position: {z:.2f}")
            return z
        except Exception as e:
            logger.error(f"  GETZ failed: {e}")
            raise

    def test_get_fov(self) -> Tuple[float, float]:
        """Test GETFOV command."""
        logger.info("Testing GETFOV...")
        try:
            self.socket.send(ExtendedCommand.GETFOV)
            response = self.socket.recv(8)
            fov_x, fov_y = struct.unpack("!ff", response)
            logger.info(f"  Field of View: X={fov_x:.2f}µm, Y={fov_y:.2f}µm")
            return fov_x, fov_y
        except Exception as e:
            logger.error(f"  GETFOV failed: {e}")
            raise

    def test_get_rotation(self) -> float:
        """Test GETR command."""
        logger.info("Testing GETR...")
        try:
            self.socket.send(ExtendedCommand.GETR)
            response = self.socket.recv(4)
            angle = struct.unpack("!f", response)[0]
            logger.info(f"  Current rotation angle: {angle:.1f}°")
            return angle
        except Exception as e:
            logger.error(f"  GETR failed: {e}")
            raise

    def test_move_xy(self, x: float, y: float):
        """Test MOVE command."""
        logger.info(f"Testing MOVE to X={x}, Y={y}...")
        try:
            self.socket.send(ExtendedCommand.MOVE)
            coords = struct.pack("!ff", x, y)
            self.socket.send(coords)
            time.sleep(0.5)  # Give time for movement
            logger.info("  Move command sent successfully")
        except Exception as e:
            logger.error(f"  MOVE failed: {e}")
            raise

    def test_move_z(self, z: float):
        """Test MOVEZ command."""
        logger.info(f"Testing MOVEZ to Z={z}...")
        try:
            self.socket.send(ExtendedCommand.MOVEZ)
            z_data = struct.pack("!f", z)
            self.socket.send(z_data)
            time.sleep(0.5)  # Give time for movement
            logger.info("  Move Z command sent successfully")
        except Exception as e:
            logger.error(f"  MOVEZ failed: {e}")
            raise

    def test_move_rotation(self, angle: float):
        """Test MOVER command."""
        logger.info(f"Testing MOVER to {angle}°...")
        try:
            self.socket.send(ExtendedCommand.MOVER)
            angle_data = struct.pack("!f", angle)
            self.socket.send(angle_data)
            time.sleep(0.5)  # Give time for rotation
            logger.info("  Rotation command sent successfully")
        except Exception as e:
            logger.error(f"  MOVER failed: {e}")
            raise

    def test_status(self) -> str:
        """Test STATUS command."""
        logger.info("Testing STATUS...")
        try:
            self.socket.send(ExtendedCommand.STATUS)
            response = self.socket.recv(16)
            status = response.decode().strip()
            logger.info(f"  Acquisition status: {status}")
            return status
        except Exception as e:
            logger.error(f"  STATUS failed: {e}")
            raise

    def test_progress(self) -> Tuple[int, int]:
        """Test PROGRESS command."""
        logger.info("Testing PROGRESS...")
        try:
            self.socket.send(ExtendedCommand.PROGRESS)
            response = self.socket.recv(8)
            current, total = struct.unpack("!II", response)
            logger.info(f"  Acquisition progress: {current}/{total}")
            return current, total
        except Exception as e:
            logger.error(f"  PROGRESS failed: {e}")
            raise

    def test_cancel(self):
        """Test CANCEL command."""
        logger.info("Testing CANCEL...")
        try:
            self.socket.send(ExtendedCommand.CANCEL)
            response = self.socket.recv(3)  # Expecting 'ACK'
            if response == b"ACK":
                logger.info("  Cancel acknowledged")
            else:
                logger.warning(f"  Unexpected cancel response: {response}")
        except Exception as e:
            logger.error(f"  CANCEL failed: {e}")
            raise

    def test_acquisition(
        self,
        yaml_path: str,
        projects_path: str,
        sample_label: str,
        scan_type: str,
        region_name: str,
        angles: str = "(0,7,-7,90)",
        monitor: bool = True,
    ):
        """Test ACQUIRE command with monitoring."""
        logger.info("Testing ACQUIRE...")
        logger.info(f"  YAML: {yaml_path}")
        logger.info(f"  Projects: {projects_path}")
        logger.info(f"  Sample: {sample_label}")
        logger.info(f"  Scan type: {scan_type}")
        logger.info(f"  Region: {region_name}")
        logger.info(f"  Angles: {angles}")

        try:
            # Send ACQUIRE command
            self.socket.send(ExtendedCommand.ACQUIRE)

            # Build the message
            message = (
                f"--yaml {yaml_path} "
                f"--projects {projects_path} "
                f"--sample {sample_label} "
                f"--scan-type {scan_type} "
                f"--region {region_name} "
                f"--angles {angles} "
                f"{END_MARKER}"
            )

            # Send the message
            self.socket.send(message.encode())
            logger.info("  Acquisition started")

            if monitor:
                # Monitor the acquisition
                self._monitor_acquisition()

        except Exception as e:
            logger.error(f"  ACQUIRE failed: {e}")
            raise

    def test_background_acquisition(
        self,
        yaml_path: str,
        output_path: str,
        modality: str,
        angles: str = "(0,7,-7,90)",
        exposures: Optional[str] = None,
    ):
        """Test BGACQUIRE command."""
        logger.info("Testing BGACQUIRE...")
        logger.info(f"  YAML: {yaml_path}")
        logger.info(f"  Output: {output_path}")
        logger.info(f"  Modality: {modality}")
        logger.info(f"  Angles: {angles}")
        if exposures:
            logger.info(f"  Exposures: {exposures}")

        try:
            # Send BGACQUIRE command
            self.socket.send(ExtendedCommand.BGACQUIRE)

            # Build the message
            message = (
                f"--yaml {yaml_path} "
                f"--output {output_path} "
                f"--modality {modality} "
                f"--angles {angles} "
            )
            if exposures:
                message += f"--exposures {exposures} "
            message += END_MARKER

            # Send the message
            self.socket.send(message.encode())

            # Wait for response
            response_parts = []
            while True:
                chunk = self.socket.recv(1024)
                if not chunk:
                    break
                response_parts.append(chunk.decode())
                response = "".join(response_parts)
                if response.startswith("SUCCESS:") or response.startswith("FAILED:"):
                    break

            response = "".join(response_parts)
            if response.startswith("SUCCESS:"):
                output_dir = response.replace("SUCCESS:", "")
                logger.info(f"  Background acquisition successful: {output_dir}")
            else:
                logger.error(f"  Background acquisition failed: {response}")

        except Exception as e:
            logger.error(f"  BGACQUIRE failed: {e}")
            raise

    def test_snap(
        self,
        angle: float,
        exposure_ms: float,
        output_path: str,
        debayer: str = "auto",
    ) -> Optional[str]:
        """
        Test SNAP command - simple fixed-exposure acquisition.

        Args:
            angle: Rotation angle in degrees
            exposure_ms: Fixed exposure time in milliseconds
            output_path: Full path for output image file
            debayer: Debayering mode - "auto" (default), "true", or "false"
                     Auto mode detects camera type (MicroPublisher6 needs debayer,
                     JAI prism camera does not)

        Returns:
            Output path on success, None on failure
        """
        logger.info("Testing SNAP (fixed exposure)...")
        logger.info(f"  Angle: {angle} deg")
        logger.info(f"  Exposure: {exposure_ms} ms")
        logger.info(f"  Output: {output_path}")
        logger.info(f"  Debayer mode: {debayer}")

        try:
            # Send SNAP command (8 bytes)
            self.socket.sendall(ExtendedCommand.SNAP)

            # Build the message
            message = (
                f"--angle {angle} "
                f"--exposure {exposure_ms} "
                f"--output {output_path} "
                f"--debayer {debayer} "
            )
            message += END_MARKER

            # Send the message - use sendall to ensure complete transmission
            self.socket.sendall(message.encode())

            # Wait for response (no STARTED acknowledgment for simple snap)
            # Timeout needs to account for exposure time + image processing + save
            timeout = max(60.0, exposure_ms / 1000.0 * 2 + 30)  # At least 60s, or 2x exposure + 30s
            response_parts = []
            self.socket.settimeout(timeout)
            logger.debug(f"  Waiting for SNAP response (timeout: {timeout:.1f}s)")
            try:
                while True:
                    chunk = self.socket.recv(1024)
                    if not chunk:
                        logger.warning("  Connection closed while waiting for SNAP response")
                        break
                    response_parts.append(chunk.decode())
                    response = "".join(response_parts)
                    if response.startswith("SUCCESS:") or response.startswith("FAILED:"):
                        break
            finally:
                self.socket.settimeout(None)

            response = "".join(response_parts)
            if response.startswith("SUCCESS:"):
                result_path = response.replace("SUCCESS:", "")
                logger.info(f"  SNAP successful: {result_path}")
                return result_path
            else:
                logger.error(f"  SNAP failed: {response}")
                return None

        except Exception as e:
            logger.error(f"  SNAP failed: {e}")
            raise

    def _monitor_acquisition(self, interval: float = 1.0):
        """Monitor acquisition progress until complete."""
        logger.info("  Monitoring acquisition progress...")

        while True:
            time.sleep(interval)

            # Get status
            status = self.test_status()

            # Get progress
            current, total = self.test_progress()

            # Log progress
            if total > 0:
                percent = (current / total) * 100
                logger.info(f"    Status: {status}, Progress: {current}/{total} ({percent:.1f}%)")
            else:
                logger.info(f"    Status: {status}")

            # Check if complete
            if status in ["COMPLETED", "FAILED", "CANCELLED", "IDLE"]:
                logger.info(f"  Acquisition finished with status: {status}")
                break

    def run_all_tests(self):
        """Run all basic tests (excluding acquisition)."""
        logger.info("=" * 60)
        logger.info("Running all basic server tests...")
        logger.info("=" * 60)

        try:
            # Position queries
            x, y = self.test_get_xy()
            z = self.test_get_z()
            fov_x, fov_y = self.test_get_fov()

            # Rotation (if supported)
            try:
                angle = self.test_get_rotation()
                has_rotation = True
            except:
                logger.info("Rotation not supported on this system")
                has_rotation = False

            # Small movements to test
            logger.info("\nTesting movement commands...")

            # Move slightly and return
            self.test_move_xy(x + 10, y + 10)
            time.sleep(1)
            self.test_move_xy(x, y)  # Move back

            self.test_move_z(z + 5)
            time.sleep(1)
            self.test_move_z(z)  # Move back

            if has_rotation:
                self.test_move_rotation(angle + 5)
                time.sleep(1)
                self.test_move_rotation(angle)  # Rotate back

            # Status commands
            logger.info("\nTesting status commands...")
            self.test_status()
            self.test_progress()
            self.test_cancel()

            logger.info("\n" + "=" * 60)
            logger.info("All basic tests completed successfully!")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"\nTest failed: {e}")
            raise


def main():
    """Main function to run tests."""
    parser = argparse.ArgumentParser(description="Test client for QuPath microscope server")
    parser.add_argument("--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    parser.add_argument(
        "--port", type=int, default=TCP_PORT, help=f"Server port (default: {TCP_PORT})"
    )
    parser.add_argument(
        "--test",
        choices=["all", "position", "movement", "status", "acquire", "background"],
        default="all",
        help="Which test to run",
    )

    # Acquisition-specific arguments
    parser.add_argument("--yaml", help="YAML config path for acquisition test")
    parser.add_argument("--projects", help="Projects folder path for acquisition test")
    parser.add_argument("--sample", help="Sample label for acquisition test")
    parser.add_argument("--scan-type", help="Scan type for acquisition test")
    parser.add_argument("--region", help="Region name for acquisition test")
    parser.add_argument("--angles", default="(0,7,-7,90)", help="Angles for acquisition")
    parser.add_argument("--modality", help="Modality for background acquisition")
    parser.add_argument("--output", help="Output path for background acquisition")

    args = parser.parse_args()

    # Create test client
    client = QuPathTestClient(args.host, args.port)

    try:
        # Connect to server
        if not client.connect():
            sys.exit(1)

        # Run requested tests
        if args.test == "all":
            client.run_all_tests()

        elif args.test == "position":
            logger.info("Running position query tests...")
            client.test_get_xy()
            client.test_get_z()
            client.test_get_fov()
            try:
                client.test_get_rotation()
            except:
                logger.info("Rotation not supported")

        elif args.test == "movement":
            logger.info("Running movement tests...")
            x, y = client.test_get_xy()
            z = client.test_get_z()

            # Test small movements
            client.test_move_xy(x + 100, y + 100)
            time.sleep(2)
            client.test_move_xy(x, y)

            client.test_move_z(z + 10)
            time.sleep(2)
            client.test_move_z(z)

            try:
                angle = client.test_get_rotation()
                client.test_move_rotation(angle + 10)
                time.sleep(2)
                client.test_move_rotation(angle)
            except:
                logger.info("Rotation movement not tested")

        elif args.test == "status":
            logger.info("Running status tests...")
            client.test_status()
            client.test_progress()
            client.test_cancel()

        elif args.test == "acquire":
            if not all([args.yaml, args.projects, args.sample, args.scan_type, args.region]):
                logger.error(
                    "Acquisition test requires: --yaml, --projects, --sample, --scan-type, --region"
                )
                sys.exit(1)

            client.test_acquisition(
                yaml_path=args.yaml,
                projects_path=args.projects,
                sample_label=args.sample,
                scan_type=args.scan_type,
                region_name=args.region,
                angles=args.angles,
                monitor=True,
            )

        elif args.test == "background":
            if not all([args.yaml, args.output, args.modality]):
                logger.error("Background acquisition test requires: --yaml, --output, --modality")
                sys.exit(1)

            client.test_background_acquisition(
                yaml_path=args.yaml,
                output_path=args.output,
                modality=args.modality,
                angles=args.angles,
            )

    finally:
        # Always disconnect
        client.disconnect()


if __name__ == "__main__":
    main()
