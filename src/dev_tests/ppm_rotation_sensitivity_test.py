#!/usr/bin/env python3
"""
PPM Rotation Sensitivity Testing Suite

This module integrates with the existing qp_server infrastructure to systematically
test PPM rotation sensitivity by acquiring images at precise angles and analyzing
the impact of angular deviations on image quality and birefringence calculations.

This script leverages existing utilities from:
- qp_utils.py: PolarizerCalibrationUtils, BackgroundCorrectionUtils
- test_client.py: QuPathTestClient for server communication
- qp_server.py: Server command protocol
"""

import sys
import os
import socket
import struct
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from existing project infrastructure
from smart_wsi_scanner.tests.test_client import QuPathTestClient
from smart_wsi_scanner.qp_utils import (
    PolarizerCalibrationUtils,
    BackgroundCorrectionUtils,
)
from smart_wsi_scanner.qp_server_config import ExtendedCommand
from smart_wsi_scanner.config import ConfigManager

# Import the analysis module from same directory (dev_tests)
sys.path.insert(0, str(Path(__file__).parent))
try:
    from ppm_rotation_sensitivity_analysis import PPMRotationAnalyzer
    ANALYZER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: PPM rotation analyzer not found ({e}). Analysis will be skipped.")
    ANALYZER_AVAILABLE = False


class PPMRotationSensitivityTester:
    """
    Comprehensive PPM rotation sensitivity testing using existing infrastructure.

    This class coordinates systematic image acquisition at precise angles and
    analyzes the impact on image quality and birefringence calculations.
    """

    # Background correction exposures from calibration
    # These are the exposures that achieve proper intensity at each angle
    BACKGROUND_EXPOSURES_MS = {
        -7.0: 21.1,
        0.0: 92.16,
        7.0: 22.63,
        90.0: 0.57,
    }

    def __init__(self,
                 config_yaml: str,
                 output_dir: str = None,
                 host: str = "127.0.0.1",
                 port: int = 5000):
        """
        Initialize the tester with configuration and connection parameters.

        Args:
            config_yaml: Path to microscope configuration YAML file
            output_dir: Output directory for test results (default: configurations/ppm_sensitivity_tests/)
            host: qp_server host address
            port: qp_server port
        """
        self.config_yaml = Path(config_yaml)

        # Default output to configurations/ppm_sensitivity_tests/
        if output_dir is None:
            config_dir = Path(__file__).parent.parent / "smart_wsi_scanner" / "configurations"
            self.output_dir = config_dir / "ppm_sensitivity_tests" / f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize logging
        self.setup_logging()

        # Load configuration
        self.config_manager = ConfigManager()
        self.config = self.config_manager.load_config_file(str(self.config_yaml))

        # Initialize client connection
        self.client = QuPathTestClient(host=host, port=port)
        self.connected = False

        # Initialize utilities
        self.pol_utils = PolarizerCalibrationUtils()
        self.bg_utils = BackgroundCorrectionUtils()

        # Store test results
        self.test_results = {}
        self.acquired_images = {}

        # Standard PPM angles from config or defaults
        self.standard_angles = self.config.get('ppm', {}).get('angles',
            [0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 91])

        # Load imaging profile exposures if available
        self.imaging_profile_exposures = self._load_imaging_profile_exposures()

        self.logger.info(f"PPM Rotation Sensitivity Tester initialized")
        self.logger.info(f"Config: {self.config_yaml}")
        self.logger.info(f"Output: {self.output_dir}")

    def _load_imaging_profile_exposures(self) -> Dict[str, float]:
        """
        Load exposure times from imaging profile config.

        Returns:
            Dictionary mapping angle categories to exposure times (ms):
            {'negative': 800, 'crossed': 1200, 'positive': 800, 'uncrossed': 15}
        """
        exposures = {}
        try:
            # Try to load imageprocessing config
            config_dir = Path(__file__).parent.parent / "smart_wsi_scanner" / "configurations"
            imgproc_file = config_dir / "imageprocessing_PPM.yml"

            if imgproc_file.exists():
                imgproc = self.config_manager.load_config_file(str(imgproc_file))

                # Get current objective/detector from main config
                objective = self.config.get('objective', 'LOCI_OBJECTIVE_OLYMPUS_20X_POL_001')
                detector = self.config.get('detector', 'LOCI_DETECTOR_TELEDYNE_001')

                # Navigate to exposures
                profiles = imgproc.get('imaging_profiles', {}).get('ppm', {})
                obj_profile = profiles.get(objective, {})
                det_profile = obj_profile.get(detector, {})
                exp_config = det_profile.get('exposures_ms', {})

                # Extract exposures (handle both simple and per-channel formats)
                for category in ['negative', 'crossed', 'positive', 'uncrossed']:
                    val = exp_config.get(category)
                    if isinstance(val, dict):
                        exposures[category] = val.get('all', list(val.values())[0])
                    elif val is not None:
                        exposures[category] = float(val)

                self.logger.info(f"Loaded imaging profile exposures: {exposures}")
            else:
                self.logger.warning(f"Imaging profile not found at {imgproc_file}")

        except Exception as e:
            self.logger.warning(f"Could not load imaging profile exposures: {e}")

        return exposures

    def get_exposure_for_angle(self, angle: float) -> float:
        """
        Get the appropriate exposure time for an angle based on background calibration.

        Uses hardcoded background correction exposures and interpolates for
        angles between calibration points.

        Calibration points:
        - -7.0 deg: 21.1 ms (negative polarization)
        - 0.0 deg: 92.16 ms (crossed polars - darkest)
        - 7.0 deg: 22.63 ms (positive polarization)
        - 90.0 deg: 0.57 ms (uncrossed - brightest)

        Args:
            angle: Rotation angle in degrees

        Returns:
            Exposure time in ms
        """
        bg = self.BACKGROUND_EXPOSURES_MS

        # Check for exact match first
        if angle in bg:
            return bg[angle]

        # Map angle to nearest calibration category
        abs_angle = abs(angle)

        # Near 0 (crossed polars): use 0 deg exposure
        if abs_angle <= 3:
            return bg[0.0]

        # Near +/-7 (polarization angles): use 7 or -7 deg exposure
        elif 4 <= abs_angle <= 10:
            if angle >= 0:
                return bg[7.0]
            else:
                return bg[-7.0]

        # Near 90 (uncrossed): use 90 deg exposure
        elif 85 <= abs_angle <= 95:
            return bg[90.0]

        # For angles between 7-90, interpolate logarithmically
        # (exposures span ~2 orders of magnitude)
        elif 10 < abs_angle < 85:
            # Interpolate between 7 deg (22.63ms) and 90 deg (0.57ms)
            # Using log interpolation for large exposure ranges
            import math
            exp_7 = bg[7.0]
            exp_90 = bg[90.0]
            # Normalize angle to 0-1 range between 7 and 90
            t = (abs_angle - 7) / (90 - 7)
            # Log interpolation
            log_exp = math.log(exp_7) * (1 - t) + math.log(exp_90) * t
            return math.exp(log_exp)

        # Fallback for any edge cases
        else:
            return bg[7.0]  # Default to 7 deg exposure

    def setup_logging(self):
        """Setup comprehensive logging for the test session."""
        log_file = self.output_dir / "ppm_sensitivity_test.log"

        # Configure logger
        self.logger = logging.getLogger("PPMSensitivityTest")
        self.logger.setLevel(logging.DEBUG)

        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def connect(self) -> bool:
        """Connect to the qp_server."""
        try:
            self.client.connect()
            self.connected = True
            self.logger.info(f"Connected to server at {self.client.host}:{self.client.port}")

            # Test connection with status check
            status = self.client.test_status()
            self.logger.info(f"Server status: {status}")

            return True
        except Exception as e:
            self.logger.error(f"Failed to connect: {e}")
            self.connected = False
            return False

    def disconnect(self):
        """Disconnect from the server."""
        if self.connected:
            try:
                self.client.disconnect()
                self.connected = False
                self.logger.info("Disconnected from server")
            except Exception as e:
                self.logger.error(f"Error during disconnect: {e}")

    def _drain_socket(self):
        """Drain any leftover data from the socket buffer."""
        if not self.client.socket:
            return
        try:
            self.client.socket.setblocking(False)
            while True:
                try:
                    data = self.client.socket.recv(4096)
                    if not data:
                        break
                    self.logger.debug(f"Drained {len(data)} bytes from socket")
                except BlockingIOError:
                    break  # No more data
                except Exception:
                    break
        finally:
            self.client.socket.setblocking(True)
            self.client.socket.settimeout(10.0)

    def _reconnect(self):
        """Reconnect to the server to reset socket state."""
        self.logger.info("Reconnecting to reset socket state...")
        try:
            if self.client.socket:
                try:
                    self.client.socket.close()
                except:
                    pass
                self.client.socket = None
            self.connected = False
            time.sleep(0.5)
            return self.connect()
        except Exception as e:
            self.logger.error(f"Reconnection failed: {e}")
            return False

    def acquire_at_angle(self, angle: float, save_name: str,
                        exposure_ms: float = None) -> Optional[Path]:
        """
        Acquire a single image at specified rotation angle.

        Args:
            angle: Rotation angle in degrees
            save_name: Name for the saved image
            exposure_ms: Exposure time in milliseconds (uses config default if None)

        Returns:
            Path to acquired image or None if failed
        """
        if not self.connected:
            self.logger.error("Not connected to server")
            return None

        try:
            # Step 1: Move to angle and wait for completion
            self.logger.debug(f"Moving to angle {angle} degrees")
            self.client.test_move_rotation(angle)
            time.sleep(0.5)  # Allow settling

            # Step 2: Verify position (this is a complete request/response cycle)
            actual_angle = self.client.test_get_rotation()
            angle_error = abs(actual_angle - angle)
            self.logger.info(f"Set: {angle:.2f} deg, Read: {actual_angle:.2f} deg, Error: {angle_error:.3f} deg")

            if angle_error > 0.5:  # Warning threshold
                self.logger.warning(f"Large angle error: {angle_error:.3f} degrees")

            # Step 3: Build and send BGACQUIRE command
            output_path = self.output_dir / save_name

            message = (
                f"--yaml {self.config_yaml} "
                f"--output {output_path.parent} "
                f"--modality {self.config.get('modality', 'ppm_20x')} "
                f"--angles {angle} "
            )

            if exposure_ms:
                message += f"--exposures {exposure_ms} "

            message += "END_MARKER"

            self.logger.debug(f"Sending BGACQUIRE: {message[:80]}...")
            self.client.socket.sendall(ExtendedCommand.BGACQUIRE)
            self.client.socket.sendall(message.encode('utf-8'))

            # Step 4: Read the COMPLETE response - DO NOT send any other commands
            # The response will be either:
            #   - "STARTED:..." followed eventually by "SUCCESS:..." or "FAILED:..."
            #   - "FAILED:..." immediately
            response_data = b""
            self.client.socket.settimeout(120.0)  # Long timeout for acquisition

            # Read until we get a final SUCCESS or FAILED (after any STARTED)
            got_started = False
            while True:
                try:
                    chunk = self.client.socket.recv(4096)
                    if not chunk:
                        self.logger.error("Connection closed while waiting for response")
                        break

                    response_data += chunk
                    response = response_data.decode('utf-8', errors='replace')

                    # Check for STARTED
                    if 'STARTED:' in response:
                        if not got_started:
                            self.logger.debug("Acquisition started, waiting for completion...")
                            got_started = True

                    # Check for final response (SUCCESS or FAILED after STARTED, or immediate FAILED)
                    if got_started:
                        # Look for SUCCESS or FAILED after the STARTED message
                        after_started = response[response.index('STARTED:'):]
                        if 'SUCCESS:' in after_started:
                            self.logger.info(f"Acquisition complete: {save_name}")
                            return output_path
                        elif after_started.count('FAILED:') > 0 and 'FAILED:' in after_started[8:]:
                            # FAILED that appears after STARTED (not part of STARTED message)
                            self.logger.error(f"Acquisition failed: {response[-300:]}")
                            return None
                    else:
                        # No STARTED yet - check for immediate FAILED
                        if 'FAILED:' in response:
                            self.logger.error(f"Acquisition failed: {response[:300]}")
                            return None
                        # Or immediate SUCCESS (unlikely but possible)
                        if 'SUCCESS:' in response:
                            self.logger.info(f"Acquisition complete: {save_name}")
                            return output_path

                except socket.timeout:
                    self.logger.error("Timeout waiting for acquisition response")
                    # Check if file exists as last resort
                    if output_path.exists():
                        self.logger.info(f"File exists despite timeout: {save_name}")
                        return output_path
                    return None

            return None

        except Exception as e:
            self.logger.error(f"Error acquiring at angle {angle}: {e}")
            return None

    def run_standard_angles_test(self) -> Dict[float, Path]:
        """
        Acquire images at all standard PPM angles.

        Returns:
            Dictionary mapping angles to image paths
        """
        self.logger.info("=" * 60)
        self.logger.info("STANDARD ANGLES TEST")
        self.logger.info("=" * 60)

        acquired = {}

        for i, angle in enumerate(self.standard_angles):
            self.logger.info(f"[{i+1}/{len(self.standard_angles)}] Acquiring at {angle} degrees")

            save_name = f"standard_{angle:05.1f}deg.tif"
            image_path = self.acquire_at_angle(angle, save_name)

            if image_path:
                acquired[angle] = image_path
            else:
                self.logger.warning(f"Failed to acquire at {angle} degrees")

        self.logger.info(f"Acquired {len(acquired)}/{len(self.standard_angles)} standard angles")
        self.test_results['standard_angles'] = acquired
        return acquired

    def run_fine_deviation_test(self, base_angle: float = 7.0,
                               deviations: List[float] = None,
                               fixed_exposure_ms: float = None,
                               acquire_zero_reference: bool = True) -> Dict[float, Path]:
        """
        Test fine angular deviations around a base angle.

        All images in a deviation cluster use the SAME exposure (from background
        calibration) to ensure valid sensitivity comparisons.

        Exposure selection:
        1. If fixed_exposure_ms is provided, use that
        2. Otherwise, use background calibration exposure for base angle

        Args:
            base_angle: Center angle for testing
            deviations: List of deviations to test (positive and negative)
            fixed_exposure_ms: Override exposure time for ALL images in cluster.
                              If None, uses background calibration exposure.
            acquire_zero_reference: If True, also acquire 0 deg image with
                                   the cluster's exposure for comparison.

        Returns:
            Dictionary mapping angles to image paths
        """
        if deviations is None:
            deviations = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.7, 1.0]

        self.logger.info("=" * 60)
        self.logger.info(f"FINE DEVIATION TEST: {base_angle} degrees")
        self.logger.info("=" * 60)
        self.logger.info(f"Testing deviations: {deviations}")

        # Determine exposure for this cluster from background calibration
        cluster_exposure = fixed_exposure_ms
        if cluster_exposure is None:
            cluster_exposure = self.get_exposure_for_angle(base_angle)
            self.logger.info(f"Using BACKGROUND CALIBRATION exposure for {base_angle} deg: "
                           f"{cluster_exposure:.2f} ms")

        self.logger.info(f"FIXED exposure for entire cluster: {cluster_exposure:.2f} ms")
        self.logger.info("All images in this cluster will have identical exposure")

        acquired = {}

        # First, acquire 0 degree reference with THIS cluster's exposure
        # This allows valid comparison to crossed polars baseline
        if acquire_zero_reference and base_angle != 0:
            self.logger.info("-" * 40)
            self.logger.info(f"Acquiring 0 deg reference with cluster exposure ({cluster_exposure:.2f} ms)")
            save_name = f"ref_0deg_at_{base_angle}deg_exposure.tif"
            image_path = self.acquire_at_angle(0.0, save_name, exposure_ms=cluster_exposure)
            if image_path:
                # Store with special key to indicate it's a reference
                acquired[f"0.0_ref_{base_angle}"] = image_path
                self.logger.info(f"Acquired 0 deg reference for {base_angle} deg cluster comparison")
            self.logger.info("-" * 40)

        # Acquire deviation images
        for i, dev in enumerate(deviations):
            # Test positive deviation
            angle_pos = base_angle + dev
            if 0 <= angle_pos <= 180:  # Check hardware limits
                self.logger.info(f"[{i*2+1}/{len(deviations)*2}] Testing {angle_pos:.2f} deg (+{dev})")
                save_name = f"{angle_pos:.2f}.tif"
                image_path = self.acquire_at_angle(angle_pos, save_name,
                                                   exposure_ms=cluster_exposure)
                if image_path:
                    acquired[angle_pos] = image_path

            # Test negative deviation (skip 0)
            if dev > 0:
                angle_neg = base_angle - dev
                if 0 <= angle_neg <= 180:
                    self.logger.info(f"[{i*2+2}/{len(deviations)*2}] Testing {angle_neg:.2f} deg (-{dev})")
                    save_name = f"{angle_neg:.2f}.tif"
                    image_path = self.acquire_at_angle(angle_neg, save_name,
                                                       exposure_ms=cluster_exposure)
                    if image_path:
                        acquired[angle_neg] = image_path

        self.logger.info(f"Acquired {len(acquired)} images (including reference)")
        self.logger.info(f"Cluster exposure used: {cluster_exposure:.2f} ms")
        self.test_results[f'deviation_{base_angle}deg'] = acquired
        return acquired

    def run_repeatability_test(self, test_angle: float = 7.0,
                              n_repeats: int = 10,
                              include_large_movements: bool = True) -> List[Dict]:
        """
        Test mechanical repeatability with both small and large movements.

        Tests both small angular movements (e.g., 0 -> 7 deg) and large movements
        (e.g., 0 -> 90 deg, 90 -> 7 deg) to characterize positioning accuracy
        across the full range of operation.

        Args:
            test_angle: Primary angle to test repeatedly
            n_repeats: Number of repetitions for each movement type
            include_large_movements: If True, also test 90+ degree movements

        Returns:
            List of measurement dictionaries
        """
        self.logger.info("=" * 60)
        self.logger.info(f"REPEATABILITY TEST: {test_angle} degrees x {n_repeats}")
        if include_large_movements:
            self.logger.info("Including large movement tests (90+ degrees)")
        self.logger.info("=" * 60)

        all_measurements = []

        # ========== SMALL MOVEMENT TEST (0 -> test_angle) ==========
        self.logger.info("-" * 40)
        self.logger.info(f"SMALL MOVEMENTS: 0 -> {test_angle} deg ({test_angle} deg travel)")
        self.logger.info("-" * 40)

        for i in range(n_repeats):
            self.logger.info(f"Small movement {i+1}/{n_repeats}")

            # Move away first to test return accuracy
            if i > 0:
                self.client.test_move_rotation(0)
                time.sleep(1)

            # Move to test angle
            self.client.test_move_rotation(test_angle)
            time.sleep(0.5)

            # Read actual position
            actual = self.client.test_get_rotation()
            error = actual - test_angle

            measurement = {
                'repetition': i + 1,
                'test_type': 'small_movement',
                'from_angle': 0,
                'set_angle': test_angle,
                'travel_deg': test_angle,
                'read_angle': actual,
                'error': error,
                'timestamp': datetime.now().isoformat()
            }
            all_measurements.append(measurement)

            self.logger.info(f"  Set: {test_angle:.3f}, Read: {actual:.3f}, Error: {error:.4f}")

        # ========== LARGE MOVEMENT TESTS ==========
        if include_large_movements:
            # Test 1: 0 -> 90 degrees (large movement to uncrossed)
            self.logger.info("-" * 40)
            self.logger.info("LARGE MOVEMENTS: 0 -> 90 deg (90 deg travel)")
            self.logger.info("-" * 40)

            for i in range(n_repeats):
                self.logger.info(f"Large movement (0->90) {i+1}/{n_repeats}")

                # Return to 0
                self.client.test_move_rotation(0)
                time.sleep(1)

                # Move to 90 degrees
                self.client.test_move_rotation(90)
                time.sleep(0.5)

                actual = self.client.test_get_rotation()
                error = actual - 90

                measurement = {
                    'repetition': i + 1,
                    'test_type': 'large_movement_0_to_90',
                    'from_angle': 0,
                    'set_angle': 90,
                    'travel_deg': 90,
                    'read_angle': actual,
                    'error': error,
                    'timestamp': datetime.now().isoformat()
                }
                all_measurements.append(measurement)

                self.logger.info(f"  Set: 90.000, Read: {actual:.3f}, Error: {error:.4f}")

            # Test 2: 90 -> test_angle (large movement back)
            self.logger.info("-" * 40)
            self.logger.info(f"LARGE MOVEMENTS: 90 -> {test_angle} deg ({90 - test_angle} deg travel)")
            self.logger.info("-" * 40)

            for i in range(n_repeats):
                self.logger.info(f"Large movement (90->{test_angle}) {i+1}/{n_repeats}")

                # Move to 90 first
                self.client.test_move_rotation(90)
                time.sleep(1)

                # Move to test angle
                self.client.test_move_rotation(test_angle)
                time.sleep(0.5)

                actual = self.client.test_get_rotation()
                error = actual - test_angle

                measurement = {
                    'repetition': i + 1,
                    'test_type': 'large_movement_90_to_target',
                    'from_angle': 90,
                    'set_angle': test_angle,
                    'travel_deg': 90 - test_angle,
                    'read_angle': actual,
                    'error': error,
                    'timestamp': datetime.now().isoformat()
                }
                all_measurements.append(measurement)

                self.logger.info(f"  Set: {test_angle:.3f}, Read: {actual:.3f}, Error: {error:.4f}")

            # Test 3: Full range 0 -> 90 -> 0 (bidirectional)
            self.logger.info("-" * 40)
            self.logger.info("BIDIRECTIONAL: 0 -> 90 -> 0 (hysteresis test)")
            self.logger.info("-" * 40)

            for i in range(n_repeats):
                self.logger.info(f"Bidirectional {i+1}/{n_repeats}")

                # Start at 0
                self.client.test_move_rotation(0)
                time.sleep(0.5)
                start_pos = self.client.test_get_rotation()

                # Move to 90
                self.client.test_move_rotation(90)
                time.sleep(0.5)
                mid_pos = self.client.test_get_rotation()

                # Return to 0
                self.client.test_move_rotation(0)
                time.sleep(0.5)
                end_pos = self.client.test_get_rotation()

                hysteresis = end_pos - start_pos

                measurement = {
                    'repetition': i + 1,
                    'test_type': 'bidirectional_hysteresis',
                    'start_angle': start_pos,
                    'mid_angle': mid_pos,
                    'end_angle': end_pos,
                    'travel_deg': 180,  # Total travel: 0->90->0
                    'hysteresis': hysteresis,
                    'timestamp': datetime.now().isoformat()
                }
                all_measurements.append(measurement)

                self.logger.info(f"  Start: {start_pos:.4f}, Mid: {mid_pos:.3f}, End: {end_pos:.4f}")
                self.logger.info(f"  Hysteresis: {hysteresis:.4f} deg")

        # ========== CALCULATE STATISTICS BY TEST TYPE ==========
        self.logger.info("=" * 60)
        self.logger.info("REPEATABILITY STATISTICS BY TEST TYPE")
        self.logger.info("=" * 60)

        stats_by_type = {}

        # Group measurements by test type
        test_types = set(m.get('test_type', 'unknown') for m in all_measurements)

        for test_type in test_types:
            type_measurements = [m for m in all_measurements if m.get('test_type') == test_type]

            if test_type == 'bidirectional_hysteresis':
                # Special handling for hysteresis test
                hysteresis_values = [m['hysteresis'] for m in type_measurements]
                stats_by_type[test_type] = {
                    'n_measurements': len(type_measurements),
                    'mean_hysteresis': float(np.mean(hysteresis_values)),
                    'max_hysteresis': float(np.max(np.abs(hysteresis_values))),
                    'std_hysteresis': float(np.std(hysteresis_values))
                }
                self.logger.info(f"\n{test_type}:")
                self.logger.info(f"  N measurements: {stats_by_type[test_type]['n_measurements']}")
                self.logger.info(f"  Mean hysteresis: {stats_by_type[test_type]['mean_hysteresis']:.4f} deg")
                self.logger.info(f"  Max hysteresis: {stats_by_type[test_type]['max_hysteresis']:.4f} deg")
            else:
                # Standard error statistics
                errors = [m['error'] for m in type_measurements]
                travel = type_measurements[0].get('travel_deg', 0) if type_measurements else 0

                stats_by_type[test_type] = {
                    'n_measurements': len(type_measurements),
                    'travel_deg': travel,
                    'mean_error': float(np.mean(errors)),
                    'std_error': float(np.std(errors)),
                    'max_error': float(np.max(np.abs(errors))),
                    'repeatability_2sigma': float(2 * np.std(errors))
                }
                self.logger.info(f"\n{test_type} ({travel} deg travel):")
                self.logger.info(f"  N measurements: {stats_by_type[test_type]['n_measurements']}")
                self.logger.info(f"  Mean error: {stats_by_type[test_type]['mean_error']:.4f} deg")
                self.logger.info(f"  Std deviation: {stats_by_type[test_type]['std_error']:.4f} deg")
                self.logger.info(f"  Max error: {stats_by_type[test_type]['max_error']:.4f} deg")
                self.logger.info(f"  2-sigma repeatability: {stats_by_type[test_type]['repeatability_2sigma']:.4f} deg")

        # Overall statistics (all non-hysteresis measurements)
        all_errors = [m['error'] for m in all_measurements if 'error' in m]
        overall_stats = {
            'total_measurements': len(all_measurements),
            'mean_error': float(np.mean(all_errors)) if all_errors else 0,
            'std_error': float(np.std(all_errors)) if all_errors else 0,
            'max_error': float(np.max(np.abs(all_errors))) if all_errors else 0,
            'repeatability_2sigma': float(2 * np.std(all_errors)) if all_errors else 0
        }

        self.logger.info("\n" + "=" * 40)
        self.logger.info("OVERALL REPEATABILITY (all movement types):")
        self.logger.info(f"  Total measurements: {overall_stats['total_measurements']}")
        self.logger.info(f"  Mean error: {overall_stats['mean_error']:.4f} deg")
        self.logger.info(f"  Max error: {overall_stats['max_error']:.4f} deg")
        self.logger.info(f"  2-sigma repeatability: {overall_stats['repeatability_2sigma']:.4f} deg")

        self.test_results['repeatability'] = {
            'measurements': all_measurements,
            'statistics_by_type': stats_by_type,
            'overall_statistics': overall_stats
        }

        return all_measurements

    def run_polarizer_calibration_comparison(self) -> Dict:
        """
        Run polarizer calibration to find crossed positions and test sensitivity.

        Returns:
            Calibration results dictionary
        """
        self.logger.info("=" * 60)
        self.logger.info("POLARIZER CALIBRATION COMPARISON")
        self.logger.info("=" * 60)

        # Use existing polarizer calibration utility
        output_folder = self.output_dir / "polarizer_calibration"
        output_folder.mkdir(exist_ok=True)

        # NOTE: calibrate_hardware_offset_two_stage requires direct hardware access
        # The actual method signature is:
        #   calibrate_hardware_offset_two_stage(hardware, coarse_range_deg, coarse_step_deg,
        #                                       fine_range_deg, fine_step_deg, exposure_ms,
        #                                       channel, logger_instance)
        # This test currently connects via socket to qp_server and doesn't have direct
        # hardware access. To use this calibration, either:
        #   1. Run calibration through qp_server command protocol
        #   2. Or refactor to get hardware object from the server connection

        self.logger.warning(
            "Polarizer calibration skipped - requires direct hardware access. "
            "Use qp_server commands for calibration instead."
        )

        # Placeholder results
        results = {
            'status': 'skipped',
            'reason': 'Direct hardware access required - not available via socket connection',
            'suggestion': 'Run calibration through qp_server CALIBRATE command'
        }

        self.test_results['polarizer_calibration'] = results
        return results

    def analyze_results(self) -> Dict:
        """
        Analyze all acquired images using the PPMRotationAnalyzer.

        Returns:
            Analysis results dictionary
        """
        if not ANALYZER_AVAILABLE:
            self.logger.warning("Analyzer not available, skipping analysis")
            return {}

        self.logger.info("=" * 60)
        self.logger.info("ANALYZING RESULTS")
        self.logger.info("=" * 60)

        try:
            # Initialize analyzer
            analyzer = PPMRotationAnalyzer(
                base_path=self.output_dir,
                output_dir=self.output_dir / "analysis"
            )

            # Try loading standard images first, then deviation images
            images = analyzer.load_images()

            if len(images) < 3:
                self.logger.info("No standard angle images found, trying deviation images...")
                images = analyzer.load_deviation_images()

            if len(images) < 3:
                self.logger.warning("Insufficient images for analysis (need at least 3)")
                return {}

            # Determine reference angle from loaded images
            # Use the center angle (e.g., 45.0 for deviation test, 7.0 for standard test)
            available_angles = sorted(images.keys())
            center_idx = len(available_angles) // 2
            reference_angle = available_angles[center_idx]
            self.logger.info(f"Using reference angle: {reference_angle:.2f} deg")

            # Run fine sensitivity analysis - this is the key analysis!
            self.logger.info("=" * 60)
            self.logger.info("FINE ANGULAR SENSITIVITY ANALYSIS")
            self.logger.info("=" * 60)

            # Analyze sensitivity around key angles (7 and 45 degrees)
            fine_sensitivity = analyzer.analyze_fine_sensitivity(base_angles=[7, 45])

            # Also compute adjacent differences for the full picture
            self.logger.info("\n" + "=" * 60)
            self.logger.info("ADJACENT ANGLE COMPARISONS")
            self.logger.info("=" * 60)
            df_adjacent = analyzer.compute_adjacent_differences()

            # Show fine-grained adjacent comparisons (delta < 1 degree)
            if not df_adjacent.empty:
                fine_steps = df_adjacent[df_adjacent['delta_deg'] < 1.0]
                if not fine_steps.empty:
                    self.logger.info("\nFine steps (< 1 degree apart):")
                    for _, row in fine_steps.iterrows():
                        self.logger.info(
                            f"  {row['angle1']:.2f} -> {row['angle2']:.2f} "
                            f"(delta={row['delta_deg']:.2f}): "
                            f"MAE={row['mae']:.2f}, {row['pct_change']:.3f}% change, "
                            f"intensity {row['median_intensity_1']:.0f} -> {row['median_intensity_2']:.0f}"
                        )

            # Compute standard differences for comparison (optional)
            self.logger.info("\nComputing standard reference differences...")
            df_differences = analyzer.compute_image_differences(reference_angle=reference_angle)

            # Skip birefringence analysis for deviation test (needs specific angle pairs)
            import pandas as pd
            df_birefringence = pd.DataFrame()

            # Save detailed results to CSV
            if not df_adjacent.empty:
                csv_path = self.output_dir / "adjacent_differences.csv"
                df_adjacent.to_csv(csv_path, index=False)
                self.logger.info(f"\nSaved adjacent differences to: {csv_path}")

            # Generate visualizations
            self.logger.info("Generating visualizations...")
            analyzer.visualize_difference_maps(reference_angle=reference_angle)

            # Generate report with all collected data
            self.logger.info("Generating analysis report...")
            report = analyzer.generate_report(
                df_differences=df_differences,
                df_birefringence=df_birefringence,
                df_adjacent=df_adjacent,
                fine_sensitivity=fine_sensitivity
            )

            self.test_results['analysis'] = report
            return report

        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            return {}

    def save_test_summary(self):
        """Save comprehensive test summary."""
        summary_file = self.output_dir / "test_summary.json"

        summary = {
            'test_date': datetime.now().isoformat(),
            'config_file': str(self.config_yaml),
            'output_directory': str(self.output_dir),
            'tests_performed': list(self.test_results.keys()),
            'results': {}
        }

        # Summarize each test
        for test_name, results in self.test_results.items():
            if test_name == 'standard_angles':
                summary['results'][test_name] = {
                    'angles_acquired': len(results),
                    'angles_list': sorted(results.keys())
                }
            elif test_name.startswith('deviation_'):
                if results:
                    # Filter to numeric keys only (exclude reference image string keys)
                    numeric_angles = [k for k in results.keys() if isinstance(k, (int, float))]
                    summary['results'][test_name] = {
                        'images_acquired': len(results),
                        'angle_range': [min(numeric_angles), max(numeric_angles)] if numeric_angles else [],
                        'reference_images': [k for k in results.keys() if isinstance(k, str)]
                    }
                else:
                    summary['results'][test_name] = {
                        'images_acquired': 0,
                        'angle_range': [],
                        'error': 'No images acquired - acquisition may have timed out'
                    }
            elif test_name == 'repeatability':
                # Include both overall and per-type statistics
                summary['results'][test_name] = {
                    'overall': results.get('overall_statistics', {}),
                    'by_type': results.get('statistics_by_type', {})
                }
            elif test_name == 'analysis':
                summary['results'][test_name] = results.get('summary', {})

        # Save JSON summary
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        # Save human-readable report
        report_file = self.output_dir / "test_report.txt"
        with open(report_file, 'w') as f:
            f.write("PPM ROTATION SENSITIVITY TEST REPORT\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Test Date: {summary['test_date']}\n")
            f.write(f"Config: {summary['config_file']}\n")
            f.write(f"Output: {summary['output_directory']}\n\n")

            f.write("TESTS PERFORMED:\n")
            f.write("-" * 40 + "\n")
            for test in summary['tests_performed']:
                f.write(f"  - {test}\n")

            if 'repeatability' in self.test_results:
                overall = self.test_results['repeatability'].get('overall_statistics', {})
                by_type = self.test_results['repeatability'].get('statistics_by_type', {})

                f.write(f"\nMECHANICAL REPEATABILITY:\n")
                f.write("=" * 50 + "\n")

                if overall:
                    f.write(f"\nOVERALL (all movement types combined):\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"  Total measurements: {overall.get('total_measurements', 0)}\n")
                    f.write(f"  Mean error: {overall.get('mean_error', 0):.4f} degrees\n")
                    f.write(f"  2-sigma repeatability: {overall.get('repeatability_2sigma', 0):.4f} degrees\n")
                    f.write(f"  Maximum error: {overall.get('max_error', 0):.4f} degrees\n")

                if by_type:
                    f.write(f"\nBY MOVEMENT TYPE:\n")
                    f.write("-" * 40 + "\n")
                    for test_type, stats in by_type.items():
                        if test_type == 'bidirectional_hysteresis':
                            f.write(f"\n  {test_type} (180 deg total travel):\n")
                            f.write(f"    Mean hysteresis: {stats.get('mean_hysteresis', 0):.4f} degrees\n")
                            f.write(f"    Max hysteresis: {stats.get('max_hysteresis', 0):.4f} degrees\n")
                        else:
                            travel = stats.get('travel_deg', 0)
                            f.write(f"\n  {test_type} ({travel} deg travel):\n")
                            f.write(f"    Mean error: {stats.get('mean_error', 0):.4f} degrees\n")
                            f.write(f"    2-sigma: {stats.get('repeatability_2sigma', 0):.4f} degrees\n")
                            f.write(f"    Max error: {stats.get('max_error', 0):.4f} degrees\n")

                if not overall and not by_type:
                    f.write(f"  Statistics not available\n")

            f.write(f"\nFull details in: {summary_file}\n")

        self.logger.info(f"Test summary saved to {summary_file}")
        self.logger.info(f"Test report saved to {report_file}")

    def run_comprehensive_test(self):
        """Run all tests in sequence."""
        self.logger.info("=" * 70)
        self.logger.info("PPM ROTATION SENSITIVITY - COMPREHENSIVE TEST")
        self.logger.info("=" * 70)

        if not self.connect():
            self.logger.error("Failed to connect to server. Aborting.")
            return

        try:
            # 1. Repeatability test first (quick, doesn't need images)
            self.run_repeatability_test(test_angle=7.0, n_repeats=10)

            # 2. Standard angles acquisition
            self.run_standard_angles_test()

            # 3. Fine deviation tests at critical angles
            for base_angle in [0, 7, 45]:
                self.run_fine_deviation_test(base_angle=base_angle)

            # 4. Optional: Polarizer calibration comparison
            # self.run_polarizer_calibration_comparison()

            # 5. Analyze all results
            self.analyze_results()

            # 6. Save comprehensive summary
            self.save_test_summary()

            self.logger.info("=" * 70)
            self.logger.info("COMPREHENSIVE TEST COMPLETE")
            self.logger.info("=" * 70)
            self.logger.info(f"All results saved to: {self.output_dir}")

        finally:
            self.disconnect()


def main():
    """Main entry point for the rotation sensitivity test."""
    import argparse

    parser = argparse.ArgumentParser(
        description='PPM Rotation Sensitivity Testing'
    )
    parser.add_argument('config_yaml',
                       help='Path to microscope configuration YAML file')
    parser.add_argument('--output', '-o',
                       help='Output directory for test results')
    parser.add_argument('--host', default='127.0.0.1',
                       help='qp_server host address')
    parser.add_argument('--port', type=int, default=5000,
                       help='qp_server port')
    parser.add_argument('--test', choices=['comprehensive', 'standard', 'deviation',
                                          'repeatability', 'calibration'],
                       default='comprehensive',
                       help='Test type to run')
    parser.add_argument('--base-angle', type=float, default=7.0,
                       help='Base angle for deviation testing')
    parser.add_argument('--repeats', type=int, default=10,
                       help='Number of repetitions for repeatability test')

    args = parser.parse_args()

    # Initialize tester
    tester = PPMRotationSensitivityTester(
        config_yaml=args.config_yaml,
        output_dir=args.output,
        host=args.host,
        port=args.port
    )

    # Run selected test
    if args.test == 'comprehensive':
        tester.run_comprehensive_test()
    else:
        if not tester.connect():
            print("Failed to connect to server")
            return

        try:
            if args.test == 'standard':
                tester.run_standard_angles_test()
            elif args.test == 'deviation':
                tester.run_fine_deviation_test(base_angle=args.base_angle)
            elif args.test == 'repeatability':
                tester.run_repeatability_test(test_angle=args.base_angle,
                                             n_repeats=args.repeats)
            elif args.test == 'calibration':
                tester.run_polarizer_calibration_comparison()

            # Always save summary
            tester.save_test_summary()

        finally:
            tester.disconnect()


if __name__ == "__main__":
    main()