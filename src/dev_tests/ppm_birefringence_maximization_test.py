#!/usr/bin/env python3
"""
PPM Birefringence Maximization Test

This script systematically acquires paired images at symmetric angles (+theta, -theta)
and computes their difference to measure birefringence signal strength. By scanning
from -10 to +10 degrees in fine steps, it identifies the optimal polarizer angle
for maximum birefringence contrast.

Key concepts:
- Birefringence signal = Image(+theta) - Image(-theta)
- At theta=0, both images are identical -> difference should be ~0 (sanity check)
- At optimal theta (typically ~7 deg for PPM), difference is maximized

Two exposure modes:
1. INTERPOLATE: Use calibration exposures from sensitivity test and interpolate
2. CALIBRATE: Run background acquisition at each angle on non-tissue area first,
   then pause for user to move to tissue before running actual test

Usage:
    python ppm_birefringence_maximization_test.py config.yml --mode interpolate
    python ppm_birefringence_maximization_test.py config.yml --mode calibrate
"""

import sys
import os
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging
import cv2

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from smart_wsi_scanner.tests.test_client import QuPathTestClient
from smart_wsi_scanner.qp_server_config import ExtendedCommand, TCP_PORT
from smart_wsi_scanner.config import ConfigManager

# Import matplotlib for visualization (optional)
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class PPMBirefringenceMaximizationTester:
    """
    Tests birefringence signal strength across a range of polarizer angles.

    Acquires paired images at +theta and -theta, computes their difference,
    and identifies the optimal angle for maximum birefringence contrast.
    """

    # Default calibration exposures (from sensitivity test at 4 key angles)
    # These are used for interpolation in INTERPOLATE mode
    CALIBRATION_EXPOSURES_MS = {
        -7.0: 21.1,
        0.0: 96.81,
        7.0: 22.63,
        90.0: 0.57,
    }

    def __init__(self,
                 config_yaml: str,
                 output_dir: str = None,
                 host: str = "127.0.0.1",
                 port: int = 5000,
                 angle_range: Tuple[float, float] = (-10.0, 10.0),
                 angle_step: float = 0.1,
                 exposure_mode: str = "interpolate",
                 keep_images: bool = True,
                 calibration_exposures: Dict[float, float] = None):
        """
        Initialize the birefringence maximization tester.

        Args:
            config_yaml: Path to microscope configuration YAML file
            output_dir: Output directory for results
            host: qp_server host address
            port: qp_server port
            angle_range: (min_angle, max_angle) in degrees (default: -10 to +10)
            angle_step: Step size in degrees (default: 0.1)
            exposure_mode: 'interpolate' or 'calibrate'
            keep_images: If True, keep .tif files after analysis
            calibration_exposures: Optional dict to override default exposures
        """
        self.config_yaml = Path(config_yaml)
        self.angle_range = angle_range
        self.angle_step = angle_step
        self.exposure_mode = exposure_mode
        self.keep_images = keep_images

        # Override calibration exposures if provided
        if calibration_exposures:
            self.CALIBRATION_EXPOSURES_MS.update(calibration_exposures)

        # Generate test angles (positive only - we'll acquire +/- pairs)
        self.test_angles = self._generate_test_angles()

        # Output directory
        if output_dir is None:
            config_dir = Path(__file__).parent.parent / "smart_wsi_scanner" / "configurations"
            self.output_dir = config_dir / "ppm_birefringence_tests" / f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.setup_logging()

        # Load configuration
        self.config_manager = ConfigManager()
        self.config = self.config_manager.load_config_file(str(self.config_yaml))

        # Initialize client
        self.client = QuPathTestClient(host=host, port=port)
        self.connected = False

        # Store results
        self.calibrated_exposures = {}  # angle -> exposure_ms (for calibrate mode)
        self.acquired_images = {}  # angle -> {'positive': path, 'negative': path}
        self.difference_images = {}  # angle -> path to difference image
        self.birefringence_metrics = {}  # angle -> metrics dict

        self.logger.info("PPM Birefringence Maximization Tester initialized")
        self.logger.info(f"  Angle range: {angle_range[0]} to {angle_range[1]} degrees")
        self.logger.info(f"  Step size: {angle_step} degrees")
        self.logger.info(f"  Test angles: {len(self.test_angles)} values")
        self.logger.info(f"  Exposure mode: {exposure_mode}")
        self.logger.info(f"  Output: {self.output_dir}")

    def _generate_test_angles(self) -> List[float]:
        """Generate list of test angles (positive values only, 0 to max)."""
        min_angle, max_angle = self.angle_range
        # We only need positive angles since we acquire +/- pairs
        # Include 0 as a sanity check
        max_abs = max(abs(min_angle), abs(max_angle))

        angles = []
        current = 0.0
        while current <= max_abs + 0.001:  # Small epsilon for float comparison
            angles.append(round(current, 2))
            current += self.angle_step

        return angles

    def setup_logging(self):
        """Setup logging for the test session."""
        log_file = self.output_dir / "birefringence_test.log"

        self.logger = logging.getLogger("PPMBirefringenceTest")
        self.logger.setLevel(logging.DEBUG)

        # Clear existing handlers
        self.logger.handlers = []

        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def connect(self) -> bool:
        """Connect to qp_server."""
        try:
            self.client.connect()
            self.connected = True
            self.logger.info(f"Connected to server at {self.client.host}:{self.client.port}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect: {e}")
            self.connected = False
            return False

    def disconnect(self):
        """Disconnect from server."""
        if self.connected:
            try:
                self.client.disconnect()
                self.connected = False
                self.logger.info("Disconnected from server")
            except Exception as e:
                self.logger.error(f"Error during disconnect: {e}")

    def get_exposure_for_angle(self, angle: float) -> float:
        """
        Get exposure time for a given angle.

        In CALIBRATE mode, uses calibrated exposures if available.
        In INTERPOLATE mode, interpolates from calibration points.

        Args:
            angle: Rotation angle in degrees

        Returns:
            Exposure time in milliseconds
        """
        # Check calibrated exposures first (from calibrate mode)
        if angle in self.calibrated_exposures:
            return self.calibrated_exposures[angle]

        # Check exact match in calibration table
        exp = self.CALIBRATION_EXPOSURES_MS
        if angle in exp:
            return exp[angle]

        # Interpolate based on angle magnitude
        abs_angle = abs(angle)

        # Near 0 (crossed polars): use 0 deg exposure
        if abs_angle <= 3:
            return exp[0.0]

        # Near +/-7 (polarization angles)
        elif 3 < abs_angle <= 10:
            if angle >= 0:
                return exp.get(7.0, exp.get(-7.0, 22.0))
            else:
                return exp.get(-7.0, exp.get(7.0, 22.0))

        # For angles beyond 10, extrapolate towards 90
        else:
            import math
            exp_7 = exp.get(7.0, 22.0)
            exp_90 = exp.get(90.0, 0.57)
            t = (abs_angle - 7) / (90 - 7)
            t = min(t, 1.0)  # Clamp to [0, 1]
            log_exp = math.log(exp_7) * (1 - t) + math.log(exp_90) * t
            return math.exp(log_exp)

    def acquire_at_angle(self, angle: float, save_name: str,
                        exposure_ms: float = None) -> Optional[Path]:
        """
        Acquire a single image at specified angle using SNAP command.

        Args:
            angle: Rotation angle in degrees
            save_name: Filename for saved image
            exposure_ms: Exposure time (uses lookup if not provided)

        Returns:
            Path to acquired image or None on failure
        """
        if not self.connected:
            self.logger.error("Not connected to server")
            return None

        if exposure_ms is None:
            exposure_ms = self.get_exposure_for_angle(angle)

        try:
            # Move to angle
            self.logger.debug(f"Moving to {angle:.2f} degrees")
            self.client.test_move_rotation(angle)
            time.sleep(0.3)

            # Verify position
            actual_angle = self.client.test_get_rotation()
            angle_error = abs(actual_angle - angle)
            if angle_error > 0.5:
                self.logger.warning(f"Large angle error: set={angle:.2f}, actual={actual_angle:.2f}")

            # Acquire image
            output_path = self.output_dir / save_name
            result = self.client.test_snap(
                angle=angle,
                exposure_ms=exposure_ms,
                output_path=str(output_path),
                debayer=True
            )

            if result:
                self.logger.debug(f"Acquired: {save_name} (exp={exposure_ms:.2f}ms)")
                return output_path
            else:
                self.logger.error(f"SNAP failed for {save_name}")
                return None

        except Exception as e:
            self.logger.error(f"Error acquiring at {angle}: {e}")
            return None

    def run_background_calibration(self) -> Dict[float, float]:
        """
        Run background calibration to determine optimal exposures at each angle.

        This is Phase 1 of CALIBRATE mode - acquire background images at non-tissue
        location to measure exposures needed for proper intensity.

        Returns:
            Dictionary mapping angle -> calibrated exposure (ms)
        """
        self.logger.info("=" * 70)
        self.logger.info("PHASE 1: BACKGROUND EXPOSURE CALIBRATION")
        self.logger.info("=" * 70)
        self.logger.info("Acquiring background images to calibrate exposures...")
        self.logger.info("Make sure the stage is positioned over a NON-TISSUE area!")
        self.logger.info("")

        # Create subdirectory for calibration images
        cal_dir = self.output_dir / "calibration"
        cal_dir.mkdir(exist_ok=True)

        calibrated = {}

        # For each unique angle (both + and -)
        all_angles = set()
        for angle in self.test_angles:
            all_angles.add(angle)
            all_angles.add(-angle)
        all_angles = sorted(all_angles)

        self.logger.info(f"Calibrating {len(all_angles)} angles...")

        for i, angle in enumerate(all_angles):
            self.logger.info(f"[{i+1}/{len(all_angles)}] Calibrating {angle:+.2f} deg...")

            # Start with interpolated exposure as initial guess
            initial_exp = self.get_exposure_for_angle(angle)

            # Move to angle
            self.client.test_move_rotation(angle)
            time.sleep(0.3)

            # Acquire with initial exposure
            save_name = f"cal_{angle:+.2f}.tif"
            output_path = cal_dir / save_name

            result = self.client.test_snap(
                angle=angle,
                exposure_ms=initial_exp,
                output_path=str(output_path),
                debayer=True
            )

            if result and output_path.exists():
                # Load and analyze image
                img = cv2.imread(str(output_path), cv2.IMREAD_UNCHANGED)
                if img is not None:
                    if len(img.shape) == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                    median_intensity = float(np.median(img))
                    max_intensity = float(np.max(img))

                    # Target: median around 30-50% of max range
                    # For 16-bit: target ~20000-30000
                    # For 8-bit: target ~75-125
                    bit_depth = 16 if img.max() > 255 else 8
                    target_median = 25000 if bit_depth == 16 else 100

                    # Adjust exposure proportionally
                    if median_intensity > 0:
                        adjusted_exp = initial_exp * (target_median / median_intensity)
                        # Clamp to reasonable range
                        adjusted_exp = max(0.1, min(adjusted_exp, 500.0))
                    else:
                        adjusted_exp = initial_exp

                    calibrated[angle] = adjusted_exp
                    self.logger.info(f"  Initial exp: {initial_exp:.2f}ms, median: {median_intensity:.0f}")
                    self.logger.info(f"  Calibrated exp: {adjusted_exp:.2f}ms")
                else:
                    calibrated[angle] = initial_exp
                    self.logger.warning(f"  Could not load image, using initial: {initial_exp:.2f}ms")
            else:
                calibrated[angle] = initial_exp
                self.logger.warning(f"  Acquisition failed, using initial: {initial_exp:.2f}ms")

        self.calibrated_exposures = calibrated

        # Save calibration data
        cal_file = self.output_dir / "calibrated_exposures.json"
        with open(cal_file, 'w') as f:
            json.dump({str(k): v for k, v in calibrated.items()}, f, indent=2)
        self.logger.info(f"Saved calibration data to {cal_file}")

        return calibrated

    def wait_for_user_stage_move(self):
        """
        Pause and wait for user to move stage to tissue area.

        This is the transition between Phase 1 (calibration) and Phase 2 (acquisition)
        in CALIBRATE mode.
        """
        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("STAGE MOVE REQUIRED")
        self.logger.info("=" * 70)
        self.logger.info("")
        self.logger.info("Background calibration complete!")
        self.logger.info("")
        self.logger.info("Please move the stage to a region with BIREFRINGENT TISSUE")
        self.logger.info("(e.g., collagen fibers, crystalline structures)")
        self.logger.info("")

        input("Press ENTER when ready to continue with tissue acquisition...")

        self.logger.info("")
        self.logger.info("Continuing with tissue acquisition...")
        self.logger.info("")

    def acquire_angle_pair(self, angle: float) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Acquire paired images at +angle and -angle.

        Args:
            angle: Absolute angle value (will acquire at +angle and -angle)

        Returns:
            Tuple of (positive_path, negative_path), either may be None on failure
        """
        positive_angle = abs(angle)
        negative_angle = -abs(angle)

        # Get exposures for each angle
        pos_exposure = self.get_exposure_for_angle(positive_angle)
        neg_exposure = self.get_exposure_for_angle(negative_angle)

        # Acquire positive angle
        pos_name = f"pos_{positive_angle:.2f}.tif"
        pos_path = self.acquire_at_angle(positive_angle, pos_name, pos_exposure)

        # Small delay between acquisitions
        time.sleep(0.2)

        # Acquire negative angle
        neg_name = f"neg_{abs(negative_angle):.2f}.tif"
        neg_path = self.acquire_at_angle(negative_angle, neg_name, neg_exposure)

        return pos_path, neg_path

    def compute_difference_image(self, pos_path: Path, neg_path: Path,
                                 angle: float) -> Optional[Path]:
        """
        Compute the difference image (birefringence signal) from paired images.

        Args:
            pos_path: Path to positive angle image
            neg_path: Path to negative angle image
            angle: The angle value (for naming)

        Returns:
            Path to difference image, or None on failure
        """
        try:
            # Load images
            pos_img = cv2.imread(str(pos_path), cv2.IMREAD_UNCHANGED)
            neg_img = cv2.imread(str(neg_path), cv2.IMREAD_UNCHANGED)

            if pos_img is None or neg_img is None:
                self.logger.error(f"Could not load images for angle {angle}")
                return None

            # Convert to grayscale if needed
            if len(pos_img.shape) == 3:
                pos_img = cv2.cvtColor(pos_img, cv2.COLOR_BGR2GRAY)
            if len(neg_img.shape) == 3:
                neg_img = cv2.cvtColor(neg_img, cv2.COLOR_BGR2GRAY)

            # Convert to float for difference calculation
            pos_float = pos_img.astype(np.float32)
            neg_float = neg_img.astype(np.float32)

            # Compute difference (birefringence signal)
            # Using absolute difference to capture signal magnitude
            diff = pos_float - neg_float

            # Save both signed and absolute difference
            diff_dir = self.output_dir / "differences"
            diff_dir.mkdir(exist_ok=True)

            # Signed difference (shifted to positive range for saving)
            signed_diff = diff + 32768  # Shift for 16-bit storage
            signed_diff = np.clip(signed_diff, 0, 65535).astype(np.uint16)
            signed_path = diff_dir / f"diff_signed_{angle:.2f}.tif"
            cv2.imwrite(str(signed_path), signed_diff)

            # Absolute difference (magnitude of birefringence signal)
            abs_diff = np.abs(diff).astype(np.uint16)
            abs_path = diff_dir / f"diff_abs_{angle:.2f}.tif"
            cv2.imwrite(str(abs_path), abs_diff)

            return abs_path

        except Exception as e:
            self.logger.error(f"Error computing difference for angle {angle}: {e}")
            return None

    def compute_birefringence_metrics(self, diff_path: Path, angle: float) -> Dict:
        """
        Compute metrics for the birefringence signal.

        Args:
            diff_path: Path to absolute difference image
            angle: The angle value

        Returns:
            Dictionary of metrics
        """
        try:
            img = cv2.imread(str(diff_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                return {'error': 'Could not load image'}

            # Compute statistics
            metrics = {
                'angle': angle,
                'mean_signal': float(np.mean(img)),
                'median_signal': float(np.median(img)),
                'max_signal': float(np.max(img)),
                'std_signal': float(np.std(img)),
                'signal_area': float(np.sum(img > np.percentile(img, 95))),  # Pixels with strong signal
            }

            # Compute percentiles for robust statistics
            metrics['p90_signal'] = float(np.percentile(img, 90))
            metrics['p95_signal'] = float(np.percentile(img, 95))
            metrics['p99_signal'] = float(np.percentile(img, 99))

            # Signal-to-background ratio
            background = np.percentile(img, 10)
            signal_region = np.percentile(img, 90)
            if background > 0:
                metrics['signal_to_bg_ratio'] = float(signal_region / background)
            else:
                metrics['signal_to_bg_ratio'] = float(signal_region)

            return metrics

        except Exception as e:
            return {'error': str(e), 'angle': angle}

    def run_paired_acquisition(self) -> Dict[float, Dict]:
        """
        Run the main paired acquisition test.

        Acquires image pairs at each test angle and computes birefringence metrics.

        Returns:
            Dictionary mapping angle -> metrics
        """
        self.logger.info("=" * 70)
        self.logger.info("PAIRED IMAGE ACQUISITION FOR BIREFRINGENCE ANALYSIS")
        self.logger.info("=" * 70)

        all_metrics = {}

        for i, angle in enumerate(self.test_angles):
            self.logger.info(f"\n[{i+1}/{len(self.test_angles)}] Testing angle pair: +/-{angle:.2f} deg")

            # Acquire the pair
            pos_path, neg_path = self.acquire_angle_pair(angle)

            if pos_path and neg_path:
                # Store paths
                self.acquired_images[angle] = {
                    'positive': pos_path,
                    'negative': neg_path
                }

                # Compute difference
                diff_path = self.compute_difference_image(pos_path, neg_path, angle)

                if diff_path:
                    self.difference_images[angle] = diff_path

                    # Compute metrics
                    metrics = self.compute_birefringence_metrics(diff_path, angle)
                    all_metrics[angle] = metrics
                    self.birefringence_metrics[angle] = metrics

                    self.logger.info(f"  Mean signal: {metrics.get('mean_signal', 0):.1f}")
                    self.logger.info(f"  Max signal: {metrics.get('max_signal', 0):.1f}")
                    self.logger.info(f"  P95 signal: {metrics.get('p95_signal', 0):.1f}")
                else:
                    self.logger.warning(f"  Failed to compute difference image")
            else:
                self.logger.warning(f"  Failed to acquire image pair")

        return all_metrics

    def find_optimal_angle(self) -> Tuple[float, Dict]:
        """
        Find the angle that maximizes birefringence signal.

        Returns:
            Tuple of (optimal_angle, metrics_at_optimal)
        """
        if not self.birefringence_metrics:
            return 0.0, {}

        # Find angle with maximum mean signal
        best_angle = 0.0
        best_signal = 0.0

        for angle, metrics in self.birefringence_metrics.items():
            signal = metrics.get('mean_signal', 0)
            if signal > best_signal:
                best_signal = signal
                best_angle = angle

        return best_angle, self.birefringence_metrics.get(best_angle, {})

    def compute_sensitivity_curve(self) -> Tuple[List[float], List[float], float, float]:
        """
        Compute the sensitivity curve (derivative of signal vs angle).

        Sensitivity = d(signal)/d(angle) - shows where small angle changes
        produce the largest intensity changes.

        Returns:
            Tuple of (angles, sensitivities, max_sensitivity_angle, max_sensitivity_value)
        """
        if len(self.birefringence_metrics) < 3:
            return [], [], 0.0, 0.0

        angles = sorted(self.birefringence_metrics.keys())
        signals = [self.birefringence_metrics[a].get('mean_signal', 0) for a in angles]

        # Compute numerical derivative (central difference where possible)
        sensitivities = []
        sensitivity_angles = []

        for i in range(len(angles)):
            if i == 0:
                # Forward difference
                if len(angles) > 1:
                    dx = angles[i+1] - angles[i]
                    dy = signals[i+1] - signals[i]
                    sens = abs(dy / dx) if dx != 0 else 0
                else:
                    sens = 0
            elif i == len(angles) - 1:
                # Backward difference
                dx = angles[i] - angles[i-1]
                dy = signals[i] - signals[i-1]
                sens = abs(dy / dx) if dx != 0 else 0
            else:
                # Central difference
                dx = angles[i+1] - angles[i-1]
                dy = signals[i+1] - signals[i-1]
                sens = abs(dy / dx) if dx != 0 else 0

            sensitivities.append(sens)
            sensitivity_angles.append(angles[i])

        # Find maximum sensitivity
        max_sens = 0.0
        max_sens_angle = 0.0
        for i, (angle, sens) in enumerate(zip(sensitivity_angles, sensitivities)):
            if sens > max_sens:
                max_sens = sens
                max_sens_angle = angle

        return sensitivity_angles, sensitivities, max_sens_angle, max_sens

    def generate_visualization(self):
        """Generate visualization plots of the birefringence analysis."""
        if not MATPLOTLIB_AVAILABLE:
            self.logger.warning("matplotlib not available, skipping visualization")
            return

        if not self.birefringence_metrics:
            self.logger.warning("No metrics to visualize")
            return

        self.logger.info("Generating visualization plots...")

        # Extract data for plotting
        angles = sorted(self.birefringence_metrics.keys())
        mean_signals = [self.birefringence_metrics[a].get('mean_signal', 0) for a in angles]
        max_signals = [self.birefringence_metrics[a].get('max_signal', 0) for a in angles]
        p95_signals = [self.birefringence_metrics[a].get('p95_signal', 0) for a in angles]

        # Compute sensitivity curve
        sens_angles, sensitivities, max_sens_angle, max_sens_value = self.compute_sensitivity_curve()

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # Get optimal angles for marking
        optimal_angle, optimal_metrics = self.find_optimal_angle()

        # Plot 1: Signal vs Angle (main result)
        ax = axes[0, 0]
        ax.plot(angles, mean_signals, 'b-o', label='Mean Signal', markersize=3, linewidth=1.5)
        ax.plot(angles, p95_signals, 'r-s', label='P95 Signal', markersize=3, linewidth=1, alpha=0.7)
        ax.axvline(x=optimal_angle, color='green', linestyle='--', linewidth=2,
                   label=f'Max Signal: {optimal_angle:.2f} deg')
        if max_sens_angle != optimal_angle:
            ax.axvline(x=max_sens_angle, color='orange', linestyle=':', linewidth=2,
                       label=f'Max Sensitivity: {max_sens_angle:.2f} deg')
        ax.set_xlabel('Angle (degrees)')
        ax.set_ylabel('Birefringence Signal Intensity')
        ax.set_title('BIREFRINGENCE SIGNAL vs POLARIZER ANGLE')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Plot 2: Sensitivity (derivative) vs Angle
        ax = axes[0, 1]
        if sens_angles and sensitivities:
            ax.plot(sens_angles, sensitivities, 'm-o', markersize=3, linewidth=1.5)
            ax.axvline(x=max_sens_angle, color='orange', linestyle='--', linewidth=2,
                       label=f'Max: {max_sens_angle:.2f} deg')
            ax.fill_between(sens_angles, sensitivities, alpha=0.3, color='magenta')
        ax.set_xlabel('Angle (degrees)')
        ax.set_ylabel('Sensitivity (signal change per degree)')
        ax.set_title('SENSITIVITY CURVE (d(Signal)/d(Angle))')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Signal-to-background ratio
        ax = axes[0, 2]
        sb_ratios = [self.birefringence_metrics[a].get('signal_to_bg_ratio', 0) for a in angles]
        ax.plot(angles, sb_ratios, 'g-^', markersize=3, linewidth=1.5)
        ax.axvline(x=optimal_angle, color='green', linestyle='--', linewidth=2)
        ax.set_xlabel('Angle (degrees)')
        ax.set_ylabel('Signal / Background Ratio')
        ax.set_title('Signal-to-Background Ratio vs Angle')
        ax.grid(True, alpha=0.3)

        # Plot 4: Difference image at OPTIMAL angle (max signal)
        ax = axes[1, 0]
        if self.difference_images and optimal_angle in self.difference_images:
            diff_img = cv2.imread(str(self.difference_images[optimal_angle]), cv2.IMREAD_UNCHANGED)
            if diff_img is not None:
                im = ax.imshow(diff_img, cmap='hot')
                ax.set_title(f'Birefringence at {optimal_angle:.2f} deg (MAX SIGNAL)')
                plt.colorbar(im, ax=ax, fraction=0.046)
                ax.axis('off')
        else:
            ax.text(0.5, 0.5, 'No image', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Optimal Angle Image')

        # Plot 5: Difference image at MAX SENSITIVITY angle
        ax = axes[1, 1]
        if self.difference_images and max_sens_angle in self.difference_images:
            sens_diff = cv2.imread(str(self.difference_images[max_sens_angle]), cv2.IMREAD_UNCHANGED)
            if sens_diff is not None:
                im = ax.imshow(sens_diff, cmap='hot')
                ax.set_title(f'Birefringence at {max_sens_angle:.2f} deg (MAX SENSITIVITY)')
                plt.colorbar(im, ax=ax, fraction=0.046)
                ax.axis('off')
        else:
            ax.text(0.5, 0.5, 'No image', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Max Sensitivity Image')

        # Plot 6: Zero-angle sanity check
        ax = axes[1, 2]
        if 0.0 in self.difference_images:
            zero_diff = cv2.imread(str(self.difference_images[0.0]), cv2.IMREAD_UNCHANGED)
            if zero_diff is not None:
                im = ax.imshow(zero_diff, cmap='hot')
                zero_mean = np.mean(zero_diff)
                ax.set_title(f'SANITY CHECK: 0 deg (mean={zero_mean:.1f})')
                plt.colorbar(im, ax=ax, fraction=0.046)
                ax.axis('off')
                self.logger.info(f"Sanity check (0 deg): mean signal = {zero_mean:.1f}")
        else:
            ax.text(0.5, 0.5, 'No 0 deg image', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Sanity Check: 0 deg')

        # Add summary text box
        summary_text = (
            f"RESULTS SUMMARY\n"
            f"---------------\n"
            f"Max Signal Angle: {optimal_angle:.2f} deg\n"
            f"  Signal: {optimal_metrics.get('mean_signal', 0):.1f}\n"
            f"\n"
            f"Max Sensitivity Angle: {max_sens_angle:.2f} deg\n"
            f"  Sensitivity: {max_sens_value:.1f}/deg\n"
        )

        plt.suptitle('PPM BIREFRINGENCE MAXIMIZATION ANALYSIS', fontsize=16, fontweight='bold')
        plt.tight_layout()

        plot_path = self.output_dir / 'birefringence_analysis.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Saved visualization to {plot_path}")

    def save_results(self):
        """Save all results to files."""
        # Save metrics as JSON
        metrics_file = self.output_dir / "birefringence_metrics.json"
        metrics_data = {
            'test_date': datetime.now().isoformat(),
            'config_file': str(self.config_yaml),
            'angle_range': list(self.angle_range),
            'angle_step': self.angle_step,
            'exposure_mode': self.exposure_mode,
            'metrics': {str(k): v for k, v in self.birefringence_metrics.items()}
        }

        # Find optimal angle and sensitivity
        optimal_angle, optimal_metrics = self.find_optimal_angle()
        sens_angles, sensitivities, max_sens_angle, max_sens_value = self.compute_sensitivity_curve()

        metrics_data['optimal_angle'] = optimal_angle
        metrics_data['optimal_metrics'] = optimal_metrics
        metrics_data['max_sensitivity_angle'] = max_sens_angle
        metrics_data['max_sensitivity_value'] = max_sens_value
        metrics_data['sensitivity_curve'] = {
            'angles': sens_angles,
            'sensitivities': sensitivities
        }

        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2)

        # Save CSV for easy analysis
        csv_file = self.output_dir / "birefringence_metrics.csv"
        with open(csv_file, 'w') as f:
            # Header
            if self.birefringence_metrics:
                first_metrics = list(self.birefringence_metrics.values())[0]
                headers = ['angle'] + [k for k in first_metrics.keys() if k != 'angle']
                f.write(','.join(headers) + '\n')

                # Data rows
                for angle in sorted(self.birefringence_metrics.keys()):
                    metrics = self.birefringence_metrics[angle]
                    row = [str(angle)]
                    for h in headers[1:]:
                        row.append(str(metrics.get(h, '')))
                    f.write(','.join(row) + '\n')

        # Save human-readable summary
        summary_file = self.output_dir / "birefringence_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("PPM BIREFRINGENCE MAXIMIZATION TEST SUMMARY\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Test Date: {datetime.now().isoformat()}\n")
            f.write(f"Config: {self.config_yaml}\n")
            f.write(f"Angle Range: {self.angle_range[0]} to {self.angle_range[1]} degrees\n")
            f.write(f"Step Size: {self.angle_step} degrees\n")
            f.write(f"Exposure Mode: {self.exposure_mode}\n")
            f.write(f"Images Retained: {self.keep_images}\n\n")

            f.write("=" * 70 + "\n")
            f.write("KEY RESULTS: MAXIMUM SIGNAL AND MAXIMUM SENSITIVITY\n")
            f.write("=" * 70 + "\n\n")

            # Compute sensitivity
            sens_angles, sensitivities, max_sens_angle, max_sens_value = self.compute_sensitivity_curve()

            f.write("1. MAXIMUM BIREFRINGENCE SIGNAL\n")
            f.write("-" * 40 + "\n")
            f.write(f"   Optimal angle: {optimal_angle:.2f} degrees\n")
            f.write(f"   Mean signal: {optimal_metrics.get('mean_signal', 0):.1f}\n")
            f.write(f"   Max signal: {optimal_metrics.get('max_signal', 0):.1f}\n")
            f.write(f"   P95 signal: {optimal_metrics.get('p95_signal', 0):.1f}\n")
            f.write(f"   Signal/BG ratio: {optimal_metrics.get('signal_to_bg_ratio', 0):.2f}\n\n")

            f.write("2. MAXIMUM SENSITIVITY (steepest signal change)\n")
            f.write("-" * 40 + "\n")
            f.write(f"   Angle of max sensitivity: {max_sens_angle:.2f} degrees\n")
            f.write(f"   Sensitivity: {max_sens_value:.2f} signal units per degree\n")
            if max_sens_angle in self.birefringence_metrics:
                sens_metrics = self.birefringence_metrics[max_sens_angle]
                f.write(f"   Signal at this angle: {sens_metrics.get('mean_signal', 0):.1f}\n")
            f.write("\n")

            f.write("3. INTERPRETATION\n")
            f.write("-" * 40 + "\n")
            if optimal_angle == max_sens_angle:
                f.write("   Max signal and max sensitivity occur at the SAME angle.\n")
                f.write(f"   --> Recommended operating angle: {optimal_angle:.2f} deg\n\n")
            else:
                f.write(f"   Max signal at {optimal_angle:.2f} deg, max sensitivity at {max_sens_angle:.2f} deg\n")
                f.write("   Consider:\n")
                f.write(f"   - Use {optimal_angle:.2f} deg for strongest birefringence contrast\n")
                f.write(f"   - Use {max_sens_angle:.2f} deg for most sensitive detection\n\n")

            # Sanity check
            f.write("=" * 70 + "\n")
            f.write("SANITY CHECK (0 degree)\n")
            f.write("=" * 70 + "\n\n")

            zero_metrics = self.birefringence_metrics.get(0.0, {})
            if zero_metrics:
                f.write(f"At 0 degrees (should be ~zero):\n")
                f.write(f"  Mean signal: {zero_metrics.get('mean_signal', 0):.1f}\n")
                f.write(f"  Max signal: {zero_metrics.get('max_signal', 0):.1f}\n")

                # Compare to optimal
                if optimal_metrics.get('mean_signal', 0) > 0:
                    ratio = zero_metrics.get('mean_signal', 0) / optimal_metrics.get('mean_signal', 1)
                    f.write(f"  Ratio vs optimal: {ratio:.4f} (should be << 1)\n")

                    if ratio < 0.1:
                        f.write("  --> GOOD: Zero-angle signal is low as expected\n")
                    elif ratio < 0.3:
                        f.write("  --> ACCEPTABLE: Some residual signal at 0 deg\n")
                    else:
                        f.write("  --> WARNING: High signal at 0 deg suggests alignment issues\n")
            f.write("\n")

            # Full table
            f.write("=" * 70 + "\n")
            f.write("FULL RESULTS TABLE\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"{'Angle':>8}  {'Mean':>10}  {'Max':>10}  {'P95':>10}  {'S/B Ratio':>10}\n")
            f.write("-" * 55 + "\n")

            for angle in sorted(self.birefringence_metrics.keys()):
                m = self.birefringence_metrics[angle]
                f.write(f"{angle:>8.2f}  {m.get('mean_signal', 0):>10.1f}  "
                       f"{m.get('max_signal', 0):>10.1f}  {m.get('p95_signal', 0):>10.1f}  "
                       f"{m.get('signal_to_bg_ratio', 0):>10.2f}\n")

            f.write("\n")
            f.write("=" * 70 + "\n")
            f.write("END OF SUMMARY\n")
            f.write("=" * 70 + "\n")

        self.logger.info(f"Saved results to {self.output_dir}")

    def cleanup_images(self):
        """Remove .tif files if keep_images=False."""
        if self.keep_images:
            self.logger.info("Keeping all .tif image files")
            return 0

        self.logger.info("Cleaning up .tif files...")
        removed = 0

        # Main directory
        for tif in self.output_dir.glob("*.tif"):
            try:
                tif.unlink()
                removed += 1
            except Exception as e:
                self.logger.warning(f"Failed to remove {tif}: {e}")

        # Subdirectories
        for subdir in ['differences', 'calibration']:
            sub_path = self.output_dir / subdir
            if sub_path.exists():
                for tif in sub_path.glob("*.tif"):
                    try:
                        tif.unlink()
                        removed += 1
                    except Exception as e:
                        self.logger.warning(f"Failed to remove {tif}: {e}")

        self.logger.info(f"Removed {removed} .tif files")
        return removed

    def run_test(self):
        """
        Run the complete birefringence maximization test.

        For INTERPOLATE mode: directly acquires paired images using interpolated exposures
        For CALIBRATE mode: first calibrates exposures, pauses for stage move, then acquires
        """
        self.logger.info("=" * 70)
        self.logger.info("PPM BIREFRINGENCE MAXIMIZATION TEST")
        self.logger.info("=" * 70)
        self.logger.info(f"Mode: {self.exposure_mode.upper()}")

        if not self.connect():
            self.logger.error("Failed to connect to server. Aborting.")
            return None

        try:
            if self.exposure_mode == "calibrate":
                # Phase 1: Background calibration
                self.run_background_calibration()

                # Pause for stage move
                self.wait_for_user_stage_move()

            # Main acquisition (Phase 2 for calibrate, only phase for interpolate)
            self.logger.info("")
            self.logger.info("=" * 70)
            self.logger.info("PHASE 2: TISSUE ACQUISITION" if self.exposure_mode == "calibrate"
                           else "PAIRED ACQUISITION")
            self.logger.info("=" * 70)

            self.run_paired_acquisition()

            # Analysis
            self.logger.info("")
            self.logger.info("=" * 70)
            self.logger.info("ANALYSIS")
            self.logger.info("=" * 70)

            # Find optimal
            optimal_angle, optimal_metrics = self.find_optimal_angle()
            self.logger.info(f"\nOPTIMAL ANGLE: {optimal_angle:.2f} degrees")
            self.logger.info(f"  Mean signal: {optimal_metrics.get('mean_signal', 0):.1f}")
            self.logger.info(f"  Max signal: {optimal_metrics.get('max_signal', 0):.1f}")

            # Sanity check
            zero_metrics = self.birefringence_metrics.get(0.0, {})
            if zero_metrics:
                self.logger.info(f"\nSANITY CHECK (0 deg): mean signal = {zero_metrics.get('mean_signal', 0):.1f}")

            # Generate outputs
            self.generate_visualization()
            self.save_results()

            # Cleanup if requested
            if not self.keep_images:
                self.cleanup_images()

            self.logger.info("")
            self.logger.info("=" * 70)
            self.logger.info("TEST COMPLETE")
            self.logger.info("=" * 70)
            self.logger.info(f"Results saved to: {self.output_dir}")

            return self.output_dir

        finally:
            self.disconnect()


def run_birefringence_maximization_test(
    config_yaml: str,
    output_dir: str = None,
    host: str = "127.0.0.1",
    port: int = 5000,
    angle_range: Tuple[float, float] = (-10.0, 10.0),
    angle_step: float = 0.1,
    exposure_mode: str = "interpolate",
    keep_images: bool = True,
    calibration_exposures: Dict[float, float] = None
) -> Optional[Path]:
    """
    Run birefringence maximization test programmatically.

    This function can be called from QuPath or other applications.

    Args:
        config_yaml: Path to microscope configuration YAML
        output_dir: Output directory (auto-generated if None)
        host: qp_server host
        port: qp_server port
        angle_range: (min, max) angles to test
        angle_step: Step size in degrees
        exposure_mode: 'interpolate' or 'calibrate'
        keep_images: If False, delete .tif files after analysis
        calibration_exposures: Optional override for calibration exposures

    Returns:
        Path to output directory on success, None on failure
    """
    tester = PPMBirefringenceMaximizationTester(
        config_yaml=config_yaml,
        output_dir=output_dir,
        host=host,
        port=port,
        angle_range=angle_range,
        angle_step=angle_step,
        exposure_mode=exposure_mode,
        keep_images=keep_images,
        calibration_exposures=calibration_exposures
    )

    return tester.run_test()


def main():
    """Main entry point for CLI usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description='PPM Birefringence Maximization Test',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with interpolated exposures
  python ppm_birefringence_maximization_test.py config.yml --mode interpolate

  # Full test with background calibration (more accurate)
  python ppm_birefringence_maximization_test.py config.yml --mode calibrate

  # Custom angle range
  python ppm_birefringence_maximization_test.py config.yml --min-angle -5 --max-angle 5 --step 0.05

  # Don't keep images (save disk space)
  python ppm_birefringence_maximization_test.py config.yml --no-keep-images
"""
    )

    parser.add_argument('config_yaml',
                       help='Path to microscope configuration YAML file')
    parser.add_argument('--output', '-o',
                       help='Output directory for results')
    parser.add_argument('--host', default='127.0.0.1',
                       help='qp_server host address')
    parser.add_argument('--port', type=int, default=5000,
                       help='qp_server port')
    parser.add_argument('--mode', choices=['interpolate', 'calibrate'],
                       default='interpolate',
                       help='Exposure mode: interpolate (use calibration points) or '
                            'calibrate (measure exposures first)')
    parser.add_argument('--min-angle', type=float, default=-10.0,
                       help='Minimum angle in degrees (default: -10)')
    parser.add_argument('--max-angle', type=float, default=10.0,
                       help='Maximum angle in degrees (default: 10)')
    parser.add_argument('--step', type=float, default=0.1,
                       help='Angle step size in degrees (default: 0.1)')
    parser.add_argument('--keep-images', dest='keep_images', action='store_true',
                       default=True,
                       help='Keep acquired .tif images (default)')
    parser.add_argument('--no-keep-images', dest='keep_images', action='store_false',
                       help='Delete .tif images after analysis')
    parser.add_argument('--calibration-exposures', type=str, default=None,
                       help='JSON dict of calibration exposures, e.g. \'{"7.0": 25.0}\'')

    args = parser.parse_args()

    # Parse calibration exposures if provided
    calibration_exposures = None
    if args.calibration_exposures:
        try:
            calibration_exposures = json.loads(args.calibration_exposures)
            calibration_exposures = {float(k): float(v) for k, v in calibration_exposures.items()}
        except Exception as e:
            print(f"Error parsing --calibration-exposures: {e}")
            return

    # Run the test
    result = run_birefringence_maximization_test(
        config_yaml=args.config_yaml,
        output_dir=args.output,
        host=args.host,
        port=args.port,
        angle_range=(args.min_angle, args.max_angle),
        angle_step=args.step,
        exposure_mode=args.mode,
        keep_images=args.keep_images,
        calibration_exposures=calibration_exposures
    )

    if result:
        print(f"\nTest complete. Results saved to: {result}")
    else:
        print("\nTest failed")


if __name__ == "__main__":
    main()
