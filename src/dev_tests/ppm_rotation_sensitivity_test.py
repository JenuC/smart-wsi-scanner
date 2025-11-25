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
from smart_wsi_scanner.qp_server_config import ExtendedCommand, END_MARKER
from smart_wsi_scanner.config import ConfigurationManager

# Import the analysis modules from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
try:
    from ppm_rotation_sensitivity_analysis import PPMRotationAnalyzer
    ANALYZER_AVAILABLE = True
except ImportError:
    print("Warning: PPM rotation analyzer not found. Analysis will be skipped.")
    ANALYZER_AVAILABLE = False


class PPMRotationSensitivityTester:
    """
    Comprehensive PPM rotation sensitivity testing using existing infrastructure.

    This class coordinates systematic image acquisition at precise angles and
    analyzes the impact on image quality and birefringence calculations.
    """

    def __init__(self,
                 config_yaml: str,
                 output_dir: str = None,
                 host: str = "127.0.0.1",
                 port: int = 5000):
        """
        Initialize the tester with configuration and connection parameters.

        Args:
            config_yaml: Path to microscope configuration YAML file
            output_dir: Output directory for test results
            host: qp_server host address
            port: qp_server port
        """
        self.config_yaml = Path(config_yaml)
        self.output_dir = Path(output_dir or f"ppm_sensitivity_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize logging
        self.setup_logging()

        # Load configuration
        self.config_manager = ConfigurationManager(str(self.config_yaml))
        self.config = self.config_manager.config

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

        self.logger.info(f"PPM Rotation Sensitivity Tester initialized")
        self.logger.info(f"Config: {self.config_yaml}")
        self.logger.info(f"Output: {self.output_dir}")

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
            # Move to angle
            self.logger.debug(f"Moving to angle {angle} degrees")
            self.client.test_move_rotation(angle)
            time.sleep(0.5)  # Allow settling

            # Verify position
            actual_angle = self.client.test_get_rotation()
            angle_error = abs(actual_angle - angle)
            self.logger.info(f"Set: {angle:.2f} deg, Read: {actual_angle:.2f} deg, Error: {angle_error:.3f} deg")

            if angle_error > 0.5:  # Warning threshold
                self.logger.warning(f"Large angle error: {angle_error:.3f} degrees")

            # Build acquisition command for background acquisition
            # We'll use the BGACQUIRE command with single angle
            output_path = self.output_dir / save_name

            message = (
                f"--yaml {self.config_yaml} "
                f"--output {output_path.parent} "
                f"--modality {self.config.get('modality', 'ppm_20x')} "
                f"--angles ({angle}) "
            )

            if exposure_ms:
                message += f"--exposures ({exposure_ms}) "

            message += END_MARKER

            # Send BGACQUIRE command
            self.logger.debug(f"Sending BGACQUIRE command")
            self.client.socket.sendall(ExtendedCommand.BGACQUIRE)
            self.client.socket.sendall(message.encode('utf-8'))

            # Wait for acknowledgment
            ack = self.client.socket.recv(256).decode()
            self.logger.debug(f"ACK: {ack}")

            if ack.startswith("STARTED:"):
                # Monitor acquisition
                self.logger.debug("Monitoring acquisition progress")
                while True:
                    time.sleep(0.5)
                    status = self.client.test_status()
                    if status == "COMPLETED":
                        # Get final response
                        result = self.client.socket.recv(1024).decode()
                        if result.startswith("SUCCESS:"):
                            self.logger.info(f"Successfully acquired image at {angle} deg -> {save_name}")
                            return output_path
                        else:
                            self.logger.error(f"Acquisition failed: {result}")
                            return None
                    elif status in ["FAILED", "CANCELLED"]:
                        self.logger.error(f"Acquisition {status}")
                        return None
            else:
                self.logger.error(f"Failed to start acquisition: {ack}")
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
                               deviations: List[float] = None) -> Dict[float, Path]:
        """
        Test fine angular deviations around a base angle.

        Args:
            base_angle: Center angle for testing
            deviations: List of deviations to test (positive and negative)

        Returns:
            Dictionary mapping angles to image paths
        """
        if deviations is None:
            deviations = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.7, 1.0]

        self.logger.info("=" * 60)
        self.logger.info(f"FINE DEVIATION TEST: {base_angle} degrees")
        self.logger.info("=" * 60)
        self.logger.info(f"Testing deviations: {deviations}")

        acquired = {}

        for i, dev in enumerate(deviations):
            # Test positive deviation
            angle_pos = base_angle + dev
            if 0 <= angle_pos <= 180:  # Check hardware limits
                self.logger.info(f"[{i*2+1}/{len(deviations)*2}] Testing {angle_pos:.2f} deg (+{dev})")
                save_name = f"deviation_{base_angle}deg_plus_{dev:.2f}.tif"
                image_path = self.acquire_at_angle(angle_pos, save_name)
                if image_path:
                    acquired[angle_pos] = image_path

            # Test negative deviation (skip 0)
            if dev > 0:
                angle_neg = base_angle - dev
                if 0 <= angle_neg <= 180:
                    self.logger.info(f"[{i*2+2}/{len(deviations)*2}] Testing {angle_neg:.2f} deg (-{dev})")
                    save_name = f"deviation_{base_angle}deg_minus_{dev:.2f}.tif"
                    image_path = self.acquire_at_angle(angle_neg, save_name)
                    if image_path:
                        acquired[angle_neg] = image_path

        self.logger.info(f"Acquired {len(acquired)} deviation test images")
        self.test_results[f'deviation_{base_angle}deg'] = acquired
        return acquired

    def run_repeatability_test(self, test_angle: float = 7.0,
                              n_repeats: int = 10) -> List[Dict]:
        """
        Test mechanical repeatability at a single angle.

        Args:
            test_angle: Angle to test repeatedly
            n_repeats: Number of repetitions

        Returns:
            List of measurement dictionaries
        """
        self.logger.info("=" * 60)
        self.logger.info(f"REPEATABILITY TEST: {test_angle} degrees x {n_repeats}")
        self.logger.info("=" * 60)

        measurements = []

        for i in range(n_repeats):
            self.logger.info(f"Repetition {i+1}/{n_repeats}")

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
                'set_angle': test_angle,
                'read_angle': actual,
                'error': error,
                'timestamp': datetime.now().isoformat()
            }
            measurements.append(measurement)

            self.logger.info(f"  Set: {test_angle:.3f}, Read: {actual:.3f}, Error: {error:.3f}")

        # Calculate statistics
        errors = [m['error'] for m in measurements]
        stats = {
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'max_error': np.max(np.abs(errors)),
            'repeatability_2sigma': 2 * np.std(errors)
        }

        self.logger.info(f"Repeatability Statistics:")
        self.logger.info(f"  Mean error: {stats['mean_error']:.4f} degrees")
        self.logger.info(f"  Std deviation: {stats['std_error']:.4f} degrees")
        self.logger.info(f"  Max error: {stats['max_error']:.4f} degrees")
        self.logger.info(f"  2-sigma repeatability: {stats['repeatability_2sigma']:.4f} degrees")

        self.test_results['repeatability'] = {
            'measurements': measurements,
            'statistics': stats
        }

        return measurements

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

        try:
            # Run calibration sweep using the utility
            results = self.pol_utils.calibrate_hardware_offset_two_stage(
                yaml_file_path=str(self.config_yaml),
                output_folder_path=str(output_folder),
                start_angle=0,
                end_angle=180,
                coarse_step=5.0,
                fine_step=0.5,
                exposure_ms=10.0,
                hardware=None,  # Will be created by utility
                config_manager=self.config_manager,
                logger=self.logger
            )

            self.logger.info(f"Calibration complete:")
            self.logger.info(f"  Crossed positions: {results.get('crossed_positions', [])}")
            self.logger.info(f"  Optimal exposure: {results.get('optimal_exposure', 'N/A')}")

            self.test_results['polarizer_calibration'] = results
            return results

        except Exception as e:
            self.logger.error(f"Polarizer calibration failed: {e}")
            return {}

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

            # Load acquired images
            images = analyzer.load_images()

            if len(images) < 3:
                self.logger.warning("Insufficient images for analysis")
                return {}

            # Run analyses
            self.logger.info("Computing image differences...")
            df_differences = analyzer.compute_image_differences(reference_angle=7.0)

            self.logger.info("Analyzing birefringence sensitivity...")
            df_birefringence = analyzer.analyze_birefringence_sensitivity(
                deviations=[0.1, 0.2, 0.3, 0.5, 1.0]
            )

            # Generate visualizations
            self.logger.info("Generating visualizations...")
            analyzer.visualize_difference_maps(reference_angle=7.0)
            analyzer.visualize_birefringence_comparison()
            analyzer.generate_sensitivity_plots(df_differences, df_birefringence)

            # Generate report
            self.logger.info("Generating analysis report...")
            report = analyzer.generate_report(df_differences, df_birefringence)

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
                summary['results'][test_name] = {
                    'images_acquired': len(results),
                    'angle_range': [min(results.keys()), max(results.keys())]
                }
            elif test_name == 'repeatability':
                summary['results'][test_name] = results.get('statistics', {})
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
                stats = self.test_results['repeatability']['statistics']
                f.write(f"\nMECHANICAL REPEATABILITY:\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Mean error: {stats['mean_error']:.4f} degrees\n")
                f.write(f"  2-sigma repeatability: {stats['repeatability_2sigma']:.4f} degrees\n")
                f.write(f"  Maximum error: {stats['max_error']:.4f} degrees\n")

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