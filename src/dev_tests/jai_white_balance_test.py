#!/usr/bin/env python3
"""
JAI Camera White Balance Test Script

Standalone script to test white balance calibration on the JAI AP-3200T-USB camera.
Runs directly with pycromanager - no additional packages required.

Usage:
    python jai_white_balance_test.py

Requirements:
    - Micro-Manager running with JAICamera device configured
    - Pycromanager connected
    - Neutral gray/white target in field of view (or defocused blank area)
"""

import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import yaml

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class JAIWhiteBalanceTest:
    """Simple white balance calibration for JAI camera."""

    DEVICE_NAME = "JAICamera"

    def __init__(self, core):
        self.core = core

    def enable_individual_exposure(self) -> None:
        """Enable per-channel exposure mode."""
        self.core.set_property(self.DEVICE_NAME, "ExposureIsIndividual", "On")
        self.core.wait_for_device(self.DEVICE_NAME)
        logger.info("Enabled individual exposure mode")

    def set_frame_rate(self, hz: float) -> None:
        """Set frame rate to allow desired exposure times."""
        hz = max(0.125, min(39.0, hz))
        self.core.set_property(self.DEVICE_NAME, "FrameRateHz", str(hz))
        self.core.wait_for_device(self.DEVICE_NAME)
        logger.info(f"Set frame rate to {hz} Hz")

    def set_channel_exposures(self, red: float, green: float, blue: float) -> None:
        """Set per-channel exposure times in milliseconds."""
        # Adjust frame rate for longest exposure
        max_exp = max(red, green, blue)
        required_frame_rate = 1000.0 / (max_exp * 1.05)  # 5% margin
        required_frame_rate = max(0.125, min(38.0, required_frame_rate))
        self.set_frame_rate(required_frame_rate)

        self.core.set_property(self.DEVICE_NAME, "Exposure_Red", str(red))
        self.core.set_property(self.DEVICE_NAME, "Exposure_Green", str(green))
        self.core.set_property(self.DEVICE_NAME, "Exposure_Blue", str(blue))
        self.core.wait_for_device(self.DEVICE_NAME)
        logger.info(f"Set exposures: R={red:.2f}ms, G={green:.2f}ms, B={blue:.2f}ms")

    def get_channel_exposures(self) -> Dict[str, float]:
        """Get current per-channel exposure times."""
        return {
            "red": float(self.core.get_property(self.DEVICE_NAME, "Exposure_Red")),
            "green": float(self.core.get_property(self.DEVICE_NAME, "Exposure_Green")),
            "blue": float(self.core.get_property(self.DEVICE_NAME, "Exposure_Blue")),
        }

    def capture_and_analyze(self) -> Dict[str, float]:
        """Capture image and return per-channel means."""
        self.core.snap_image()
        img = self.core.get_image()

        width = self.core.get_image_width()
        height = self.core.get_image_height()
        bytes_per_pixel = self.core.get_bytes_per_pixel()

        if bytes_per_pixel == 4:
            # 32bitRGB (BGRA format)
            img_array = np.frombuffer(img, dtype=np.uint8).reshape((height, width, 4))
            means = {
                "red": float(img_array[:, :, 2].mean()),
                "green": float(img_array[:, :, 1].mean()),
                "blue": float(img_array[:, :, 0].mean()),
            }
        elif bytes_per_pixel == 3:
            img_array = np.frombuffer(img, dtype=np.uint8).reshape((height, width, 3))
            means = {
                "red": float(img_array[:, :, 0].mean()),
                "green": float(img_array[:, :, 1].mean()),
                "blue": float(img_array[:, :, 2].mean()),
            }
        else:
            raise ValueError(f"Unsupported bytes_per_pixel: {bytes_per_pixel}")

        return means

    def calibrate(
        self,
        target: float = 180.0,
        tolerance: float = 5.0,
        max_iterations: int = 20,
        initial_exposure: float = 20.0,
        damping: float = 0.7,
    ) -> Tuple[Dict[str, float], Dict[str, float], bool]:
        """
        Run iterative white balance calibration.

        Args:
            target: Target intensity (0-255)
            tolerance: Acceptable deviation from target
            max_iterations: Max iterations before giving up
            initial_exposure: Starting exposure for all channels (ms)
            damping: Damping factor to prevent oscillation (0-1)

        Returns:
            Tuple of (final_exposures, final_means, converged)
        """
        logger.info(f"Starting white balance calibration (target={target}, tolerance={tolerance})")

        self.enable_individual_exposure()

        # Start with equal exposures
        exposures = {"red": initial_exposure, "green": initial_exposure, "blue": initial_exposure}
        self.set_channel_exposures(**exposures)

        for iteration in range(1, max_iterations + 1):
            # Capture and analyze
            time.sleep(0.1)  # Allow settings to stabilize
            means = self.capture_and_analyze()

            # Check convergence
            converged = all(
                abs(means[ch] - target) <= tolerance
                for ch in ["red", "green", "blue"]
            )

            logger.info(
                f"Iter {iteration}: R={means['red']:.1f} ({exposures['red']:.2f}ms), "
                f"G={means['green']:.1f} ({exposures['green']:.2f}ms), "
                f"B={means['blue']:.1f} ({exposures['blue']:.2f}ms)"
            )

            if converged:
                logger.info(f"Converged after {iteration} iterations!")
                return exposures, means, True

            # Adjust exposures proportionally with damping
            for ch in ["red", "green", "blue"]:
                if means[ch] > 0:
                    ratio = target / means[ch]
                    damped_ratio = 1.0 + (ratio - 1.0) * damping
                    new_exp = exposures[ch] * damped_ratio
                    # Clamp to reasonable range
                    exposures[ch] = max(0.1, min(500.0, new_exp))

            self.set_channel_exposures(**exposures)

        logger.warning(f"Did not converge after {max_iterations} iterations")
        return exposures, means, False

    def save_results(
        self,
        exposures: Dict[str, float],
        means: Dict[str, float],
        converged: bool,
        output_path: Path,
    ) -> None:
        """Save calibration results to YAML file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "metadata": {
                "generated": datetime.now().isoformat(),
                "camera": "JAI AP-3200T-USB",
                "converged": converged,
            },
            "per_channel_exposures_ms": {
                "red": round(exposures["red"], 2),
                "green": round(exposures["green"], 2),
                "blue": round(exposures["blue"], 2),
            },
            "final_intensities": {
                "red": round(means["red"], 1),
                "green": round(means["green"], 1),
                "blue": round(means["blue"], 1),
            },
            "combined_settings": {
                "ExposureIsIndividual": "On",
                "Exposure_Red": round(exposures["red"], 2),
                "Exposure_Green": round(exposures["green"], 2),
                "Exposure_Blue": round(exposures["blue"], 2),
            },
        }

        with open(output_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Saved results to {output_path}")


def main():
    """Run white balance calibration test."""
    from pycromanager import Core

    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    logger.info("Connecting to Micro-Manager...")
    core = Core()

    # Verify JAI camera is active
    active_camera = core.get_property("Core", "Camera")
    if active_camera != "JAICamera":
        logger.error(f"JAICamera not active. Current camera: {active_camera}")
        return

    logger.info("JAICamera detected. Starting white balance calibration...")
    print("\n" + "=" * 60)
    print("IMPORTANT: Position a neutral gray/white target in the FOV")
    print("or defocus on a blank slide area before continuing.")
    print("=" * 60)
    input("\nPress Enter when ready...")

    calibrator = JAIWhiteBalanceTest(core)

    # Run calibration
    exposures, means, converged = calibrator.calibrate(
        target=180.0,
        tolerance=5.0,
        max_iterations=20,
        initial_exposure=20.0,
    )

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = output_dir / f"white_balance_settings_{timestamp}.yml"
    calibrator.save_results(exposures, means, converged, output_path)

    # Print summary
    print("\n" + "=" * 60)
    print("WHITE BALANCE CALIBRATION RESULTS")
    print("=" * 60)
    print(f"Converged: {converged}")
    print(f"\nFinal exposures:")
    print(f"  Red:   {exposures['red']:.2f} ms")
    print(f"  Green: {exposures['green']:.2f} ms")
    print(f"  Blue:  {exposures['blue']:.2f} ms")
    print(f"\nFinal intensities:")
    print(f"  Red:   {means['red']:.1f}")
    print(f"  Green: {means['green']:.1f}")
    print(f"  Blue:  {means['blue']:.1f}")
    print(f"\nSettings saved to: {output_path}")
    print("=" * 60)

    # Restore unified exposure mode
    core.set_property("JAICamera", "ExposureIsIndividual", "Off")
    logger.info("Restored unified exposure mode")


if __name__ == "__main__":
    main()
