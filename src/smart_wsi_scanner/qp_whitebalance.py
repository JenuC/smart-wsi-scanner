#!/usr/bin/env python3
"""
White Balance Auto-Adjustment Tool
Converted from PMQI_WhiteBalance Java plugin
Author: Python conversion
License: BSD

This tool automatically adjusts white balance for color cameras by:
1. Finding optimal exposure time
2. Debayering the image using nearest neighbor
3. Calculating RGB scale factors

Structure Overview:

WhiteBalanceController - Main controller class that:

Finds optimal exposure time (find_exposure_for_wb)
Debayers raw Bayer pattern images (debayer_image)
Calculates RGB scale factors (calculate_scales)
Orchestrates the complete white balance process (run_white_balance)


Enums for constants:

WBResult - Operation result codes
CFAPattern - Color Filter Array patterns (RGGB, BGGR, etc.)
BitDepth - Camera bit depths


MockCameraInterface - Mock camera for testing (replace with actual camera control)

Key Changes from Java:

Simplified UI: Removed Swing GUI components, focusing on core algorithm
NumPy arrays: Used for efficient image processing instead of ImageJ processors
Comprehensive logging: Added debug logs throughout for troubleshooting
Type hints: Added for better code clarity
Pythonic patterns: Used enums, tuples, and exception handling

Core Algorithm Features:

Exposure Finding: Iteratively adjusts exposure to achieve target mean pixel value
Debayering: Nearest-neighbor algorithm that replicates pixels based on CFA pattern
Scale Calculation: Normalizes RGB channels to achieve white balance

Usage:
python# Create camera interface (implement your own)
camera = YourCameraInterface()

# Create controller
wb = WhiteBalanceController(camera)

# Configure
wb.set_bit_depth(BitDepth.DEPTH_16BIT)
wb.set_cfa_pattern(CFAPattern.RGGB)

# Run white balance
if wb.run_white_balance():
    print(f"Success! Scales: R={wb.red_scale}, G={wb.green_scale}, B={wb.blue_scale}")
The code includes extensive debug logging to help trace the algorithm's progress and diagnose issues.
https://claude.ai/public/artifacts/0cb3d19c-032c-40ea-8f74-77dee5521515

"""


import numpy as np
import logging
from enum import IntEnum
from typing import Tuple, Optional
import sys

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class WBResult(IntEnum):
    """White balance operation result codes"""

    FAILED_TOO_MANY_ITERATIONS = 0
    FAILED_TOO_MANY_ITERATIONS_TOO_BRIGHT = 1
    FAILED_TOO_MANY_ITERATIONS_TOO_DARK = 2
    FAILED_SATURATED_FROM_START = 3
    FAILED_TOO_DARK_FROM_START = 4
    FAILED_EXP_TOO_SHORT = 5
    FAILED_EXP_TOO_LONG = 6
    FAILED_EXCEPTION = 7
    SUCCESS = 8


class CFAPattern(IntEnum):
    """Color Filter Array patterns"""

    RGGB = 0
    BGGR = 1
    GRBG = 2
    GBRG = 3


class BitDepth(IntEnum):
    """Camera bit depths"""

    DEPTH_8BIT = 8
    DEPTH_10BIT = 10
    DEPTH_12BIT = 12
    DEPTH_14BIT = 14
    DEPTH_16BIT = 16


class WhiteBalanceController:
    """
    Main controller for white balance adjustment
    Handles exposure finding, image debayering, and scale calculation
    """

    # Constants
    MIN_EXPOSURE = 2  # ms
    MAX_EXPOSURE = 2000  # ms
    WB_EXP_ITERATIONS_MAX = 20
    WB_SUCCESS_SNAP_FACTOR = 1.75

    # Target mean values for different bit depths
    MEAN_TARGETS = {
        BitDepth.DEPTH_16BIT: (27000, 33000),
        BitDepth.DEPTH_14BIT: (6600, 8500),
        BitDepth.DEPTH_12BIT: (1700, 2300),
        BitDepth.DEPTH_10BIT: (420, 600),
        BitDepth.DEPTH_8BIT: (100, 150),  # Estimated
    }

    # ADU bias values
    ADU_BIAS_16BIT = 500
    ADU_BIAS_LESS16BIT = 150

    def __init__(self, camera_interface):
        """
        Initialize the white balance controller

        Args:
            camera_interface: Object that provides camera control methods
        """
        logger.info("Initializing WhiteBalanceController")

        self.camera = camera_interface
        self.bit_depth = BitDepth.DEPTH_16BIT
        self.cfa_pattern = CFAPattern.RGGB
        self.wb_result = None

        # Results
        self.wb_exposure = 0.0
        self.r_mean = 0.0
        self.g_mean = 0.0
        self.b_mean = 0.0
        self.red_scale = 1.0
        self.green_scale = 1.0
        self.blue_scale = 1.0

        # Get target mean range
        self.wb_mean_min, self.wb_mean_max = self.MEAN_TARGETS[self.bit_depth]

        logger.debug(
            f"Bit depth: {self.bit_depth}, Target mean: {self.wb_mean_min}-{self.wb_mean_max}"
        )

    def set_bit_depth(self, bit_depth: BitDepth):
        """Update bit depth and corresponding target mean values"""
        self.bit_depth = bit_depth
        self.wb_mean_min, self.wb_mean_max = self.MEAN_TARGETS[bit_depth]
        logger.info(
            f"Set bit depth to {bit_depth}, target mean: {self.wb_mean_min}-{self.wb_mean_max}"
        )

    def set_cfa_pattern(self, pattern: CFAPattern):
        """Update CFA pattern"""
        self.cfa_pattern = pattern
        logger.info(f"Set CFA pattern to {pattern.name}")

    def find_exposure_for_wb(self) -> Tuple[float, WBResult]:
        """
        Find optimal exposure time for white balance

        Returns:
            Tuple of (exposure_time_ms, result_code)
        """
        logger.info("Starting exposure search for white balance")

        exposure = 5.0  # Start with 5ms
        iterations = 0
        max_adu_val = 2**self.bit_depth

        try:
            # Check if image is not saturated with short exposure
            logger.debug("Checking for saturation with 1ms exposure")
            image = self.camera.snap_image(1.0)
            mean_1ms = np.mean(image)

            if mean_1ms > (max_adu_val - max_adu_val / 7.0):
                logger.error(f"Image saturated at 1ms (mean: {mean_1ms:.1f})")
                return 1.0, WBResult.FAILED_SATURATED_FROM_START

            # Check if image is not too dark with longer exposure
            logger.debug("Checking brightness with 100ms exposure")
            image = self.camera.snap_image(100.0)
            mean_100ms = np.mean(image)

            bias = (
                self.ADU_BIAS_16BIT
                if self.bit_depth == BitDepth.DEPTH_16BIT
                else self.ADU_BIAS_LESS16BIT
            )
            if mean_100ms < bias:
                logger.error(f"Image too dark at 100ms (mean: {mean_100ms:.1f})")
                return 100.0, WBResult.FAILED_TOO_DARK_FROM_START

            # Search for optimal exposure
            logger.debug("Starting iterative exposure search")
            image = self.camera.snap_image(exposure)
            current_mean = np.mean(image)
            exposure = exposure * self.wb_mean_min / current_mean

            while iterations < self.WB_EXP_ITERATIONS_MAX and exposure < self.MAX_EXPOSURE:
                image = self.camera.snap_image(exposure)
                current_mean = np.mean(image)

                logger.debug(
                    f"Iteration {iterations}: exposure={exposure:.1f}ms, mean={current_mean:.1f}"
                )

                iterations += 1

                if self.wb_mean_min <= current_mean <= self.wb_mean_max:
                    logger.info(
                        f"Found optimal exposure: {exposure:.1f}ms (mean: {current_mean:.1f})"
                    )
                    self.last_captured_image = image
                    break

                # Adjust exposure
                exposure = exposure * self.wb_mean_min / current_mean
                exposure = max(self.MIN_EXPOSURE, min(self.MAX_EXPOSURE, exposure))

            # Check failure conditions
            if iterations >= self.WB_EXP_ITERATIONS_MAX:
                if exposure >= self.MAX_EXPOSURE:
                    logger.error("Too many iterations, image too dark")
                    return exposure, WBResult.FAILED_TOO_MANY_ITERATIONS_TOO_DARK
                elif exposure <= self.MIN_EXPOSURE:
                    logger.error("Too many iterations, image too bright")
                    return exposure, WBResult.FAILED_TOO_MANY_ITERATIONS_TOO_BRIGHT
                else:
                    logger.error("Too many iterations")
                    return exposure, WBResult.FAILED_TOO_MANY_ITERATIONS

            if exposure > self.MAX_EXPOSURE:
                logger.error("Exposure too long")
                return exposure, WBResult.FAILED_EXP_TOO_LONG

            return exposure, WBResult.SUCCESS

        except Exception as e:
            logger.exception("Exception during exposure search")
            return exposure, WBResult.FAILED_EXCEPTION

    def debayer_image(self, image: np.ndarray) -> Tuple[float, float, float]:
        """
        Debayer image using nearest neighbor algorithm

        Args:
            image: Raw sensor image

        Returns:
            Tuple of (r_mean, g_mean, b_mean)
        """
        logger.info(f"Debayering image with pattern {self.cfa_pattern.name}")

        height, width = image.shape

        # Initialize color channels
        r = np.zeros_like(image, dtype=np.float32)
        g = np.zeros_like(image, dtype=np.float32)
        b = np.zeros_like(image, dtype=np.float32)

        if self.cfa_pattern in [CFAPattern.GRBG, CFAPattern.GBRG]:
            # Pattern starts with G in top-left
            # Blue pixels at (1,0), (1,2), (3,0), (3,2)...
            b[1::2, 0::2] = image[1::2, 0::2]
            b[0::2, 0::2] = image[1::2, 0::2]  # Replicate to neighbors
            b[1::2, 1::2] = image[1::2, 0::2]
            b[0::2, 1::2] = image[1::2, 0::2]

            # Red pixels at (0,1), (0,3), (2,1), (2,3)...
            r[0::2, 1::2] = image[0::2, 1::2]
            r[0::2, 0::2] = image[0::2, 1::2]  # Replicate to neighbors
            r[1::2, 1::2] = image[0::2, 1::2]
            r[1::2, 0::2] = image[0::2, 1::2]

            # Green pixels at (0,0), (0,2), (1,1), (1,3)...
            g[0::2, 0::2] = image[0::2, 0::2]
            g[0::2, 1::2] = image[0::2, 0::2]
            g[1::2, 1::2] = image[1::2, 1::2]
            g[1::2, 0::2] = image[1::2, 1::2]

            if self.cfa_pattern == CFAPattern.GRBG:
                # Swap R and B for GRBG
                r_mean = np.mean(b)
                g_mean = np.mean(g)
                b_mean = np.mean(r)
            else:  # GBRG
                r_mean = np.mean(r)
                g_mean = np.mean(g)
                b_mean = np.mean(b)

        else:  # RGGB or BGGR
            # Pattern starts with R or B in top-left
            # Blue/Red pixels at (0,0), (0,2), (2,0), (2,2)...
            b[0::2, 0::2] = image[0::2, 0::2]
            b[0::2, 1::2] = image[0::2, 0::2]  # Replicate to neighbors
            b[1::2, 0::2] = image[0::2, 0::2]
            b[1::2, 1::2] = image[0::2, 0::2]

            # Red/Blue pixels at (1,1), (1,3), (3,1), (3,3)...
            r[1::2, 1::2] = image[1::2, 1::2]
            r[0::2, 1::2] = image[1::2, 1::2]  # Replicate to neighbors
            r[1::2, 0::2] = image[1::2, 1::2]
            r[0::2, 0::2] = image[1::2, 1::2]

            # Green pixels at (0,1), (1,0), (1,2), (2,1)...
            g[0::2, 1::2] = image[0::2, 1::2]
            g[0::2, 0::2] = image[0::2, 1::2]
            g[1::2, 0::2] = image[1::2, 0::2]
            g[1::2, 1::2] = image[1::2, 0::2]

            if self.cfa_pattern == CFAPattern.RGGB:
                # For RGGB, top-left is R, bottom-right is B
                r_mean = np.mean(b)  # Swapped
                g_mean = np.mean(g)
                b_mean = np.mean(r)  # Swapped
            else:  # BGGR
                r_mean = np.mean(r)
                g_mean = np.mean(g)
                b_mean = np.mean(b)

        logger.debug(f"Channel means - R: {r_mean:.1f}, G: {g_mean:.1f}, B: {b_mean:.1f}")
        return r_mean, g_mean, b_mean

    def calculate_scales(self) -> Tuple[float, float, float]:
        """
        Calculate RGB scale factors for white balance

        Returns:
            Tuple of (red_scale, green_scale, blue_scale)
        """
        logger.info("Calculating white balance scales")

        # Start with red as reference
        red_scale = 1.0
        blue_scale = self.r_mean / self.b_mean if self.b_mean > 0 else 1.0
        green_scale = self.r_mean / self.g_mean if self.g_mean > 0 else 1.0

        # If green or blue is brighter, use that as reference
        if self.g_mean > self.r_mean or self.b_mean > self.r_mean:
            if self.g_mean > self.b_mean:
                # Green is brightest
                green_scale = 1.0
                blue_scale = self.g_mean / self.b_mean if self.b_mean > 0 else 1.0
                red_scale = self.g_mean / self.r_mean if self.r_mean > 0 else 1.0
            else:
                # Blue is brightest
                blue_scale = 1.0
                red_scale = self.b_mean / self.r_mean if self.r_mean > 0 else 1.0
                green_scale = self.b_mean / self.g_mean if self.g_mean > 0 else 1.0

        # Limit scales to 20.0 (as in original)
        red_scale = min(red_scale, 20.0)
        green_scale = min(green_scale, 20.0)
        blue_scale = min(blue_scale, 20.0)

        logger.info(
            f"Calculated scales - R: {red_scale:.4f}, G: {green_scale:.4f}, B: {blue_scale:.4f}"
        )

        return red_scale, green_scale, blue_scale

    def run_white_balance(self) -> bool:
        """
        Run the complete white balance adjustment

        Returns:
            True if successful, False otherwise
        """
        logger.info("Starting white balance adjustment")

        try:
            # Find optimal exposure
            self.wb_exposure, self.wb_result = self.find_exposure_for_wb()

            if self.wb_result != WBResult.SUCCESS:
                logger.error(f"Failed to find exposure: {self.wb_result.name}")
                return False

            # Capture image at optimal exposure
            logger.info(f"Capturing image at {self.wb_exposure:.1f}ms for white balance")
            image = self.camera.snap_image(self.wb_exposure)

            # Debayer the image
            self.r_mean, self.g_mean, self.b_mean = self.debayer_image(image)

            # Calculate scales
            self.red_scale, self.green_scale, self.blue_scale = self.calculate_scales()

            # Apply scales to camera
            logger.info("Applying white balance scales to camera")
            self.camera.set_property("Color - Red scale", self.red_scale)
            self.camera.set_property("Color - Green scale", self.green_scale)
            self.camera.set_property("Color - Blue scale", self.blue_scale)

            logger.info("White balance adjustment completed successfully")
            return True

        except Exception as e:
            logger.exception("Error during white balance adjustment")
            self.wb_result = WBResult.FAILED_EXCEPTION
            return False


class MockCameraInterface:
    """
    Mock camera interface for testing
    Replace with actual camera control implementation
    """

    def __init__(self):
        logger.info("Initializing mock camera interface")
        self.exposure_time = 1.0
        self.properties = {
            "Color - Red scale": 1.0,
            "Color - Green scale": 1.0,
            "Color - Blue scale": 1.0,
        }

    def snap_image(self, exposure_ms: float) -> np.ndarray:
        """Simulate image capture"""
        logger.debug(f"Mock snap at {exposure_ms}ms")

        # Simulate a Bayer pattern image
        size = (1024, 1024)

        # Create base noise
        image = np.random.normal(100, 10, size)

        # Add Bayer pattern with different sensitivities
        # Simulate RGGB pattern
        image[0::2, 0::2] *= 0.8  # Red pixels (slightly less sensitive)
        image[0::2, 1::2] *= 1.0  # Green pixels
        image[1::2, 0::2] *= 1.0  # Green pixels
        image[1::2, 1::2] *= 1.2  # Blue pixels (slightly more sensitive)

        # Scale by exposure
        image *= exposure_ms / 10.0

        # Add some gradient to simulate uneven illumination
        y_gradient = np.linspace(0.8, 1.2, size[0])[:, np.newaxis]
        image *= y_gradient

        # Clip to bit depth (assuming 12-bit for mock)
        image = np.clip(image, 0, 4095).astype(np.uint16)

        return image

    def set_property(self, prop_name: str, value: float):
        """Set camera property"""
        logger.debug(f"Setting {prop_name} to {value:.4f}")
        self.properties[prop_name] = value


def main():
    """Main function for testing"""
    logger.info("Starting White Balance Tool")

    # Create mock camera
    camera = MockCameraInterface()

    # Create controller
    wb_controller = WhiteBalanceController(camera)

    # Set parameters
    wb_controller.set_bit_depth(BitDepth.DEPTH_12BIT)
    wb_controller.set_cfa_pattern(CFAPattern.RGGB)

    # Run white balance
    success = wb_controller.run_white_balance()

    if success:
        print("\nWhite Balance Results:")
        print(f"Exposure: {wb_controller.wb_exposure:.1f} ms")
        print(
            f"Channel means - R: {wb_controller.r_mean:.1f}, "
            f"G: {wb_controller.g_mean:.1f}, B: {wb_controller.b_mean:.1f}"
        )
        print(
            f"Scale factors - R: {wb_controller.red_scale:.4f}, "
            f"G: {wb_controller.green_scale:.4f}, B: {wb_controller.blue_scale:.4f}"
        )
    else:
        print(f"\nWhite balance failed: {wb_controller.wb_result.name}")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
