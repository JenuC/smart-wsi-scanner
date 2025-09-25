#!/usr/bin/env python3
"""
Test script for tissue detection functionality across different imaging modalities.

Usage:
    python test_tissue_detection.py <image_path> [modality]

Examples:
    python test_tissue_detection.py sample_ppm.tif PPM_20x
    python test_tissue_detection.py sample_bf.tif brightfield
    python test_tissue_detection.py sample_shg.tif SHG
"""

import sys
import pathlib
import numpy as np
import tifffile as tf
import logging
from src.smart_wsi_scanner.qp_utils import AutofocusUtils

def setup_logging():
    """Setup logging for the test."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def load_test_image(image_path: str) -> np.ndarray:
    """Load and validate test image."""
    path = pathlib.Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        image = tf.imread(str(path))
        return image
    except Exception as e:
        raise ValueError(f"Could not load image {image_path}: {e}")

def main():
    logger = setup_logging()

    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    image_path = sys.argv[1]
    modality = sys.argv[2] if len(sys.argv) > 2 else "unknown"

    logger.info(f"Testing tissue detection on: {image_path}")
    logger.info(f"Modality: {modality}")

    try:
        # Load the image
        image = load_test_image(image_path)
        logger.info(f"Loaded image: {image.shape}, dtype: {image.dtype}")

        # Test with default parameters
        logger.info("\n" + "="*50)
        logger.info("TESTING WITH DEFAULT PARAMETERS")
        logger.info("="*50)

        has_tissue_default = AutofocusUtils.has_sufficient_tissue(
            image, logger=logger, modality=modality
        )
        logger.info(f"Result with defaults: {'PASS' if has_tissue_default else 'FAIL'}")

        # Comprehensive testing
        logger.info("\n" + "="*50)
        logger.info("COMPREHENSIVE ANALYSIS")
        logger.info("="*50)

        analysis = AutofocusUtils.test_tissue_detection(
            image=image,
            modality=modality,
            texture_thresholds=[0.005, 0.01, 0.015, 0.02, 0.03, 0.05],
            area_thresholds=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
            show_analysis=True,
            logger=logger
        )

        # Summary and recommendations
        logger.info("\n" + "="*50)
        logger.info("RECOMMENDATIONS FOR THIS IMAGE/MODALITY")
        logger.info("="*50)

        recs = analysis['recommendations']
        logger.info(f"Recommended texture threshold: {recs['suggested_texture_threshold']:.4f}")
        logger.info(f"Recommended area threshold: {recs['suggested_area_threshold']:.3f}")
        logger.info(f"Best tissue mask strategy: {recs['best_tissue_mask']}")
        logger.info(f"Image has good contrast: {recs['has_good_contrast']}")

        # Test with recommended parameters
        logger.info("\n" + "="*50)
        logger.info("TESTING WITH RECOMMENDED PARAMETERS")
        logger.info("="*50)

        has_tissue_recommended = AutofocusUtils.has_sufficient_tissue(
            image,
            texture_threshold=recs['suggested_texture_threshold'],
            tissue_area_threshold=recs['suggested_area_threshold'],
            modality=modality,
            logger=logger
        )
        logger.info(f"Result with recommendations: {'PASS' if has_tissue_recommended else 'FAIL'}")

        # Final summary
        logger.info("\n" + "="*50)
        logger.info("FINAL SUMMARY")
        logger.info("="*50)
        logger.info(f"Image: {image_path}")
        logger.info(f"Modality: {modality}")
        logger.info(f"Default result: {'PASS' if has_tissue_default else 'FAIL'}")
        logger.info(f"Recommended result: {'PASS' if has_tissue_recommended else 'FAIL'}")

        # Count how many threshold combinations passed
        passed_combinations = sum(1 for result in analysis['threshold_results'] if result['has_tissue'])
        total_combinations = len(analysis['threshold_results'])
        logger.info(f"Passed {passed_combinations}/{total_combinations} threshold combinations")

        if passed_combinations == 0:
            logger.warning("⚠️  No threshold combinations passed - image may lack sufficient tissue")
        elif passed_combinations < total_combinations * 0.3:
            logger.warning("⚠️  Few threshold combinations passed - tissue detection may be challenging")
        else:
            logger.info("✅ Good tissue detection prospects for this image")

    except Exception as e:
        logger.error(f"Test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()