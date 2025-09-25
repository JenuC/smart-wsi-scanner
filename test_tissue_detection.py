#!/usr/bin/env python3
"""
Test script for tissue detection functionality across different imaging modalities.

Usage:
    python test_tissue_detection.py <folder_path> [modality]

Examples:
    python test_tissue_detection.py /path/to/images/ PPM_20x
    python test_tissue_detection.py /path/to/images/ brightfield
    python test_tissue_detection.py /path/to/images/ SHG
"""

import sys
import pathlib
import numpy as np
import tifffile as tf
import logging
from datetime import datetime
from PIL import Image
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
        # Try with tifffile first (handles TIFF files best)
        if path.suffix.lower() in ['.tif', '.tiff']:
            image = tf.imread(str(path))
        else:
            # Use PIL for other formats (JPG, PNG, etc.)
            pil_image = Image.open(str(path))
            image = np.array(pil_image)
        return image
    except Exception as e:
        raise ValueError(f"Could not load image {image_path}: {e}")

def find_image_files(folder_path: str) -> list:
    """Find all image files in the specified folder."""
    folder = pathlib.Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Folder not found or not a directory: {folder_path}")

    image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
    image_files = []

    for file_path in folder.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            image_files.append(file_path)

    return sorted(image_files)  # Sort for consistent ordering

def process_single_image(image_path: pathlib.Path, modality: str, logger) -> dict:
    """Process a single image and return results."""
    try:
        # Load the image
        image = load_test_image(str(image_path))

        # Test with default parameters
        has_tissue_default = AutofocusUtils.has_sufficient_tissue(
            image, logger=None, modality=modality  # Suppress logging for bulk processing
        )

        # Get comprehensive analysis (without detailed logging)
        analysis = AutofocusUtils.test_tissue_detection(
            image=image,
            modality=modality,
            texture_thresholds=[0.005, 0.01, 0.015, 0.02, 0.03, 0.05],
            area_thresholds=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
            show_analysis=False,
            logger=None
        )

        # Test with recommended parameters
        recs = analysis['recommendations']
        has_tissue_recommended = AutofocusUtils.has_sufficient_tissue(
            image,
            texture_threshold=recs['suggested_texture_threshold'],
            tissue_area_threshold=recs['suggested_area_threshold'],
            modality=modality,
            logger=None
        )

        # Count passed threshold combinations
        passed_combinations = sum(1 for result in analysis['threshold_results'] if result['has_tissue'])
        total_combinations = len(analysis['threshold_results'])

        return {
            'success': True,
            'image_path': str(image_path),
            'image_shape': image.shape,
            'image_dtype': str(image.dtype),
            'has_tissue_default': has_tissue_default,
            'has_tissue_recommended': has_tissue_recommended,
            'recommendations': recs,
            'passed_combinations': passed_combinations,
            'total_combinations': total_combinations,
            'error': None
        }

    except Exception as e:
        return {
            'success': False,
            'image_path': str(image_path),
            'error': str(e)
        }

def main():
    logger = setup_logging()

    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    folder_path = sys.argv[1]
    modality = sys.argv[2] if len(sys.argv) > 2 else "unknown"

    logger.info(f"Testing tissue detection on folder: {folder_path}")
    logger.info(f"Modality: {modality}")

    try:
        # Find all image files in the folder
        image_files = find_image_files(folder_path)

        if not image_files:
            logger.warning(f"No image files found in {folder_path}")
            sys.exit(1)

        logger.info(f"Found {len(image_files)} image files")

        # Process all images
        results = []
        for i, image_file in enumerate(image_files, 1):
            logger.info(f"Processing {i}/{len(image_files)}: {image_file.name}")
            result = process_single_image(image_file, modality, logger)
            results.append(result)

        # Print summary to console
        logger.info("\n" + "="*60)
        logger.info("TISSUE DETECTION RESULTS SUMMARY")
        logger.info("="*60)

        pass_default = sum(1 for r in results if r.get('success') and r.get('has_tissue_default'))
        pass_recommended = sum(1 for r in results if r.get('success') and r.get('has_tissue_recommended'))
        total_successful = sum(1 for r in results if r.get('success'))
        total_failed = len(results) - total_successful

        logger.info(f"Total images processed: {len(results)}")
        logger.info(f"Successfully analyzed: {total_successful}")
        logger.info(f"Failed to analyze: {total_failed}")
        logger.info(f"Passed with default parameters: {pass_default}/{total_successful} ({100*pass_default/max(total_successful,1):.1f}%)")
        logger.info(f"Passed with recommended parameters: {pass_recommended}/{total_successful} ({100*pass_recommended/max(total_successful,1):.1f}%)")

        # Write detailed results to text file
        output_file = pathlib.Path(folder_path) / f"tissue_detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        with open(output_file, 'w') as f:
            f.write(f"Tissue Detection Analysis Results\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Folder: {folder_path}\n")
            f.write(f"Modality: {modality}\n")
            f.write(f"="*80 + "\n\n")

            f.write(f"SUMMARY:\n")
            f.write(f"Total images processed: {len(results)}\n")
            f.write(f"Successfully analyzed: {total_successful}\n")
            f.write(f"Failed to analyze: {total_failed}\n")
            f.write(f"Passed with default parameters: {pass_default}/{total_successful} ({100*pass_default/max(total_successful,1):.1f}%)\n")
            f.write(f"Passed with recommended parameters: {pass_recommended}/{total_successful} ({100*pass_recommended/max(total_successful,1):.1f}%)\n\n")

            f.write(f"DETAILED RESULTS:\n")
            f.write(f"-"*80 + "\n")

            for result in results:
                f.write(f"\nImage: {pathlib.Path(result['image_path']).name}\n")

                if result['success']:
                    f.write(f"  Shape: {result['image_shape']}\n")
                    f.write(f"  Data type: {result['image_dtype']}\n")
                    f.write(f"  Default result: {'PASS' if result['has_tissue_default'] else 'FAIL'}\n")
                    f.write(f"  Recommended result: {'PASS' if result['has_tissue_recommended'] else 'FAIL'}\n")
                    f.write(f"  Threshold combinations passed: {result['passed_combinations']}/{result['total_combinations']} ({100*result['passed_combinations']/result['total_combinations']:.1f}%)\n")

                    recs = result['recommendations']
                    f.write(f"  Recommended texture threshold: {recs['suggested_texture_threshold']:.4f}\n")
                    f.write(f"  Recommended area threshold: {recs['suggested_area_threshold']:.3f}\n")
                    f.write(f"  Best tissue mask strategy: {recs['best_tissue_mask']}\n")
                    f.write(f"  Good contrast: {recs['has_good_contrast']}\n")

                    if result['passed_combinations'] == 0:
                        f.write(f"  ⚠️  Warning: No threshold combinations passed - may lack sufficient tissue\n")
                    elif result['passed_combinations'] < result['total_combinations'] * 0.3:
                        f.write(f"  ⚠️  Warning: Few threshold combinations passed - detection may be challenging\n")
                    else:
                        f.write(f"  ✅ Good tissue detection prospects\n")
                else:
                    f.write(f"  ERROR: {result['error']}\n")

        logger.info(f"\nDetailed results written to: {output_file}")
        logger.info("Analysis complete!")

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()