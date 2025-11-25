#!/usr/bin/env python3
"""
Batch tissue detection testing - processes all images in a folder and outputs CSV summary.

Usage:
    python test_tissue_batch.py /path/to/folder
    python test_tissue_batch.py /path/to/folder --recursive
"""

import sys
import pathlib
import numpy as np
import tifffile as tf
import csv
from datetime import datetime

sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))
from smart_wsi_scanner.qp_utils import AutofocusUtils


def analyze_image_stats(image_path: pathlib.Path, modality: str = "ppm"):
    """
    Analyze a single image and return statistics dictionary.
    """
    try:
        image = tf.imread(str(image_path))

        # Run analysis
        analysis = AutofocusUtils.test_tissue_detection(
            image,
            modality=modality,
            texture_thresholds=[0.005, 0.01, 0.015, 0.02, 0.03],
            area_thresholds=[0.05, 0.10, 0.15, 0.20],
            show_analysis=False,  # Suppress logging
            logger=None
        )

        # Test with various threshold combinations
        test_combos = [
            ("Current_Default", 0.015, 0.15),
            ("Relaxed", 0.010, 0.10),
            ("Strict", 0.020, 0.20),
            ("Very_Relaxed", 0.005, 0.05),
        ]

        results = {
            'image_path': str(image_path),
            'image_name': image_path.name,
            'shape': str(image.shape),
            'dtype': str(image.dtype),
            'min': float(image.min()),
            'max': float(image.max()),
            'mean': float(image.mean()),
            'std': float(image.std()),
        }

        # Add intensity stats
        int_stats = analysis['intensity_stats']
        results.update({
            'norm_mean': int_stats['mean'],
            'norm_std': int_stats['std'],
            'norm_median': int_stats['median'],
        })

        # Add gradient stats
        grad_stats = analysis['gradient_stats']
        results.update({
            'gradient_mean': grad_stats['mean'],
            'gradient_std': grad_stats['std'],
            'gradient_max': grad_stats['max'],
            'gradient_p95': grad_stats['p95'],
        })

        # Add mask analysis
        for mask_name, mask_stats in analysis['mask_analysis'].items():
            clean_name = mask_name.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
            results[f'mask_{clean_name}_texture'] = mask_stats['texture']
            results[f'mask_{clean_name}_area'] = mask_stats['area_fraction']

        # Add recommendations
        recs = analysis['recommendations']
        results.update({
            'rec_texture_threshold': recs['suggested_texture_threshold'],
            'rec_area_threshold': recs['suggested_area_threshold'],
            'rec_has_good_contrast': recs['has_good_contrast'],
        })

        # Test threshold combinations
        for combo_name, tex_thresh, area_thresh in test_combos:
            has_tissue = AutofocusUtils.has_sufficient_tissue(
                image, tex_thresh, area_thresh, modality=modality
            )
            results[f'test_{combo_name}_tex{tex_thresh}_area{area_thresh}'] = 'PASS' if has_tissue else 'FAIL'

        return results, None

    except Exception as e:
        return None, str(e)


def find_tif_images(directory: pathlib.Path, recursive: bool = False):
    """
    Find all .tif images in directory.
    """
    if recursive:
        return sorted(directory.rglob("*.tif"))
    else:
        return sorted(directory.glob("*.tif"))


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python test_tissue_batch.py <directory>")
        print("  python test_tissue_batch.py <directory> --recursive")
        print("\nProcesses all .tif images and outputs CSV summary.")
        sys.exit(1)

    input_dir = pathlib.Path(sys.argv[1])
    recursive = "--recursive" in sys.argv or "-r" in sys.argv

    if not input_dir.exists():
        print(f"Error: Directory not found: {input_dir}")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"BATCH TISSUE DETECTION ANALYSIS")
    print(f"{'='*70}")
    print(f"Directory: {input_dir}")
    print(f"Recursive: {recursive}")
    print(f"{'='*70}\n")

    # Find images
    print("Searching for .tif images...")
    images = find_tif_images(input_dir, recursive)

    if not images:
        print(f"No .tif images found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(images)} images\n")

    # Ask for modality
    modality = input("Enter modality (default: ppm): ").strip() or "ppm"

    # Process images
    results_list = []
    errors = []

    print(f"\nProcessing images...")
    print(f"{'='*70}\n")

    for idx, img_path in enumerate(images, 1):
        print(f"[{idx}/{len(images)}] {img_path.name}...", end=" ")

        results, error = analyze_image_stats(img_path, modality)

        if error:
            print(f"ERROR: {error}")
            errors.append((str(img_path), error))
        else:
            print("OK")
            results_list.append(results)

    if not results_list:
        print("\nNo images successfully processed!")
        sys.exit(1)

    # Write CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = input_dir / f"tissue_detection_analysis_{timestamp}.csv"

    print(f"\n{'='*70}")
    print(f"Writing results to: {output_csv}")
    print(f"{'='*70}\n")

    # Get all field names from first result
    fieldnames = list(results_list[0].keys())

    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_list)

    print(f"✓ Processed {len(results_list)} images successfully")

    if errors:
        print(f"✗ {len(errors)} images failed:")
        for img_path, error in errors:
            print(f"  - {img_path}: {error}")

    # Print summary statistics
    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS")
    print(f"{'='*70}\n")

    # Count how many pass each test
    test_columns = [col for col in fieldnames if col.startswith('test_')]

    for test_col in test_columns:
        pass_count = sum(1 for r in results_list if r.get(test_col) == 'PASS')
        fail_count = len(results_list) - pass_count
        pass_pct = 100 * pass_count / len(results_list)

        test_name = test_col.replace('test_', '').replace('_', ' ')
        print(f"{test_name:40s}: {pass_count:3d} PASS ({pass_pct:5.1f}%), {fail_count:3d} FAIL")

    # Gradient statistics
    print(f"\nGradient Texture Statistics:")
    grad_stds = [r['gradient_std'] for r in results_list]
    print(f"  Min:    {min(grad_stds):.6f}")
    print(f"  Max:    {max(grad_stds):.6f}")
    print(f"  Mean:   {np.mean(grad_stds):.6f}")
    print(f"  Median: {np.median(grad_stds):.6f}")

    # Recommended thresholds
    print(f"\nRecommended Texture Thresholds:")
    rec_tex = [r['rec_texture_threshold'] for r in results_list]
    print(f"  Min:    {min(rec_tex):.6f}")
    print(f"  Max:    {max(rec_tex):.6f}")
    print(f"  Mean:   {np.mean(rec_tex):.6f}")
    print(f"  Median: {np.median(rec_tex):.6f}")

    print(f"\n{'='*70}")
    print(f"RECOMMENDATION")
    print(f"{'='*70}\n")

    # Find the most relaxed setting that passes most images
    median_rec_tex = np.median(rec_tex)
    median_rec_area = np.median([r['rec_area_threshold'] for r in results_list])

    print(f"Based on {len(results_list)} images, use these settings:")
    print(f"  texture_threshold = {median_rec_tex:.6f}")
    print(f"  tissue_area_threshold = {median_rec_area:.3f}")

    print(f"\nIn qp_acquisition.py around line 474:")
    print(f"    has_tissue, tissue_stats = AutofocusUtils.has_sufficient_tissue(")
    print(f"        test_img,")
    print(f"        texture_threshold={median_rec_tex:.6f},")
    print(f"        tissue_area_threshold={median_rec_area:.3f},")
    print(f"        modality=modality,")
    print(f"        logger=logger,")
    print(f"        return_stats=True")
    print(f"    )")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
