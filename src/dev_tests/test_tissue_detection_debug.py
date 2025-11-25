#!/usr/bin/env python3
"""
Debug script for testing tissue detection with different thresholds.

Usage:
    python test_tissue_detection_debug.py /path/to/test/image.tif

This script will:
1. Load an image that failed tissue detection during acquisition
2. Test various threshold combinations
3. Display visual analysis of what the algorithm "sees"
4. Save debug images showing tissue masks and gradient maps
5. Recommend optimal thresholds for your data
"""

import sys
import pathlib
import numpy as np
import tifffile as tf
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Add the src directory to path
sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))

from smart_wsi_scanner.qp_utils import AutofocusUtils


def visualize_tissue_detection(image: np.ndarray, modality: str = "ppm"):
    """
    Comprehensive visualization of tissue detection algorithm.

    Shows:
    - Original image
    - Grayscale conversion
    - Gradient magnitude map
    - Tissue masks at different intensity ranges
    - Detection results at various thresholds
    """

    # Convert to grayscale
    if len(image.shape) == 3:
        img_gray = np.mean(image, axis=2).astype(np.float32)
    else:
        img_gray = image.astype(np.float32)

    # Normalize
    img_norm = (img_gray - img_gray.min()) / (img_gray.max() - img_gray.min() + 1e-10)

    # Calculate gradient
    gy, gx = np.gradient(img_norm)
    gradient_magnitude = np.sqrt(gx**2 + gy**2)

    # Different tissue masks
    masks = {
        'Conservative (0.1-0.9)': (img_norm > 0.1) & (img_norm < 0.9),
        'PPM Default (0.05-0.95)': (img_norm > 0.05) & (img_norm < 0.95),
        'Brightfield (0.15-0.85)': (img_norm > 0.15) & (img_norm < 0.85),
        'Narrow (0.2-0.8)': (img_norm > 0.2) & (img_norm < 0.8),
    }

    # Create figure
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Original image
    ax1 = fig.add_subplot(gs[0, 0])
    if len(image.shape) == 3:
        ax1.imshow(image)
    else:
        ax1.imshow(image, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')

    # Normalized grayscale
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(img_norm, cmap='gray', vmin=0, vmax=1)
    ax2.set_title(f'Normalized Gray\nMean: {img_norm.mean():.3f}, Std: {img_norm.std():.3f}')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046)

    # Gradient magnitude
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(gradient_magnitude, cmap='hot')
    ax3.set_title(f'Gradient Magnitude\nMean: {gradient_magnitude.mean():.4f}, Std: {gradient_magnitude.std():.4f}')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046)

    # Tissue masks
    for idx, (mask_name, mask) in enumerate(masks.items()):
        ax = fig.add_subplot(gs[1, idx % 3])
        ax.imshow(mask, cmap='gray')

        if np.any(mask):
            texture = np.std(gradient_magnitude[mask])
            area = np.sum(mask) / mask.size
        else:
            texture = 0.0
            area = 0.0

        ax.set_title(f'{mask_name}\nTexture: {texture:.4f}, Area: {area:.3f}')
        ax.axis('off')

    # Histogram of normalized intensities
    ax_hist = fig.add_subplot(gs[2, 0])
    ax_hist.hist(img_norm.ravel(), bins=50, alpha=0.7, color='blue')
    ax_hist.axvline(0.05, color='red', linestyle='--', label='PPM lower (0.05)')
    ax_hist.axvline(0.95, color='red', linestyle='--', label='PPM upper (0.95)')
    ax_hist.axvline(0.1, color='orange', linestyle='--', label='Default lower (0.1)')
    ax_hist.axvline(0.9, color='orange', linestyle='--', label='Default upper (0.9)')
    ax_hist.set_xlabel('Normalized Intensity')
    ax_hist.set_ylabel('Frequency')
    ax_hist.set_title('Intensity Distribution')
    ax_hist.legend(fontsize=8)
    ax_hist.grid(True, alpha=0.3)

    # Gradient histogram
    ax_grad_hist = fig.add_subplot(gs[2, 1])
    ax_grad_hist.hist(gradient_magnitude.ravel(), bins=50, alpha=0.7, color='green')
    ax_grad_hist.axvline(0.015, color='red', linestyle='--', label='PPM threshold (0.015)')
    ax_grad_hist.axvline(0.02, color='orange', linestyle='--', label='Default threshold (0.02)')
    ax_grad_hist.set_xlabel('Gradient Magnitude')
    ax_grad_hist.set_ylabel('Frequency')
    ax_grad_hist.set_title('Gradient Distribution')
    ax_grad_hist.legend(fontsize=8)
    ax_grad_hist.grid(True, alpha=0.3)
    ax_grad_hist.set_yscale('log')

    # Threshold test results
    ax_results = fig.add_subplot(gs[2, 2])
    texture_thresholds = [0.005, 0.01, 0.015, 0.02, 0.03]
    area_thresholds = [0.05, 0.10, 0.15, 0.20, 0.25]

    results_text = "Threshold Test Results:\n"
    results_text += "=" * 35 + "\n"

    for tex_thresh in texture_thresholds:
        for area_thresh in area_thresholds:
            has_tissue = AutofocusUtils.has_sufficient_tissue(
                image, tex_thresh, area_thresh, modality=modality
            )
            status = "✓ PASS" if has_tissue else "✗ FAIL"
            results_text += f"tex={tex_thresh:.3f}, area={area_thresh:.2f} → {status}\n"

    ax_results.text(0.05, 0.95, results_text, fontsize=8, family='monospace',
                    verticalalignment='top', transform=ax_results.transAxes)
    ax_results.axis('off')

    plt.suptitle(f'Tissue Detection Analysis - Modality: {modality.upper()}',
                 fontsize=16, fontweight='bold')

    return fig


def test_single_image(image_path: str, modality: str = "ppm", save_debug: bool = True):
    """
    Test tissue detection on a single image and provide detailed feedback.
    """
    print(f"\n{'='*70}")
    print(f"TISSUE DETECTION DEBUG TEST")
    print(f"{'='*70}")
    print(f"Image: {image_path}")
    print(f"Modality: {modality}")
    print(f"{'='*70}\n")

    # Load image
    image = tf.imread(image_path)
    print(f"Image shape: {image.shape}")
    print(f"Image dtype: {image.dtype}")
    print(f"Intensity range: {image.min()} - {image.max()}")

    if len(image.shape) == 3:
        print(f"RGB means: R={image[:,:,0].mean():.1f}, G={image[:,:,1].mean():.1f}, B={image[:,:,2].mean():.1f}")

    print("\n" + "="*70)
    print("RUNNING COMPREHENSIVE ANALYSIS")
    print("="*70 + "\n")

    # Run the test function
    analysis = AutofocusUtils.test_tissue_detection(
        image,
        modality=modality,
        texture_thresholds=[0.005, 0.01, 0.015, 0.02, 0.03, 0.05],
        area_thresholds=[0.05, 0.10, 0.15, 0.20, 0.25],
        show_analysis=True,
        logger=None
    )

    # Print recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    recs = analysis['recommendations']
    print(f"Best tissue mask: {recs['best_tissue_mask']}")
    print(f"Suggested texture threshold: {recs['suggested_texture_threshold']:.4f}")
    print(f"Suggested area threshold: {recs['suggested_area_threshold']:.3f}")
    print(f"Has good contrast: {recs['has_good_contrast']}")
    print(f"Intensity range: {recs['intensity_range']}")

    # Create visualization
    print("\n" + "="*70)
    print("CREATING VISUALIZATION")
    print("="*70)

    fig = visualize_tissue_detection(image, modality)

    if save_debug:
        debug_path = pathlib.Path(image_path).parent / f"tissue_detection_debug_{pathlib.Path(image_path).stem}.png"
        fig.savefig(debug_path, dpi=150, bbox_inches='tight')
        print(f"Saved debug visualization to: {debug_path}")

    plt.show()

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    # Test with current default settings
    has_tissue_default = AutofocusUtils.has_sufficient_tissue(
        image, texture_threshold=0.015, tissue_area_threshold=0.15, modality=modality
    )

    print(f"Current default settings (texture=0.015, area=0.15):")
    print(f"  Result: {'PASS ✓' if has_tissue_default else 'FAIL ✗'}")

    # Test with recommended settings
    has_tissue_rec = AutofocusUtils.has_sufficient_tissue(
        image,
        texture_threshold=recs['suggested_texture_threshold'],
        tissue_area_threshold=recs['suggested_area_threshold'],
        modality=modality
    )

    print(f"\nRecommended settings (texture={recs['suggested_texture_threshold']:.4f}, area={recs['suggested_area_threshold']:.3f}):")
    print(f"  Result: {'PASS ✓' if has_tissue_rec else 'FAIL ✗'}")

    print("\n" + "="*70)
    print("TO USE THESE SETTINGS IN YOUR ACQUISITION:")
    print("="*70)
    print("\nIn qp_acquisition.py, modify the has_sufficient_tissue call around line 474:")
    print(f"    has_tissue, tissue_stats = AutofocusUtils.has_sufficient_tissue(")
    print(f"        test_img,")
    print(f"        texture_threshold={recs['suggested_texture_threshold']:.4f},  # <-- Add this")
    print(f"        tissue_area_threshold={recs['suggested_area_threshold']:.3f},  # <-- Add this")
    print(f"        modality=modality,")
    print(f"        logger=logger,")
    print(f"        return_stats=True")
    print(f"    )")
    print("\n" + "="*70 + "\n")


def find_test_images(acquisition_dir: str, max_images: int = 5):
    """
    Find images from a failed acquisition to test.
    Looks for raw images that were acquired during failed autofocus.
    """
    print(f"\nSearching for test images in: {acquisition_dir}")

    acq_path = pathlib.Path(acquisition_dir)

    # Look for raw images in typical structure
    raw_dirs = list(acq_path.rglob("Raw"))

    if not raw_dirs:
        print("No 'Raw' directories found. Looking for .tif files...")
        tif_files = list(acq_path.rglob("*.tif"))[:max_images]
        return tif_files

    # Collect images from Raw directories
    test_images = []
    for raw_dir in raw_dirs[:max_images]:
        # Get angle subdirectories (typically 90.0 for autofocus)
        angle_dirs = [d for d in raw_dir.iterdir() if d.is_dir()]
        for angle_dir in angle_dirs:
            tif_files = list(angle_dir.glob("*.tif"))
            if tif_files:
                test_images.append(tif_files[0])  # Take first image from each angle
                if len(test_images) >= max_images:
                    break
        if len(test_images) >= max_images:
            break

    return test_images


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python test_tissue_detection_debug.py <path/to/image.tif>")
        print("  python test_tissue_detection_debug.py <path/to/acquisition/dir> --find")
        print("\nExamples:")
        print("  # Test specific image")
        print("  python test_tissue_detection_debug.py /path/to/image.tif")
        print("")
        print("  # Find and test images from failed acquisition")
        print("  python test_tissue_detection_debug.py D:/2025QPSC/data/ExistingImageTrial/ppm_10x_1 --find")
        sys.exit(1)

    input_path = sys.argv[1]

    # Check if we should search for images
    if len(sys.argv) > 2 and sys.argv[2] == "--find":
        print("Finding test images from acquisition directory...")
        test_images = find_test_images(input_path, max_images=5)

        if not test_images:
            print(f"No test images found in {input_path}")
            sys.exit(1)

        print(f"\nFound {len(test_images)} test images:")
        for idx, img_path in enumerate(test_images):
            print(f"  {idx+1}. {img_path}")

        # Test first image
        print(f"\n{'='*70}")
        print(f"Testing first image: {test_images[0]}")
        print(f"{'='*70}")

        test_single_image(str(test_images[0]), modality="ppm", save_debug=True)

        # Offer to test more
        if len(test_images) > 1:
            response = input(f"\nTest additional images? (y/n): ")
            if response.lower() == 'y':
                for img_path in test_images[1:]:
                    test_single_image(str(img_path), modality="ppm", save_debug=True)
    else:
        # Test single image directly
        test_single_image(input_path, modality="ppm", save_debug=True)
