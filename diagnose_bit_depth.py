#!/usr/bin/env python3
"""
Diagnostic script to analyze TIFF bit depth and investigate data loss hypothesis.

This script examines raw TIFF images to determine:
1. Whether images are 8-bit or 16-bit
2. Actual min/max pixel values
3. Whether cv2.cvtColor preserves 16-bit data
4. Whether difference images show quantization artifacts
"""

import cv2
import numpy as np
from pathlib import Path


def analyze_image(filepath, name="Image"):
    """Analyze a single image file for bit depth and value range."""
    print(f"\n{'='*70}")
    print(f"Analyzing: {name}")
    print(f"File: {filepath}")
    print(f"{'='*70}")

    # Load with default flags
    img_default = cv2.imread(str(filepath))
    if img_default is not None:
        print(f"\nDefault cv2.imread():")
        print(f"  dtype: {img_default.dtype}")
        print(f"  shape: {img_default.shape}")
        print(f"  min value: {img_default.min()}")
        print(f"  max value: {img_default.max()}")
        if len(img_default.shape) == 3:
            print(f"  channels: {img_default.shape[2]} (RGB/BGR)")

    # Load with IMREAD_UNCHANGED to preserve original bit depth
    img_unchanged = cv2.imread(str(filepath), cv2.IMREAD_UNCHANGED)
    if img_unchanged is not None:
        print(f"\ncv2.imread(IMREAD_UNCHANGED):")
        print(f"  dtype: {img_unchanged.dtype}")
        print(f"  shape: {img_unchanged.shape}")
        print(f"  min value: {img_unchanged.min()}")
        print(f"  max value: {img_unchanged.max()}")
        if len(img_unchanged.shape) == 3:
            print(f"  channels: {img_unchanged.shape[2]} (RGB/BGR)")

    # Check if loading as IMREAD_UNCHANGED makes a difference
    if img_default is not None and img_unchanged is not None:
        if img_default.dtype != img_unchanged.dtype:
            print(f"\n  WARNING: Default loading changed dtype!")
            print(f"  Default: {img_default.dtype}, Unchanged: {img_unchanged.dtype}")
        else:
            print(f"\n  OK: Both loading methods produce same dtype")

    return img_unchanged


def test_cvtcolor_16bit():
    """Test whether cv2.cvtColor preserves 16-bit data."""
    print(f"\n{'='*70}")
    print("Testing cv2.cvtColor with 16-bit RGB image")
    print(f"{'='*70}")

    # Load a 16-bit RGB image
    test_file = Path("/home/msnelson/QPSC_Project/OtherDocuments/test_20251210_190220/pos_0.00.tif")
    img_rgb = cv2.imread(str(test_file), cv2.IMREAD_UNCHANGED)

    if img_rgb is None:
        print("ERROR: Could not load test image")
        return

    print(f"\nOriginal RGB image:")
    print(f"  dtype: {img_rgb.dtype}")
    print(f"  shape: {img_rgb.shape}")
    print(f"  min value: {img_rgb.min()}")
    print(f"  max value: {img_rgb.max()}")

    # Convert to grayscale using cv2.cvtColor
    if len(img_rgb.shape) == 3 and img_rgb.shape[2] == 3:
        gray_cvt = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

        print(f"\nAfter cv2.cvtColor(COLOR_BGR2GRAY):")
        print(f"  dtype: {gray_cvt.dtype}")
        print(f"  shape: {gray_cvt.shape}")
        print(f"  min value: {gray_cvt.min()}")
        print(f"  max value: {gray_cvt.max()}")

        if gray_cvt.dtype != img_rgb.dtype:
            print(f"\n  CRITICAL: cv2.cvtColor CHANGED dtype from {img_rgb.dtype} to {gray_cvt.dtype}")
            print(f"  This is the source of 8-bit quantization!")
        else:
            print(f"\n  OK: cv2.cvtColor preserved {img_rgb.dtype}")

        # Manual grayscale conversion to compare
        if img_rgb.dtype == np.uint16:
            # OpenCV uses: 0.299*R + 0.587*G + 0.114*B
            gray_manual = (0.299 * img_rgb[:,:,2].astype(np.float32) +
                          0.587 * img_rgb[:,:,1].astype(np.float32) +
                          0.114 * img_rgb[:,:,0].astype(np.float32))
            gray_manual = gray_manual.astype(np.uint16)

            print(f"\nManual grayscale conversion (uint16):")
            print(f"  dtype: {gray_manual.dtype}")
            print(f"  min value: {gray_manual.min()}")
            print(f"  max value: {gray_manual.max()}")

            # Compare
            diff = np.abs(gray_manual.astype(np.int32) - gray_cvt.astype(np.int32))
            print(f"\nDifference between manual and cv2.cvtColor:")
            print(f"  max difference: {diff.max()}")
            print(f"  mean difference: {diff.mean():.2f}")
    else:
        print(f"  Image is not 3-channel, skipping cvtColor test")


def check_quantization(img, name="Image"):
    """Check if image values show quantization artifacts (e.g., only whole numbers or X.5 values)."""
    print(f"\n{'='*70}")
    print(f"Checking quantization artifacts: {name}")
    print(f"{'='*70}")

    # Get unique values (sample if too many)
    flat = img.flatten()
    unique_vals = np.unique(flat)

    print(f"\nUnique values: {len(unique_vals)}")
    print(f"First 20 unique values: {unique_vals[:20]}")
    print(f"Last 20 unique values: {unique_vals[-20:]}")

    # Check if values are whole numbers (for float images)
    if img.dtype in [np.float32, np.float64]:
        is_whole = np.all(unique_vals == np.round(unique_vals))
        print(f"\nAll values are whole numbers: {is_whole}")

        # Check for X.5 pattern
        fractional_parts = unique_vals - np.floor(unique_vals)
        unique_fractions = np.unique(fractional_parts)
        print(f"Unique fractional parts: {unique_fractions[:20]}")

        if len(unique_fractions) <= 2:
            print(f"  WARNING: Limited fractional values suggest 8-bit quantization!")

    # For integer images, check value distribution
    if img.dtype in [np.uint8, np.uint16]:
        print(f"\nValue distribution:")
        hist, bins = np.histogram(flat, bins=min(256, len(unique_vals)))
        print(f"  Histogram bins: {len(hist)}")
        print(f"  Non-zero bins: {np.count_nonzero(hist)}")

        # Check for 256-level pattern in uint16 images
        if img.dtype == np.uint16:
            # Check if values cluster around 256-level steps
            val_mod_256 = unique_vals % 256
            unique_mods = np.unique(val_mod_256)
            print(f"  Unique (value % 256): {len(unique_mods)}")
            if len(unique_mods) < 10:
                print(f"  WARNING: Values cluster at 256-step intervals!")
                print(f"  This suggests 8-bit source data scaled to 16-bit")


def main():
    """Main diagnostic routine."""
    base_dir = Path("/home/msnelson/QPSC_Project/OtherDocuments/test_20251210_190220")

    print("="*70)
    print("TIFF Bit Depth Diagnostic Analysis")
    print("="*70)

    # 1. Analyze a positive angle image
    pos_file = base_dir / "pos_0.00.tif"
    pos_img = analyze_image(pos_file, "Positive 0.00 degrees")

    # 2. Analyze a negative angle image
    neg_file = base_dir / "neg_0.00.tif"
    neg_img = analyze_image(neg_file, "Negative 0.00 degrees")

    # 3. Analyze a difference image
    diff_file = base_dir / "differences" / "diff_abs_0.00.tif"
    diff_img = analyze_image(diff_file, "Absolute Difference 0.00 degrees")

    # 4. Test cv2.cvtColor behavior with 16-bit images
    test_cvtcolor_16bit()

    # 5. Check for quantization in difference image
    if diff_img is not None:
        check_quantization(diff_img, "Difference Image")

    # 6. Additional test: check normalized values
    if pos_img is not None and neg_img is not None:
        print(f"\n{'='*70}")
        print("Testing normalization behavior")
        print(f"{'='*70}")

        # Simulate what happens during difference calculation
        if pos_img.dtype == np.uint16:
            print("\nSimulating grayscale conversion + difference:")

            # If cvtColor converts to uint8, simulate that
            pos_gray = cv2.cvtColor(pos_img, cv2.COLOR_BGR2GRAY)
            neg_gray = cv2.cvtColor(neg_img, cv2.COLOR_BGR2GRAY)

            print(f"Pos gray: dtype={pos_gray.dtype}, range=[{pos_gray.min()}, {pos_gray.max()}]")
            print(f"Neg gray: dtype={neg_gray.dtype}, range=[{neg_gray.min()}, {neg_gray.max()}]")

            # Normalize to 0-1
            if pos_gray.dtype == np.uint8:
                pos_norm = pos_gray.astype(np.float32) / 255.0
                neg_norm = neg_gray.astype(np.float32) / 255.0
                print(f"\nNormalized by 255 (uint8):")
            else:
                pos_norm = pos_gray.astype(np.float32) / 65535.0
                neg_norm = neg_gray.astype(np.float32) / 65535.0
                print(f"\nNormalized by 65535 (uint16):")

            print(f"Pos normalized: range=[{pos_norm.min():.6f}, {pos_norm.max():.6f}]")
            print(f"Neg normalized: range=[{neg_norm.min():.6f}, {neg_norm.max():.6f}]")

            # Sample some values
            sample_pos = pos_norm[100:105, 100:105].flatten()
            print(f"\nSample normalized pos values:")
            print(f"  {sample_pos}")

            # Check if values are quantized
            unique_sample = np.unique(sample_pos)
            if len(unique_sample) < len(sample_pos) / 2:
                print(f"  WARNING: High repetition in normalized values!")

            # Check fractional parts
            scaled = sample_pos * 255  # What you'd get from uint8
            fractions = scaled - np.floor(scaled)
            print(f"\nFractional parts after scaling by 255:")
            print(f"  {fractions}")
            if np.allclose(fractions, 0):
                print(f"  CONFIRMED: Values are quantized to 8-bit levels!")

    print("\n" + "="*70)
    print("Diagnostic analysis complete")
    print("="*70)


if __name__ == "__main__":
    main()
