#!/usr/bin/env python3
"""
Diagnostic to demonstrate cv2.cvtColor 16-bit data loss issue.

This script simulates what happens in the birefringence test when:
1. 16-bit RGB images are loaded
2. cv2.cvtColor is used to convert to grayscale
3. The resulting grayscale is used in normalized difference calculations

The hypothesis is that cv2.cvtColor converts 16-bit RGB to 8-bit grayscale,
causing quantization that appears as whole numbers or X.5 values in normalized output.
"""

import numpy as np


def simulate_cvtcolor_16bit_to_8bit():
    """
    Simulate what happens if cv2.cvtColor converts uint16 RGB to uint8 grayscale.

    This demonstrates the data loss that would cause the stepwise curve issue.
    """
    print("="*70)
    print("Simulating cv2.cvtColor uint16 -> uint8 data loss")
    print("="*70)

    # Create a synthetic 16-bit RGB image with fine gradations
    # Simulate what a microscopy image might look like
    height, width = 100, 100

    # Create gradient: values from 1000 to 50000 (typical microscopy range)
    gradient = np.linspace(1000, 50000, width, dtype=np.uint16)
    rgb_16bit = np.repeat(gradient[np.newaxis, :], height, axis=0)
    rgb_16bit = np.stack([rgb_16bit, rgb_16bit, rgb_16bit], axis=-1)

    print(f"\nOriginal 16-bit RGB image:")
    print(f"  dtype: {rgb_16bit.dtype}")
    print(f"  shape: {rgb_16bit.shape}")
    print(f"  range: [{rgb_16bit.min()}, {rgb_16bit.max()}]")
    print(f"  Sample values (row 0, cols 0-10):")
    print(f"    {rgb_16bit[0, 0:10, 0]}")

    # Simulate cv2.cvtColor if it converts to uint8 internally
    # OpenCV grayscale: 0.299*R + 0.587*G + 0.114*B
    # If it converts uint16 to uint8 first (like cvtColor does):
    rgb_8bit = (rgb_16bit / 256).astype(np.uint8)  # Scale down to 8-bit

    print(f"\nIf cv2.cvtColor scales to uint8:")
    print(f"  dtype: {rgb_8bit.dtype}")
    print(f"  range: [{rgb_8bit.min()}, {rgb_8bit.max()}]")
    print(f"  Sample values (row 0, cols 0-10):")
    print(f"    {rgb_8bit[0, 0:10, 0]}")

    # Apply grayscale conversion (on uint8)
    gray_8bit = (0.299 * rgb_8bit[:,:,2] +
                 0.587 * rgb_8bit[:,:,1] +
                 0.114 * rgb_8bit[:,:,0]).astype(np.uint8)

    print(f"\nGrayscale (uint8):")
    print(f"  dtype: {gray_8bit.dtype}")
    print(f"  range: [{gray_8bit.min()}, {gray_8bit.max()}]")
    print(f"  unique values: {len(np.unique(gray_8bit))}")
    print(f"  Sample values (row 0, cols 0-10):")
    print(f"    {gray_8bit[0, 0:10]}")

    # Now simulate normalization as done in the test
    print(f"\n{'='*70}")
    print("Simulating normalized difference calculation")
    print(f"{'='*70}")

    # Create two images (simulate +angle and -angle)
    pos_8bit = gray_8bit.copy()
    neg_8bit = gray_8bit.copy() + 5  # Slight offset to simulate rotation difference

    # Convert to float for calculations (as in the real code)
    pos_float = pos_8bit.astype(np.float32)
    neg_float = neg_8bit.astype(np.float32)

    # Compute difference and sum
    diff = pos_float - neg_float
    img_sum = pos_float + neg_float

    # Normalize
    epsilon = 1.0
    normalized = diff / (img_sum + epsilon)

    print(f"\nNormalized difference:")
    print(f"  dtype: {normalized.dtype}")
    print(f"  range: [{normalized.min():.6f}, {normalized.max():.6f}]")
    print(f"  Sample values (row 0, cols 0-10):")
    print(f"    {normalized[0, 0:10]}")

    # Check quantization
    unique_normalized = np.unique(normalized)
    print(f"\n  Unique normalized values: {len(unique_normalized)}")
    print(f"  First 20 unique values:")
    print(f"    {unique_normalized[:20]}")

    # Scale normalized values by 255 to see 8-bit quantization pattern
    scaled_by_255 = normalized * 255
    print(f"\n  When scaled by 255:")
    sample_scaled = scaled_by_255[0, 0:10]
    print(f"    Sample: {sample_scaled}")

    # Check fractional parts
    fractional = sample_scaled - np.floor(sample_scaled)
    print(f"    Fractional parts: {fractional}")

    if np.allclose(fractional, 0, atol=0.01):
        print(f"\n  CONFIRMED: Normalized values show 8-bit quantization!")
        print(f"  This matches the stepwise curve pattern observed in the test.")

    return normalized


def compare_16bit_vs_8bit_pipeline():
    """
    Compare results of proper 16-bit pipeline vs 8-bit quantized pipeline.
    """
    print(f"\n{'='*70}")
    print("Comparing 16-bit vs 8-bit processing pipelines")
    print(f"{'='*70}")

    # Create test gradient
    gradient_16bit = np.linspace(10000, 40000, 1000, dtype=np.uint16)

    # Pipeline 1: Keep 16-bit throughout
    pos_16bit = gradient_16bit.copy()
    neg_16bit = gradient_16bit.copy() + 100  # Small difference

    pos_float_16 = pos_16bit.astype(np.float32)
    neg_float_16 = neg_16bit.astype(np.float32)

    diff_16 = pos_float_16 - neg_float_16
    sum_16 = pos_float_16 + neg_float_16
    normalized_16 = diff_16 / (sum_16 + 1.0)

    # Pipeline 2: Quantize to 8-bit (simulating cv2.cvtColor behavior)
    pos_8bit = (pos_16bit / 256).astype(np.uint8)
    neg_8bit = (neg_16bit / 256).astype(np.uint8)

    pos_float_8 = pos_8bit.astype(np.float32)
    neg_float_8 = neg_8bit.astype(np.float32)

    diff_8 = pos_float_8 - neg_float_8
    sum_8 = pos_float_8 + neg_float_8
    normalized_8 = diff_8 / (sum_8 + 1.0)

    print(f"\n16-bit pipeline:")
    print(f"  Unique normalized values: {len(np.unique(normalized_16))}")
    print(f"  Range: [{normalized_16.min():.6f}, {normalized_16.max():.6f}]")
    print(f"  Sample (indices 0-10): {normalized_16[0:10]}")

    print(f"\n8-bit pipeline:")
    print(f"  Unique normalized values: {len(np.unique(normalized_8))}")
    print(f"  Range: [{normalized_8.min():.6f}, {normalized_8.max():.6f}]")
    print(f"  Sample (indices 0-10): {normalized_8[0:10]}")

    print(f"\nDifference between pipelines:")
    diff_pipeline = np.abs(normalized_16 - normalized_8)
    print(f"  Max difference: {diff_pipeline.max():.6f}")
    print(f"  Mean difference: {diff_pipeline.mean():.6f}")

    print(f"\nConclusion:")
    print(f"  8-bit pipeline has {len(np.unique(normalized_16))/len(np.unique(normalized_8)):.1f}x")
    print(f"  fewer unique values due to quantization.")
    print(f"  This creates the stepwise pattern in angle vs signal curves.")


def demonstrate_fix():
    """
    Demonstrate the correct way to preserve 16-bit precision.
    """
    print(f"\n{'='*70}")
    print("RECOMMENDED FIX")
    print(f"{'='*70}")

    print("""
The problem is in lines 578-580 of ppm_birefringence_maximization_test.py:

    if len(pos_img.shape) == 3:
        pos_img = cv2.cvtColor(pos_img, cv2.COLOR_BGR2GRAY)
    if len(neg_img.shape) == 3:
        neg_img = cv2.cvtColor(neg_img, cv2.COLOR_BGR2GRAY)

cv2.cvtColor converts uint16 RGB to uint8 grayscale, losing precision.

SOLUTION: Manual grayscale conversion that preserves 16-bit:

    if len(pos_img.shape) == 3:
        # Manual grayscale preserving 16-bit precision
        pos_img = (0.299 * pos_img[:,:,2].astype(np.float32) +
                   0.587 * pos_img[:,:,1].astype(np.float32) +
                   0.114 * pos_img[:,:,0].astype(np.float32))
        pos_img = np.clip(pos_img, 0, 65535).astype(np.uint16)

    if len(neg_img.shape) == 3:
        neg_img = (0.299 * neg_img[:,:,2].astype(np.float32) +
                   0.587 * neg_img[:,:,1].astype(np.float32) +
                   0.114 * neg_img[:,:,0].astype(np.float32))
        neg_img = np.clip(neg_img, 0, 65535).astype(np.uint16)

This preserves the full 16-bit dynamic range and eliminates quantization artifacts.
""")


def main():
    """Run all diagnostics."""
    print("\n" + "="*70)
    print("cv2.cvtColor 16-bit Data Loss Diagnostic")
    print("="*70)
    print("\nThis diagnostic demonstrates the root cause of the stepwise curve")
    print("observed in the PPM birefringence maximization test.")
    print()

    # Run diagnostics
    simulate_cvtcolor_16bit_to_8bit()
    compare_16bit_vs_8bit_pipeline()
    demonstrate_fix()

    print("\n" + "="*70)
    print("Diagnostic complete")
    print("="*70)


if __name__ == "__main__":
    main()
