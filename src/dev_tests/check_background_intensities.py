#!/usr/bin/env python3
"""
Quick diagnostic to check background image intensities.
Usage: python check_background_intensities.py /path/to/background/folder
"""

import sys
import pathlib
import tifffile as tf
import numpy as np

def check_backgrounds(background_dir: pathlib.Path):
    """Check mean intensities of all background images."""
    print(f"Checking backgrounds in: {background_dir}\n")

    angles = [0.0, 7.0, -7.0, 90.0]

    for angle in angles:
        # Try different file locations
        paths_to_try = [
            background_dir / f"{angle}.tif",
            background_dir / str(angle) / "background.tif",
            background_dir / "ppm" / str(angle) / "background.tif",
        ]

        for path in paths_to_try:
            if path.exists():
                img = tf.imread(str(path))
                mean_intensity = img.mean()
                per_channel = img.mean(axis=(0,1)) if len(img.shape) == 3 else None

                print(f"Angle {angle:>5}deg: {path}")
                print(f"  Mean intensity: {mean_intensity:.1f}")
                if per_channel is not None and len(per_channel) >= 3:
                    print(f"  Per channel (R,G,B): ({per_channel[0]:.1f}, {per_channel[1]:.1f}, {per_channel[2]:.1f})")
                print(f"  Shape: {img.shape}, dtype: {img.dtype}")
                print()
                break
        else:
            print(f"Angle {angle:>5}deg: NOT FOUND")
            print(f"  Tried: {[str(p) for p in paths_to_try]}")
            print()

    # Check for birefringence calculation consistency
    print("\n" + "="*60)
    print("BIREFRINGENCE CONSISTENCY CHECK")
    print("="*60)

    # Try to load +7 and -7
    pos_path = None
    neg_path = None

    for base_path in [background_dir, background_dir / "ppm"]:
        if not pos_path:
            for p in [base_path / "7.0.tif", base_path / "7.0" / "background.tif"]:
                if p.exists():
                    pos_path = p
                    break
        if not neg_path:
            for p in [base_path / "-7.0.tif", base_path / "-7.0" / "background.tif"]:
                if p.exists():
                    neg_path = p
                    break

    if pos_path and neg_path:
        pos_img = tf.imread(str(pos_path))
        neg_img = tf.imread(str(neg_path))

        print(f"\n+7deg background mean: {pos_img.mean():.1f}")
        print(f"-7deg background mean: {neg_img.mean():.1f}")
        print(f"Difference: {abs(pos_img.mean() - neg_img.mean()):.1f}")

        # Calculate what the birefringence background would be
        diff = np.abs(pos_img.astype(np.int16) - neg_img.astype(np.int16))
        if len(diff.shape) == 3:
            biref_sim = np.sum(diff, axis=2)
        else:
            biref_sim = diff

        print(f"\nSimulated birefringence background mean: {biref_sim.mean():.1f}")
        print(f"Simulated birefringence background range: [{biref_sim.min()}, {biref_sim.max()}]")

        if biref_sim.mean() > 20:
            print(f"\nWARNING: Birefringence background mean ({biref_sim.mean():.1f}) is too high!")
            print(f"Expected: Close to 0 (black background)")
            print(f"This indicates +7 and -7 backgrounds have different intensities")
    else:
        print(f"\nCould not find both +7 and -7 backgrounds for comparison")
        if pos_path:
            print(f"  Found +7: {pos_path}")
        if neg_path:
            print(f"  Found -7: {neg_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_background_intensities.py /path/to/background/folder")
        sys.exit(1)

    bg_dir = pathlib.Path(sys.argv[1])
    if not bg_dir.exists():
        print(f"Error: Directory does not exist: {bg_dir}")
        sys.exit(1)

    check_backgrounds(bg_dir)
