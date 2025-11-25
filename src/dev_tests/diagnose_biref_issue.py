#!/usr/bin/env python3
"""
Diagnose birefringence background brightness issue.
Checks for mismatches between background images and settings files.
"""

import sys
import pathlib
import yaml
import tifffile as tf
import numpy as np

def diagnose(background_base: pathlib.Path):
    """Check for issues causing bright birefringence backgrounds."""

    print("="*70)
    print("BIREFRINGENCE BACKGROUND DIAGNOSTIC")
    print("="*70)

    # Find the PPM background directory
    ppm_dirs = list(background_base.glob("*/ppm/*"))
    if not ppm_dirs:
        ppm_dirs = list(background_base.glob("ppm/*"))
    if not ppm_dirs:
        print(f"\nERROR: No PPM background directories found in {background_base}")
        return

    for ppm_dir in ppm_dirs[:1]:  # Check first one
        print(f"\nChecking: {ppm_dir}")
        print("-"*70)

        # Check for background_settings.yml
        settings_file = ppm_dir / "background_settings.yml"
        if settings_file.exists():
            with open(settings_file) as f:
                settings = yaml.safe_load(f)

            print("\n1. BACKGROUND SETTINGS FILE:")
            print(f"   File: {settings_file}")
            print(f"   Modality: {settings.get('modality')}")
            print(f"   Objective: {settings.get('objective')}")

            if 'angle_exposures' in settings:
                print(f"   Angles defined: {len(settings['angle_exposures'])}")
                for ae in settings['angle_exposures']:
                    print(f"     {ae['angle']:>6}deg: {ae['exposure_ms']:>8.2f}ms")
        else:
            print(f"\n1. BACKGROUND SETTINGS FILE: NOT FOUND at {settings_file}")

        # Check for actual background images
        print("\n2. ACTUAL BACKGROUND IMAGE FILES:")
        bg_files = sorted(ppm_dir.glob("*.tif")) + sorted(ppm_dir.glob("*/background.tif"))

        if not bg_files:
            print("   No background images found!")

        bg_stats = {}
        for bg_file in bg_files:
            # Extract angle from filename
            if bg_file.name == "background.tif":
                angle_str = bg_file.parent.name
            else:
                angle_str = bg_file.stem

            try:
                angle = float(angle_str)
                img = tf.imread(str(bg_file))
                mean_int = img.mean()
                bg_stats[angle] = mean_int

                print(f"   {angle:>6}deg: mean={mean_int:>6.1f}, file={bg_file.name}")
            except:
                print(f"   ??? : Could not parse angle from {bg_file}")

        # Check for +7/-7 pair
        print("\n3. BIREFRINGENCE PAIR ANALYSIS:")
        if 7.0 in bg_stats and -7.0 in bg_stats:
            diff = abs(bg_stats[7.0] - bg_stats[-7.0])
            print(f"   +7deg background mean: {bg_stats[7.0]:.1f}")
            print(f"   -7deg background mean: {bg_stats[-7.0]:.1f}")
            print(f"   Difference: {diff:.1f}")

            if diff > 10:
                print(f"\n   WARNING: Backgrounds differ by {diff:.1f}!")
                print(f"   Expected: < 5 (should be acquired at same target intensity)")
                print(f"   This will cause bright backgrounds in birefringence images")
            else:
                print(f"   OK: Backgrounds are well-matched (diff < 10)")

            # Simulate birefringence
            for bg_file in bg_files:
                angle_str = bg_file.stem if bg_file.name != "background.tif" else bg_file.parent.name
                try:
                    if float(angle_str) == 7.0:
                        pos_img = tf.imread(str(bg_file))
                    elif float(angle_str) == -7.0:
                        neg_img = tf.imread(str(bg_file))
                except:
                    pass

            if 'pos_img' in locals() and 'neg_img' in locals():
                # Uncorrected birefringence
                diff_img = np.abs(pos_img.astype(np.int16) - neg_img.astype(np.int16))
                if len(diff_img.shape) == 3:
                    biref_sim = np.sum(diff_img, axis=2)
                else:
                    biref_sim = diff_img

                print(f"\n4. SIMULATED BIREFRINGENCE BACKGROUND (uncorrected):")
                print(f"   Mean: {biref_sim.mean():.1f}")
                print(f"   Range: [{biref_sim.min()}, {biref_sim.max()}]")

                # Corrected birefringence (with scaling)
                pos_mean = pos_img.astype(np.float32).mean()
                neg_mean = neg_img.astype(np.float32).mean()
                scale_factor = pos_mean / neg_mean if neg_mean > 0 else 1.0

                neg_scaled = neg_img.astype(np.float32) * scale_factor
                diff_corrected = np.abs(pos_img.astype(np.float32) - neg_scaled)
                if len(diff_corrected.shape) == 3:
                    biref_corrected = np.sum(diff_corrected, axis=2)
                else:
                    biref_corrected = diff_corrected

                print(f"\n5. SIMULATED BIREFRINGENCE BACKGROUND (with scale correction):")
                print(f"   Scale factor applied to -7deg: {scale_factor:.4f}")
                print(f"   Mean: {biref_corrected.mean():.1f}")
                print(f"   Range: [{biref_corrected.min():.0f}, {biref_corrected.max():.0f}]")
                print(f"   Improvement: {biref_sim.mean():.1f} -> {biref_corrected.mean():.1f} ({(1 - biref_corrected.mean()/biref_sim.mean())*100:.0f}% reduction)")

                if biref_corrected.mean() > 20:
                    print(f"\n   WARNING: Corrected birefringence background still elevated!")
                    print(f"   This may indicate per-channel color differences")
                    print(f"   Consider checking white balance settings")
                elif biref_corrected.mean() > 10:
                    print(f"\n   OK: Birefringence background is acceptable with correction")
                else:
                    print(f"\n   GOOD: Birefringence background is well-matched")
        else:
            print(f"   WARNING: Missing +7 or -7 background image")
            print(f"   Found angles: {sorted(bg_stats.keys())}")

        # Check if settings match files
        print("\n5. SETTINGS vs FILES CONSISTENCY:")
        if settings_file.exists():
            settings_angles = {ae['angle'] for ae in settings.get('angle_exposures', [])}
            file_angles = set(bg_stats.keys())

            if settings_angles == file_angles:
                print("   OK: Settings file matches background files")
            else:
                print("   WARNING: MISMATCH!")
                print(f"   Settings has: {sorted(settings_angles)}")
                print(f"   Files have: {sorted(file_angles)}")
                if 5.0 in settings_angles or -5.0 in settings_angles:
                    print("\n   LIKELY ISSUE: Settings still reference 5/-5 degrees")
                    print("   but background images are for 7/-7 degrees!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnose_biref_issue.py /path/to/background/base/folder")
        print("\nExample: python diagnose_biref_issue.py D:/backgrounds")
        sys.exit(1)

    bg_base = pathlib.Path(sys.argv[1])
    if not bg_base.exists():
        print(f"ERROR: Directory not found: {bg_base}")
        sys.exit(1)

    diagnose(bg_base)
