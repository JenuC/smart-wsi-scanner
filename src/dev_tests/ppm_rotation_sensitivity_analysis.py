#!/usr/bin/env python3
"""
PPM Rotation Sensitivity Analysis

This script analyzes the impact of angular deviations on Polarized Phase Microscopy (PPM)
image quality and birefringence calculations. It provides quantitative metrics for how
small angular errors affect:
1. Direct image differences at various angles
2. Birefringence image quality and pattern stability
3. Retardance and orientation calculations

Usage:
    python ppm_rotation_sensitivity_analysis.py <image_directory> [options]
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from typing import Dict, List, Tuple, Optional
import pandas as pd
from scipy import ndimage
from skimage import metrics
import json
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class PPMRotationAnalyzer:
    """Analyzes rotation sensitivity in PPM imaging."""

    def __init__(self, base_path: Path, output_dir: Path = None):
        """
        Initialize the analyzer.

        Args:
            base_path: Directory containing PPM images at different angles
            output_dir: Directory for output results (creates timestamped folder if None)
        """
        self.base_path = Path(base_path)
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = Path(f"ppm_rotation_analysis_{timestamp}")
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Standard PPM angles (in degrees)
        self.standard_angles = [0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 91]

        # Store loaded images
        self.images = {}
        self.image_shape = None

    def load_images(self, angle_pattern: str = "*_{angle}deg*.tif") -> Dict[float, np.ndarray]:
        """
        Load PPM images at different angles.

        Args:
            angle_pattern: Filename pattern with {angle} placeholder

        Returns:
            Dictionary mapping angles to image arrays
        """
        print("Loading PPM images...")

        # Try to find images with various naming conventions
        patterns = [
            angle_pattern,
            "*_{angle}degree*.tif",
            "*_angle{angle}*.tif",
            "*_{angle}*.tif"
        ]

        for angle in self.standard_angles:
            for pattern in patterns:
                search_pattern = pattern.replace("{angle}", str(angle))
                files = list(self.base_path.glob(search_pattern))
                if files:
                    # Load the first matching file
                    img = cv2.imread(str(files[0]), cv2.IMREAD_UNCHANGED)
                    if img is not None:
                        # Convert to float32 for calculations
                        img = img.astype(np.float32)
                        if len(img.shape) == 3:
                            # Convert to grayscale if color
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        self.images[angle] = img
                        if self.image_shape is None:
                            self.image_shape = img.shape
                        print(f"  Loaded {angle} degree image: {files[0].name}")
                        break

        print(f"Loaded {len(self.images)} images")
        return self.images

    def load_deviation_images(self) -> Dict[float, np.ndarray]:
        """
        Load PPM deviation test images in BGACQUIRE format: {angle}.tif (e.g., 45.05.tif, 44.0.tif)

        This is the format used by qp_acquisition.simple_background_collection().

        Returns:
            Dictionary mapping actual angles to image arrays
        """
        import re
        print("Loading PPM deviation images...")

        # Load BGACQUIRE format: {angle}.tif (e.g., 45.05.tif)
        angle_pattern = re.compile(r'^(-?\d+\.?\d*)\.tif$', re.IGNORECASE)

        for file_path in self.base_path.glob("*.tif"):
            match = angle_pattern.match(file_path.name)
            if match:
                try:
                    angle = float(match.group(1))
                    img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
                    if img is not None:
                        img = img.astype(np.float32)
                        if len(img.shape) == 3:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        self.images[angle] = img
                        if self.image_shape is None:
                            self.image_shape = img.shape
                        print(f"  Loaded {angle:.2f} deg image: {file_path.name}")
                except ValueError:
                    pass  # Not a valid angle filename

        print(f"Loaded {len(self.images)} deviation images")
        return self.images

    def simulate_angular_deviations(self, reference_angle: float = 7.0,
                                   deviations: List[float] = [0.1, 0.2, 0.3, 0.5, 1.0]) -> Dict:
        """
        Simulate images at slightly deviated angles through interpolation.

        Args:
            reference_angle: Reference angle in degrees
            deviations: List of angular deviations to test

        Returns:
            Dictionary of simulated images at deviated angles
        """
        if reference_angle not in self.images:
            print(f"Warning: Reference angle {reference_angle} not found in loaded images")
            return {}

        simulated = {}
        ref_image = self.images[reference_angle]

        # Find nearest neighbor angles for interpolation
        angles = sorted(self.images.keys())
        ref_idx = angles.index(reference_angle)

        for dev in deviations:
            target_angle = reference_angle + dev

            # Simple rotation simulation (small angle approximation)
            # In reality, this would be more complex due to polarizer physics
            rotation_matrix = cv2.getRotationMatrix2D(
                (ref_image.shape[1]//2, ref_image.shape[0]//2),
                dev, 1.0
            )
            rotated = cv2.warpAffine(ref_image, rotation_matrix,
                                    (ref_image.shape[1], ref_image.shape[0]))

            # Also try interpolation between neighboring angles if available
            if ref_idx < len(angles) - 1:
                next_angle = angles[ref_idx + 1]
                if next_angle in self.images:
                    # Linear interpolation weight
                    weight = dev / (next_angle - reference_angle)
                    interpolated = (1 - weight) * ref_image + weight * self.images[next_angle]
                    simulated[target_angle] = interpolated
                else:
                    simulated[target_angle] = rotated
            else:
                simulated[target_angle] = rotated

        return simulated

    def compute_image_differences(self, reference_angle: float = 7.0) -> pd.DataFrame:
        """
        Compute various difference metrics between reference and deviated images.

        Args:
            reference_angle: Reference angle for comparison

        Returns:
            DataFrame with difference metrics
        """
        results = []

        if reference_angle not in self.images:
            print(f"Error: Reference angle {reference_angle} not found")
            return pd.DataFrame()

        ref_image = self.images[reference_angle]

        # Test against actual angles
        for angle, img in self.images.items():
            if angle == reference_angle:
                continue

            # Compute various metrics
            mae = np.mean(np.abs(img - ref_image))
            mse = np.mean((img - ref_image) ** 2)
            rmse = np.sqrt(mse)

            # Normalized metrics (relative to image intensity range)
            img_range = ref_image.max() - ref_image.min()
            if img_range > 0:
                nmae = mae / img_range
                nrmse = rmse / img_range
            else:
                nmae = nmae = 0

            # Structural similarity
            ssim = metrics.structural_similarity(ref_image, img, data_range=img.max()-img.min())

            # Peak signal-to-noise ratio
            psnr = metrics.peak_signal_noise_ratio(ref_image, img, data_range=img.max()-img.min())

            # Pearson correlation
            correlation = np.corrcoef(ref_image.flatten(), img.flatten())[0, 1]

            results.append({
                'reference_angle': reference_angle,
                'comparison_angle': angle,
                'angular_difference': abs(angle - reference_angle),
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'normalized_mae': nmae,
                'normalized_rmse': nrmse,
                'ssim': ssim,
                'psnr': psnr,
                'correlation': correlation,
                'max_pixel_diff': float(np.max(np.abs(img - ref_image))),
                'std_diff': np.std(img - ref_image)
            })

        return pd.DataFrame(results)

    def compute_adjacent_differences(self) -> pd.DataFrame:
        """
        Compare each angle to its immediate neighbors to measure fine sensitivity.

        This answers: "How much does the image change between 7.00 and 7.05 degrees?"

        Returns:
            DataFrame with columns: angle1, angle2, delta_deg, mae, pct_change, ssim
        """
        results = []
        sorted_angles = sorted(self.images.keys())

        print(f"\nComputing adjacent angle differences for {len(sorted_angles)} angles...")

        for i in range(len(sorted_angles) - 1):
            angle1 = sorted_angles[i]
            angle2 = sorted_angles[i + 1]
            delta = angle2 - angle1

            img1 = self.images[angle1]
            img2 = self.images[angle2]

            # Compute metrics
            mae = float(np.mean(np.abs(img2 - img1)))
            img_range = max(img1.max() - img1.min(), 1)
            pct_change = (mae / img_range) * 100

            # SSIM for structural similarity
            try:
                ssim = metrics.structural_similarity(img1, img2, data_range=img1.max()-img1.min())
            except Exception:
                ssim = np.nan

            results.append({
                'angle1': angle1,
                'angle2': angle2,
                'delta_deg': delta,
                'mae': mae,
                'pct_change': pct_change,
                'ssim': ssim,
                'median_intensity_1': float(np.median(img1)),
                'median_intensity_2': float(np.median(img2)),
                'intensity_change': float(np.median(img2) - np.median(img1))
            })

        return pd.DataFrame(results)

    def analyze_fine_sensitivity(self, base_angles: List[float] = None) -> Dict:
        """
        Analyze sensitivity to fine angular changes around specific base angles.

        Groups angles by proximity to base angles (e.g., 7, 45) and computes
        statistics for fine deviations within each group.

        Args:
            base_angles: List of nominal angles to analyze (default: [7, 45])

        Returns:
            Dictionary with sensitivity statistics per base angle
        """
        if base_angles is None:
            base_angles = [7, 45]

        results = {}
        sorted_angles = sorted(self.images.keys())

        for base in base_angles:
            # Find angles within +/- 2 degrees of base
            nearby = [a for a in sorted_angles if abs(a - base) <= 2.0]

            if len(nearby) < 2:
                continue

            print(f"\n=== Fine sensitivity analysis around {base} deg ===")
            print(f"Found {len(nearby)} angles: {[f'{a:.2f}' for a in nearby]}")

            # Compare each angle to the base angle (or nearest to base)
            base_actual = min(nearby, key=lambda x: abs(x - base))
            if base_actual not in self.images:
                continue

            base_img = self.images[base_actual]
            group_results = []

            for angle in nearby:
                if angle == base_actual:
                    continue

                img = self.images[angle]
                deviation = angle - base_actual

                mae = float(np.mean(np.abs(img - base_img)))
                img_range = max(base_img.max() - base_img.min(), 1)
                pct_change = (mae / img_range) * 100

                group_results.append({
                    'deviation': deviation,
                    'angle': angle,
                    'mae': mae,
                    'pct_change': pct_change
                })

                print(f"  {base_actual:.2f} -> {angle:.2f} (delta={deviation:+.2f}): "
                      f"MAE={mae:.2f}, {pct_change:.3f}% change")

            if group_results:
                # Compute sensitivity rate (% change per degree)
                deviations = [abs(r['deviation']) for r in group_results]
                pct_changes = [r['pct_change'] for r in group_results]

                # Linear fit for sensitivity rate
                if len(deviations) > 1:
                    slope = np.polyfit(deviations, pct_changes, 1)[0]
                else:
                    slope = pct_changes[0] / deviations[0] if deviations[0] != 0 else 0

                results[base] = {
                    'base_angle': base_actual,
                    'n_samples': len(group_results),
                    'sensitivity_pct_per_deg': slope,
                    'details': group_results
                }

                print(f"  --> Sensitivity: ~{slope:.2f}% intensity change per degree")

        return results

    def compute_birefringence(self, angles: List[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute birefringence (retardance and orientation) from PPM images.

        Args:
            angles: List of angles to use (defaults to all available)

        Returns:
            Tuple of (retardance, orientation) arrays
        """
        if angles is None:
            angles = sorted(self.images.keys())

        # Filter to available angles
        angles = [a for a in angles if a in self.images]

        if len(angles) < 3:
            print("Error: Need at least 3 angles for birefringence calculation")
            return None, None

        # Stack images
        image_stack = np.array([self.images[a] for a in angles])
        angles_rad = np.array(angles) * np.pi / 180

        # Fourier analysis for retardance and orientation
        # This is a simplified version - actual implementation depends on specific PPM setup
        n_pixels = image_stack.shape[1] * image_stack.shape[2]

        # Compute Fourier coefficients
        a0 = np.mean(image_stack, axis=0)
        a2 = np.zeros_like(a0)
        b2 = np.zeros_like(a0)

        for i, angle in enumerate(angles_rad):
            a2 += (2/len(angles)) * image_stack[i] * np.cos(2 * angle)
            b2 += (2/len(angles)) * image_stack[i] * np.sin(2 * angle)

        # Retardance (magnitude)
        retardance = np.sqrt(a2**2 + b2**2) / (a0 + 1e-10)

        # Orientation (phase)
        orientation = 0.5 * np.arctan2(b2, a2)

        return retardance, orientation

    def analyze_birefringence_sensitivity(self, deviations: List[float] = [0.1, 0.2, 0.5, 1.0]) -> pd.DataFrame:
        """
        Analyze how angular deviations affect birefringence calculations.

        Args:
            deviations: Angular deviations to test (in degrees)

        Returns:
            DataFrame with birefringence sensitivity metrics
        """
        results = []

        # Compute reference birefringence with all correct angles
        ref_retardance, ref_orientation = self.compute_birefringence()

        if ref_retardance is None:
            return pd.DataFrame()

        # Test with one angle deviated at a time
        base_angles = sorted(self.images.keys())

        for target_angle in [7, 14, 21]:  # Test key angles
            if target_angle not in base_angles:
                continue

            for deviation in deviations:
                # Create modified angle list
                modified_angles = base_angles.copy()
                idx = modified_angles.index(target_angle)

                # Simulate the deviated image
                simulated = self.simulate_angular_deviations(target_angle, [deviation])

                if target_angle + deviation in simulated:
                    # Replace the original image with simulated one
                    temp_img = self.images[target_angle]
                    self.images[target_angle] = simulated[target_angle + deviation]

                    # Compute birefringence with deviated angle
                    dev_retardance, dev_orientation = self.compute_birefringence()

                    # Restore original image
                    self.images[target_angle] = temp_img

                    if dev_retardance is not None:
                        # Compute differences
                        retardance_diff = dev_retardance - ref_retardance
                        orientation_diff = dev_orientation - ref_orientation

                        # Wrap orientation difference to [-pi, pi]
                        orientation_diff = np.angle(np.exp(1j * orientation_diff))

                        results.append({
                            'deviated_angle': target_angle,
                            'deviation': deviation,
                            'mean_retardance_error': np.mean(np.abs(retardance_diff)),
                            'max_retardance_error': np.max(np.abs(retardance_diff)),
                            'std_retardance_error': np.std(retardance_diff),
                            'mean_orientation_error_deg': np.mean(np.abs(orientation_diff)) * 180/np.pi,
                            'max_orientation_error_deg': np.max(np.abs(orientation_diff)) * 180/np.pi,
                            'std_orientation_error_deg': np.std(orientation_diff) * 180/np.pi,
                            'retardance_rmse': np.sqrt(np.mean(retardance_diff**2)),
                            'orientation_rmse_deg': np.sqrt(np.mean(orientation_diff**2)) * 180/np.pi
                        })

        return pd.DataFrame(results)

    def visualize_difference_maps(self, reference_angle: float = 7.0,
                                 comparison_angles: List[float] = None):
        """
        Create visual difference maps between reference and comparison images.

        Args:
            reference_angle: Reference angle
            comparison_angles: Angles to compare (defaults to small deviations)
        """
        if reference_angle not in self.images:
            print(f"Error: Reference angle {reference_angle} not found")
            return

        ref_image = self.images[reference_angle]

        if comparison_angles is None:
            # Use small deviations if available
            comparison_angles = [a for a in self.images.keys()
                               if a != reference_angle and abs(a - reference_angle) <= 7]

        n_comparisons = len(comparison_angles)
        if n_comparisons == 0:
            print("No comparison angles available")
            return

        fig, axes = plt.subplots(3, n_comparisons, figsize=(4*n_comparisons, 12))
        if n_comparisons == 1:
            axes = axes.reshape(-1, 1)

        for i, angle in enumerate(comparison_angles):
            if angle not in self.images:
                continue

            comp_image = self.images[angle]
            diff_image = comp_image - ref_image
            abs_diff = np.abs(diff_image)

            # Original comparison image
            im1 = axes[0, i].imshow(comp_image, cmap='gray')
            axes[0, i].set_title(f'{angle} degrees')
            axes[0, i].axis('off')
            plt.colorbar(im1, ax=axes[0, i], fraction=0.046)

            # Difference map
            vmax = np.percentile(abs_diff, 99)
            im2 = axes[1, i].imshow(diff_image, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            axes[1, i].set_title(f'Difference from {reference_angle} deg')
            axes[1, i].axis('off')
            plt.colorbar(im2, ax=axes[1, i], fraction=0.046)

            # Absolute difference map
            im3 = axes[2, i].imshow(abs_diff, cmap='hot')
            axes[2, i].set_title(f'Absolute difference')
            axes[2, i].axis('off')
            plt.colorbar(im3, ax=axes[2, i], fraction=0.046)

        plt.suptitle(f'PPM Image Differences (Reference: {reference_angle} degrees)', fontsize=14)
        plt.tight_layout()
        plt.savefig(self.output_dir / f'difference_maps_ref{reference_angle}.png', dpi=150)
        plt.close()  # Don't block - file is already saved

    def visualize_birefringence_comparison(self, angle_sets: List[List[float]] = None):
        """
        Compare birefringence calculations with different angle sets.

        Args:
            angle_sets: List of angle sets to compare
        """
        if angle_sets is None:
            # Default: compare full set vs set with one angle slightly off
            full_set = sorted(self.images.keys())
            angle_sets = [
                full_set,
                full_set  # Will modify one angle in the second set
            ]

        fig, axes = plt.subplots(2, len(angle_sets), figsize=(6*len(angle_sets), 10))

        for i, angles in enumerate(angle_sets):
            retardance, orientation = self.compute_birefringence(angles)

            if retardance is None:
                continue

            # Retardance map
            im1 = axes[0, i].imshow(retardance, cmap='viridis')
            axes[0, i].set_title(f'Retardance (Angles: {len(angles)})')
            axes[0, i].axis('off')
            plt.colorbar(im1, ax=axes[0, i], fraction=0.046, label='Retardance')

            # Orientation map (convert to degrees)
            orientation_deg = orientation * 180 / np.pi
            im2 = axes[1, i].imshow(orientation_deg, cmap='hsv', vmin=-90, vmax=90)
            axes[1, i].set_title('Orientation')
            axes[1, i].axis('off')
            plt.colorbar(im2, ax=axes[1, i], fraction=0.046, label='Orientation (deg)')

        plt.suptitle('Birefringence Analysis Comparison', fontsize=14)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'birefringence_comparison.png', dpi=150)
        plt.close()  # Don't block - file is already saved

    def generate_sensitivity_plots(self, df_differences: pd.DataFrame,
                                  df_birefringence: pd.DataFrame):
        """
        Generate plots showing sensitivity to angular deviations.

        Args:
            df_differences: DataFrame from compute_image_differences
            df_birefringence: DataFrame from analyze_birefringence_sensitivity
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Plot 1: Image difference metrics vs angular deviation
        if not df_differences.empty:
            ax = axes[0, 0]
            ax.plot(df_differences['angular_difference'],
                   df_differences['normalized_mae'], 'o-', label='Normalized MAE')
            ax.plot(df_differences['angular_difference'],
                   1 - df_differences['ssim'], 's-', label='1 - SSIM')
            ax.set_xlabel('Angular Difference (degrees)')
            ax.set_ylabel('Error Metric')
            ax.set_title('Image Difference vs Angular Deviation')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Plot 2: Correlation vs angular deviation
        if not df_differences.empty:
            ax = axes[0, 1]
            ax.plot(df_differences['angular_difference'],
                   df_differences['correlation'], 'o-', color='blue')
            ax.set_xlabel('Angular Difference (degrees)')
            ax.set_ylabel('Correlation Coefficient')
            ax.set_title('Image Correlation vs Angular Deviation')
            ax.grid(True, alpha=0.3)

        # Plot 3: PSNR vs angular deviation
        if not df_differences.empty:
            ax = axes[0, 2]
            ax.plot(df_differences['angular_difference'],
                   df_differences['psnr'], 'o-', color='green')
            ax.set_xlabel('Angular Difference (degrees)')
            ax.set_ylabel('PSNR (dB)')
            ax.set_title('Peak Signal-to-Noise Ratio')
            ax.grid(True, alpha=0.3)

        # Plot 4: Retardance error vs deviation
        if not df_birefringence.empty:
            ax = axes[1, 0]
            for angle in df_birefringence['deviated_angle'].unique():
                subset = df_birefringence[df_birefringence['deviated_angle'] == angle]
                ax.plot(subset['deviation'], subset['mean_retardance_error'],
                       'o-', label=f'{angle} deg')
            ax.set_xlabel('Angular Deviation (degrees)')
            ax.set_ylabel('Mean Retardance Error')
            ax.set_title('Retardance Sensitivity to Angular Deviation')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Plot 5: Orientation error vs deviation
        if not df_birefringence.empty:
            ax = axes[1, 1]
            for angle in df_birefringence['deviated_angle'].unique():
                subset = df_birefringence[df_birefringence['deviated_angle'] == angle]
                ax.plot(subset['deviation'], subset['mean_orientation_error_deg'],
                       's-', label=f'{angle} deg')
            ax.set_xlabel('Angular Deviation (degrees)')
            ax.set_ylabel('Mean Orientation Error (degrees)')
            ax.set_title('Orientation Sensitivity to Angular Deviation')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Plot 6: Combined RMSE metrics
        if not df_birefringence.empty:
            ax = axes[1, 2]
            mean_by_dev = df_birefringence.groupby('deviation').mean()
            ax.plot(mean_by_dev.index, mean_by_dev['retardance_rmse'],
                   'o-', label='Retardance RMSE')
            ax2 = ax.twinx()
            ax2.plot(mean_by_dev.index, mean_by_dev['orientation_rmse_deg'],
                    's-', color='red', label='Orientation RMSE')
            ax.set_xlabel('Angular Deviation (degrees)')
            ax.set_ylabel('Retardance RMSE', color='blue')
            ax2.set_ylabel('Orientation RMSE (deg)', color='red')
            ax.set_title('Birefringence RMSE vs Angular Deviation')
            ax.grid(True, alpha=0.3)

        plt.suptitle('PPM Rotation Sensitivity Analysis', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'sensitivity_analysis.png', dpi=150)
        plt.close()  # Don't block - file is already saved

    def generate_report(self, df_differences: pd.DataFrame,
                       df_birefringence: pd.DataFrame):
        """
        Generate a comprehensive report of the analysis.

        Args:
            df_differences: DataFrame from compute_image_differences
            df_birefringence: DataFrame from analyze_birefringence_sensitivity
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'base_path': str(self.base_path),
            'n_images_loaded': len(self.images),
            'angles_available': sorted(self.images.keys()),
            'image_shape': self.image_shape,
            'summary': {}
        }

        # Summarize image difference sensitivity
        if not df_differences.empty:
            report['summary']['image_differences'] = {
                'correlation_vs_angle': {
                    str(row['angular_difference']): row['correlation']
                    for _, row in df_differences.iterrows()
                },
                'ssim_vs_angle': {
                    str(row['angular_difference']): row['ssim']
                    for _, row in df_differences.iterrows()
                },
                'critical_angle': float(df_differences.loc[
                    df_differences['correlation'] < 0.95
                ]['angular_difference'].min()) if any(df_differences['correlation'] < 0.95) else None
            }

        # Summarize birefringence sensitivity
        if not df_birefringence.empty:
            report['summary']['birefringence_sensitivity'] = {
                'max_retardance_error_per_degree': float(
                    df_birefringence['mean_retardance_error'].max() /
                    df_birefringence['deviation'].max()
                ),
                'max_orientation_error_per_degree': float(
                    df_birefringence['mean_orientation_error_deg'].max() /
                    df_birefringence['deviation'].max()
                ),
                'critical_deviation_retardance': float(
                    df_birefringence[
                        df_birefringence['mean_retardance_error'] > 0.01
                    ]['deviation'].min()
                ) if any(df_birefringence['mean_retardance_error'] > 0.01) else None,
                'critical_deviation_orientation': float(
                    df_birefringence[
                        df_birefringence['mean_orientation_error_deg'] > 1.0
                    ]['deviation'].min()
                ) if any(df_birefringence['mean_orientation_error_deg'] > 1.0) else None
            }

        # Save detailed DataFrames
        df_differences.to_csv(self.output_dir / 'image_differences.csv', index=False)
        df_birefringence.to_csv(self.output_dir / 'birefringence_sensitivity.csv', index=False)

        # Save JSON report
        with open(self.output_dir / 'analysis_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)

        # Generate text summary
        with open(self.output_dir / 'summary.txt', 'w') as f:
            f.write("PPM Rotation Sensitivity Analysis Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Analysis Date: {report['timestamp']}\n")
            f.write(f"Images Analyzed: {report['n_images_loaded']}\n")
            f.write(f"Available Angles: {report['angles_available']}\n\n")

            if 'image_differences' in report['summary']:
                f.write("Image Quality Metrics:\n")
                f.write("-" * 30 + "\n")
                crit = report['summary']['image_differences'].get('critical_angle')
                if crit:
                    f.write(f"Critical angle (correlation < 0.95): {crit} degrees\n")
                f.write("\n")

            if 'birefringence_sensitivity' in report['summary']:
                f.write("Birefringence Sensitivity:\n")
                f.write("-" * 30 + "\n")
                bs = report['summary']['birefringence_sensitivity']
                f.write(f"Max retardance error per degree: {bs['max_retardance_error_per_degree']:.4f}\n")
                f.write(f"Max orientation error per degree: {bs['max_orientation_error_per_degree']:.2f} deg\n")
                if bs.get('critical_deviation_retardance'):
                    f.write(f"Critical deviation for retardance: {bs['critical_deviation_retardance']:.2f} deg\n")
                if bs.get('critical_deviation_orientation'):
                    f.write(f"Critical deviation for orientation: {bs['critical_deviation_orientation']:.2f} deg\n")

        print(f"\nAnalysis complete. Results saved to: {self.output_dir}")
        return report


def main():
    parser = argparse.ArgumentParser(description='Analyze PPM rotation sensitivity')
    parser.add_argument('image_dir', help='Directory containing PPM images')
    parser.add_argument('--output', '-o', help='Output directory for results')
    parser.add_argument('--reference-angle', '-r', type=float, default=7.0,
                       help='Reference angle for comparisons (default: 7.0)')
    parser.add_argument('--deviations', '-d', nargs='+', type=float,
                       default=[0.1, 0.2, 0.3, 0.5, 1.0],
                       help='Angular deviations to test (default: 0.1 0.2 0.3 0.5 1.0)')
    parser.add_argument('--skip-visualization', action='store_true',
                       help='Skip visualization plots')

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = PPMRotationAnalyzer(args.image_dir, args.output)

    # Load images
    images = analyzer.load_images()
    if len(images) < 3:
        print("Error: Need at least 3 images at different angles for analysis")
        return

    # Compute image differences
    print("\nAnalyzing image differences...")
    df_differences = analyzer.compute_image_differences(args.reference_angle)

    # Analyze birefringence sensitivity
    print("Analyzing birefringence sensitivity...")
    df_birefringence = analyzer.analyze_birefringence_sensitivity(args.deviations)

    # Generate visualizations
    if not args.skip_visualization:
        print("Generating visualizations...")
        analyzer.visualize_difference_maps(args.reference_angle)
        analyzer.visualize_birefringence_comparison()
        analyzer.generate_sensitivity_plots(df_differences, df_birefringence)

    # Generate report
    print("Generating report...")
    report = analyzer.generate_report(df_differences, df_birefringence)

    # Print summary to console
    print("\n" + "=" * 50)
    print("ANALYSIS SUMMARY")
    print("=" * 50)

    if not df_differences.empty:
        print(f"\nImage Quality at Various Angular Differences:")
        print(df_differences[['angular_difference', 'correlation', 'ssim', 'psnr']].to_string(index=False))

    if not df_birefringence.empty:
        print(f"\nBirefringence Sensitivity (Mean Errors):")
        summary = df_birefringence.groupby('deviation')[
            ['mean_retardance_error', 'mean_orientation_error_deg']
        ].mean()
        print(summary.to_string())

    print(f"\nFull results saved to: {analyzer.output_dir}")


if __name__ == "__main__":
    main()