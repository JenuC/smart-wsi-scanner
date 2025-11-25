#!/usr/bin/env python3
"""
Standalone Autofocus Benchmark Runner
=====================================

Run autofocus parameter benchmarks directly from the command line.

Usage:
    python run_autofocus_benchmark.py --reference_z -5000.0 --output ./results --objective 20X
    python run_autofocus_benchmark.py --reference_z -5000.0 --output ./results --quick --objective 10X
    python run_autofocus_benchmark.py --reference_z -5000.0 --output ./results --distances 5,10,20,30,50

SAFETY (Upright Microscope):
    This system includes safety limits to prevent objective-sample collision.
    On this upright microscope, MORE NEGATIVE Z = stage RAISED = closer to objective.

    Safety limits by objective (hardcoded in qp_autofocus_benchmark.py):
        - 10X: -5550 um (longest working distance)
        - 20X: -5500 um (moderate working distance)
        - 40X: -5400 um (shortest working distance - most conservative)

    The benchmark performs a pre-flight safety check before any movements.
    If the planned test range would exceed safety limits, the benchmark aborts.

Prerequisites:
    - Micro-Manager must be running with Pycro-Manager bridge enabled
    - Microscope must be positioned at a location with visible sample
    - User must manually verify the reference_z is the true focus position
    - Specify the correct objective for proper safety limits
"""

import argparse
import logging
import sys
import pathlib
from datetime import datetime

# Setup logging
log_dir = pathlib.Path(__file__).parent / "benchmark_logfiles"
log_dir.mkdir(parents=True, exist_ok=True)
log_filename = log_dir / f'af_benchmark_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Run autofocus parameter benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full benchmark with 20X objective
    python run_autofocus_benchmark.py --reference_z -5000.0 --output ./results --objective 20X

    # Quick benchmark with 10X (reduced parameter space)
    python run_autofocus_benchmark.py --reference_z -4800.0 --output ./results --quick --objective 10X

    # Custom distances to test with 40X (conservative safety)
    python run_autofocus_benchmark.py --reference_z -5200.0 --output ./results --distances 5,10,20 --objective 40X

    # Distance sweep with fixed parameters
    python run_autofocus_benchmark.py --reference_z -5000.0 --output ./results --sweep --distances 2,5,10,15,20 --objective 20X

SAFETY NOTES:
    - CRITICAL: Specify --objective to apply correct safety limits
    - The benchmark performs a pre-flight safety check before any movements
    - If planned test range exceeds safety limits, the benchmark will abort
    - Safety limits are MORE CONSERVATIVE for higher magnification objectives
    - More negative Z = stage raised = closer to objective = DANGER

    Safety limits (hardcoded):
        10X: -5550 um
        20X: -5500 um
        40X: -5400 um (most conservative)

Notes:
    - IMPORTANT: Manually verify the reference_z is the true focus position before running
    - The benchmark will move the stage to various Z positions
    - Results are saved to CSV and JSON files in the output directory
        """
    )

    parser.add_argument(
        "--reference_z",
        type=float,
        required=True,
        help="Known good focus Z position in micrometers (user must verify this is in focus)"
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Directory to save benchmark results"
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmark with reduced parameter space"
    )

    parser.add_argument(
        "--distances",
        type=str,
        default=None,
        help="Comma-separated list of distances to test (um from focus). Default: 5,10,20,30,50"
    )

    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run distance sweep with fixed parameters (tests many distances, fewer parameter combinations)"
    )

    parser.add_argument(
        "--n_steps",
        type=int,
        default=21,
        help="Fixed n_steps for distance sweep mode (default: 21)"
    )

    parser.add_argument(
        "--search_range",
        type=float,
        default=35.0,
        help="Fixed search range for distance sweep mode (default: 35.0 um)"
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to microscope config YAML (default: config_PPM.yml)"
    )

    parser.add_argument(
        "--objective",
        type=str,
        default=None,
        help="Objective identifier for logging"
    )

    parser.add_argument(
        "--no_adaptive",
        action="store_true",
        help="Skip adaptive autofocus testing"
    )

    parser.add_argument(
        "--no_standard",
        action="store_true",
        help="Skip standard autofocus testing"
    )

    args = parser.parse_args()

    # Parse distances if provided
    test_distances = None
    if args.distances:
        try:
            test_distances = [float(d.strip()) for d in args.distances.split(",")]
            logger.info(f"Testing distances: {test_distances} um")
        except ValueError as e:
            logger.error(f"Invalid distances format: {e}")
            sys.exit(1)

    # Initialize hardware
    logger.info("=" * 60)
    logger.info("AUTOFOCUS BENCHMARK RUNNER")
    logger.info("=" * 60)
    logger.info(f"Reference Z: {args.reference_z} um")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Log file: {log_filename}")

    try:
        logger.info("Initializing Pycro-Manager connection...")
        from smart_wsi_scanner.hardware_pycromanager import PycromanagerHardware, init_pycromanager
        from smart_wsi_scanner.config import ConfigManager

        core, studio = init_pycromanager()
        if not core:
            logger.error("Failed to connect to Micro-Manager. Is it running with Pycro-Manager?")
            sys.exit(1)

        logger.info("Pycro-Manager connected successfully")

        # Load configuration
        config_manager = ConfigManager()
        # Config files are in smart_wsi_scanner/configurations/
        swsi_dir = pathlib.Path(__file__).parent.parent / "smart_wsi_scanner"
        if args.config:
            config_path = pathlib.Path(args.config)
        else:
            config_path = swsi_dir / "configurations" / "config_PPM.yml"

        loci_rsc_file = swsi_dir / "configurations" / "resources" / "resources_LOCI.yml"

        settings = config_manager.get_config(config_path.stem.replace("config_", ""))
        if settings is None:
            settings = config_manager.load_config(str(config_path), str(loci_rsc_file))

        hardware = PycromanagerHardware(core, studio, settings)
        logger.info("Hardware initialized")

        # Get current position for verification
        current_pos = hardware.get_current_position()
        logger.info(f"Current position: X={current_pos.x:.2f}, Y={current_pos.y:.2f}, Z={current_pos.z:.2f}")

        # Warn if current Z differs significantly from reference
        if abs(current_pos.z - args.reference_z) > 5.0:
            logger.warning("!" * 60)
            logger.warning(f"CAUTION: Current Z ({current_pos.z:.2f}) differs from reference_z ({args.reference_z:.2f}) by {abs(current_pos.z - args.reference_z):.2f} um")
            logger.warning("Please verify the reference_z is correct before proceeding!")
            logger.warning("!" * 60)

            response = input("Continue anyway? (yes/no): ")
            if response.lower() not in ("yes", "y"):
                logger.info("Benchmark cancelled by user")
                sys.exit(0)

        # Import and run benchmark
        from smart_wsi_scanner.qp_autofocus_benchmark import (
            AutofocusBenchmark,
            BenchmarkConfig
        )

        benchmark = AutofocusBenchmark(hardware, config_manager, logger)

        # Configure based on mode
        if args.quick:
            logger.info("Running quick benchmark (reduced parameter space)")
            results = benchmark.run_quick_benchmark(args.reference_z, args.output)

        elif args.sweep:
            logger.info(f"Running distance sweep (n_steps={args.n_steps}, range={args.search_range}um)")
            if test_distances is None:
                test_distances = [2, 5, 10, 15, 20, 25, 30, 40, 50]
            results = benchmark.run_distance_sweep(
                args.reference_z,
                test_distances,
                args.output,
                n_steps=args.n_steps,
                search_range=args.search_range
            )

        else:
            # Full benchmark with optional customization
            config = BenchmarkConfig()

            if test_distances:
                config.test_distances = test_distances

            if args.no_adaptive:
                config.test_adaptive = False

            if args.no_standard:
                config.test_standard = False

            logger.info("Running full benchmark")
            results = benchmark.run_benchmark(
                args.reference_z,
                config,
                args.output,
                args.objective
            )

        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("BENCHMARK COMPLETE")
        logger.info("=" * 60)

        if "results_directory" in results:
            logger.info(f"Results saved to: {results['results_directory']}")

        if "fastest_standard" in results:
            fs = results["fastest_standard"]
            logger.info(f"\nRecommended standard AF config:")
            logger.info(f"  n_steps: {fs['n_steps']}")
            logger.info(f"  search_range: {fs['search_range_um']} um")
            logger.info(f"  interp_kind: {fs['interp_kind']}")
            logger.info(f"  score_metric: {fs['score_metric']}")
            logger.info(f"  (Duration: {fs['duration_ms']:.0f}ms, Error: {fs['z_error_um']:.2f}um)")

        if "fastest_adaptive" in results:
            fa = results["fastest_adaptive"]
            logger.info(f"\nRecommended adaptive AF config:")
            logger.info(f"  initial_step: {fa['initial_step_um']} um")
            logger.info(f"  score_metric: {fa['score_metric']}")
            logger.info(f"  (Duration: {fa['duration_ms']:.0f}ms, Error: {fa['z_error_um']:.2f}um)")

    except KeyboardInterrupt:
        logger.info("\nBenchmark interrupted by user")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
