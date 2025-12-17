"""
Autofocus Benchmarking Module - COMPATIBILITY SHIM

This module is a backward-compatibility shim that re-exports autofocus benchmarking
classes from their new location. New code should import directly from:

    from smart_wsi_scanner.autofocus.benchmark import (
        AutofocusBenchmark,
        BenchmarkResult,
        BenchmarkConfig,
        ZSafetyError,
    )

This shim exists for backward compatibility and will be deprecated
in a future release.
"""

# Re-export from new location
from .autofocus.benchmark import (
    AutofocusBenchmark,
    BenchmarkResult,
    BenchmarkConfig,
    ZSafetyError,
    Z_ABSOLUTE_SAFETY_LIMIT_UM,
    OBJECTIVE_SAFETY_LIMITS_UM,
    AUTOFOCUS_OVERSHOOT_MARGIN_UM,
    run_autofocus_benchmark_from_server,
)

__all__ = [
    "AutofocusBenchmark",
    "BenchmarkResult",
    "BenchmarkConfig",
    "ZSafetyError",
    "Z_ABSOLUTE_SAFETY_LIMIT_UM",
    "OBJECTIVE_SAFETY_LIMITS_UM",
    "AUTOFOCUS_OVERSHOOT_MARGIN_UM",
    "run_autofocus_benchmark_from_server",
]
