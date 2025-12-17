# Deprecated Code Archive

This folder contains code that is no longer actively used in the smart-wsi-scanner package but has been preserved for reference rather than deleted.

## Policy

- Code moved here should NOT be imported by production code
- Files are kept for historical reference and potential future use
- Each file should have a comment at the top explaining why it was deprecated
- Review this folder periodically to determine if files can be permanently removed

## Contents

### Day 1 (2024-12-17)

- **hardware_original.py** - Original hardware.py containing Position, MicroscopeHardware classes
  - Moved to: `hardware/base.py`
  - Shim: `hardware/__init__.py` re-exports Position, MicroscopeHardware

### Day 2 (2024-12-17)

- **config_original.py** - ConfigManager for YAML configuration handling
  - Moved to: `config/manager.py`
  - Shim: `config.py` re-exports ConfigManager

- **hardware_pycromanager_original.py** - Pycromanager hardware implementation
  - Moved to: `hardware/pycromanager.py`
  - Shim: `hardware_pycromanager.py` re-exports PycromanagerHardware, init_pycromanager

- **swsi_autofocus_metrics_original.py** - AutofocusMetrics class
  - Moved to: `autofocus/metrics.py`
  - Shim: `swsi_autofocus_metrics.py` re-exports AutofocusMetrics

- **swsi_empty_region_detection_original.py** - EmptyRegionDetector class
  - Moved to: `imaging/tissue_detection.py`
  - Shim: `swsi_empty_region_detection.py` re-exports EmptyRegionDetector

- **qp_autofocus_test_original.py** - Autofocus testing functions
  - Moved to: `autofocus/test.py`
  - Shim: `qp_autofocus_test.py` re-exports test functions

- **qp_autofocus_benchmark_original.py** - AutofocusBenchmark class
  - Moved to: `autofocus/benchmark.py`
  - Shim: `qp_autofocus_benchmark.py` re-exports benchmark classes

### Day 3 (2024-12-17)

- **qp_server_original.py** - QuPath socket server
  - Moved to: `server/qp_server.py`
  - Shim: `qp_server.py` re-exports all

- **qp_server_config_original.py** - Server protocol commands
  - Moved to: `server/protocol.py`
  - Shim: `qp_server_config.py` re-exports Command, ExtendedCommand

- **qp_client_original.py** - Test client
  - Moved to: `server/client.py`
  - Shim: `qp_client.py` re-exports client functions

- **qp_acquisition_original.py** - Acquisition workflow
  - Moved to: `acquisition/workflow.py`
  - Shim: `qp_acquisition.py` re-exports _acquisition_workflow

- **qp_text_pipeline_original.py** - Text-based pipeline DSL
  - Moved to: `acquisition/pipeline.py`
  - Shim: `qp_text_pipeline.py` re-exports all

## When to Move Code Here

- Functions/classes that have been replaced by better implementations
- Experimental code that didn't make it to production
- Legacy compatibility code after deprecation period ends
- Old test files that are no longer relevant

## When to Delete from Here

- After 6+ months with no references
- When the replacement code is fully stable
- After team review confirms no future need
