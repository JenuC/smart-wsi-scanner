#!/usr/bin/env python3
"""
JAI Camera Property Discovery Script

This script enumerates and tests all available properties on the JAI AP-3200T-USB
camera through Micro-Manager. Run this script to discover property ranges,
dependencies, and behaviors before implementing white balance functionality.

Requirements:
- Micro-Manager running with JAICamera device configured
- Pycromanager connected

Output:
- Console: Property enumeration with current values
- File: jai_camera_properties_report.yml with full property details

Related:
- PR #781: https://github.com/micro-manager/mmCoreAndDevices/pull/781
- Plan: claude-reports/2025-12-16_white-balance-implementation-plan.md
"""

import logging
from pathlib import Path
from datetime import datetime
import yaml

# Setup logging - ASCII only for Windows compatibility
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def discover_device_properties(core, device_name: str) -> dict:
    """
    Enumerate all properties for a device.

    Args:
        core: Pycromanager Core instance (wraps MMCore Java object)
        device_name: Name of the device to query

    Returns:
        dict: Property details including value, type, limits, allowed values

    Note:
        MMCore uses camelCase Java method names, not snake_case.
    """
    properties = {}
    n_props = core.getNumberOfDeviceProperties(device_name)

    for i in range(n_props):
        prop_name = core.getDevicePropertyNames(device_name)[i]

        prop_info = {
            'current_value': None,
            'is_read_only': False,
            'has_limits': False,
            'lower_limit': None,
            'upper_limit': None,
            'allowed_values': [],
            'property_type': 'unknown'
        }

        try:
            # Get current value
            prop_info['current_value'] = core.getProperty(device_name, prop_name)

            # Check if read-only
            prop_info['is_read_only'] = core.isPropertyReadOnly(device_name, prop_name)

            # Check for limits (numeric properties)
            prop_info['has_limits'] = core.hasPropertyLimits(device_name, prop_name)
            if prop_info['has_limits']:
                prop_info['lower_limit'] = core.getPropertyLowerLimit(device_name, prop_name)
                prop_info['upper_limit'] = core.getPropertyUpperLimit(device_name, prop_name)

            # Get allowed values (enum properties)
            allowed = core.getAllowedPropertyValues(device_name, prop_name)
            if allowed:
                prop_info['allowed_values'] = list(allowed)

            # Determine property type
            if prop_info['allowed_values']:
                prop_info['property_type'] = 'enum'
            elif prop_info['has_limits']:
                prop_info['property_type'] = 'numeric'
            else:
                prop_info['property_type'] = 'string'

        except Exception as e:
            logger.warning(f"Error reading property {prop_name}: {e}")
            prop_info['error'] = str(e)

        properties[prop_name] = prop_info

    return properties


def test_individual_exposure_mode(core, device_name: str = "JAICamera") -> dict:
    """
    Test the individual exposure mode properties.

    Tests:
    1. Toggle ExposureIsIndividual On/Off
    2. Read/write per-channel exposures when mode is On
    3. Verify per-channel values are independent
    4. Check frame rate behavior with individual exposures

    Args:
        core: Pycromanager Core instance (wraps MMCore Java object)
        device_name: Camera device name

    Returns:
        dict: Test results including success/failure and discovered limits
    """
    results = {
        'exposure_individual_toggle': None,
        'per_channel_exposure_works': False,
        'channel_exposures_independent': False,
        'frame_rate_behavior': None,
        'exposure_limits': {},
        'notes': []
    }

    try:
        # Test 1: Toggle ExposureIsIndividual
        original_mode = core.getProperty(device_name, "ExposureIsIndividual")
        results['exposure_individual_toggle'] = original_mode

        # Enable individual mode
        core.setProperty(device_name, "ExposureIsIndividual", "On")
        core.waitForDevice(device_name)
        results['notes'].append("Successfully enabled ExposureIsIndividual")

        # Test 2: Read per-channel exposures
        channels = ['Exposure_Red', 'Exposure_Green', 'Exposure_Blue']
        for channel in channels:
            try:
                value = float(core.getProperty(device_name, channel))
                results['exposure_limits'][channel] = {
                    'current': value,
                    'min': core.getPropertyLowerLimit(device_name, channel),
                    'max': core.getPropertyUpperLimit(device_name, channel)
                }
            except Exception as e:
                results['notes'].append(f"Failed to read {channel}: {e}")

        # Test 3: Verify independence
        if results['exposure_limits']:
            # Set different values for each channel
            test_values = {'Exposure_Red': 50.0, 'Exposure_Green': 75.0, 'Exposure_Blue': 100.0}
            for channel, value in test_values.items():
                core.setProperty(device_name, channel, str(value))

            core.waitForDevice(device_name)

            # Read back and verify
            readback = {}
            for channel in channels:
                readback[channel] = float(core.getProperty(device_name, channel))

            results['channel_exposures_independent'] = (
                abs(readback['Exposure_Red'] - 50.0) < 0.1 and
                abs(readback['Exposure_Green'] - 75.0) < 0.1 and
                abs(readback['Exposure_Blue'] - 100.0) < 0.1
            )
            results['per_channel_exposure_works'] = True

        # Test 4: Frame rate behavior
        frame_rate_before = core.getProperty(device_name, "FrameRateHz")
        # Set a long exposure on one channel
        core.setProperty(device_name, "Exposure_Blue", "500.0")
        core.waitForDevice(device_name)
        frame_rate_after = core.getProperty(device_name, "FrameRateHz")

        results['frame_rate_behavior'] = {
            'before': frame_rate_before,
            'after_long_exposure': frame_rate_after,
            'auto_adjusted': frame_rate_before != frame_rate_after
        }

        # Restore original mode
        core.setProperty(device_name, "ExposureIsIndividual", original_mode)

    except Exception as e:
        results['notes'].append(f"Test failed: {e}")

    return results


def test_individual_gain_mode(core, device_name: str = "JAICamera") -> dict:
    """
    Test the individual gain mode properties.

    Tests:
    1. Toggle GainIsIndividual On/Off
    2. Read/write per-channel analog gains
    3. Test digital gain properties
    4. Verify gain ranges

    Args:
        core: Pycromanager Core instance (wraps MMCore Java object)
        device_name: Camera device name

    Returns:
        dict: Test results including gain ranges and notes
    """
    results = {
        'gain_individual_toggle': None,
        'analog_gain_ranges': {},
        'digital_gain_ranges': {},
        'notes': []
    }

    try:
        # Test 1: Toggle GainIsIndividual
        original_mode = core.getProperty(device_name, "GainIsIndividual")
        results['gain_individual_toggle'] = original_mode

        core.setProperty(device_name, "GainIsIndividual", "On")
        core.waitForDevice(device_name)

        # Test 2: Analog gains
        analog_channels = ['Gain_AnalogRed', 'Gain_AnalogGreen', 'Gain_AnalogBlue']
        for channel in analog_channels:
            try:
                results['analog_gain_ranges'][channel] = {
                    'current': float(core.getProperty(device_name, channel)),
                    'min': core.getPropertyLowerLimit(device_name, channel),
                    'max': core.getPropertyUpperLimit(device_name, channel)
                }
            except Exception as e:
                results['notes'].append(f"Failed to read {channel}: {e}")

        # Test 3: Digital gains (note: Green digital gain may not exist per PR #781)
        digital_channels = ['Gain_DigitalRed', 'Gain_DigitalBlue']
        for channel in digital_channels:
            try:
                results['digital_gain_ranges'][channel] = {
                    'current': float(core.getProperty(device_name, channel)),
                    'min': core.getPropertyLowerLimit(device_name, channel),
                    'max': core.getPropertyUpperLimit(device_name, channel)
                }
            except Exception as e:
                results['notes'].append(f"Digital gain {channel} not available: {e}")

        # Restore original mode
        core.setProperty(device_name, "GainIsIndividual", original_mode)

    except Exception as e:
        results['notes'].append(f"Test failed: {e}")

    return results


def test_black_level_properties(core, device_name: str = "JAICamera") -> dict:
    """
    Test black level properties.

    Args:
        core: Pycromanager Core instance (wraps MMCore Java object)
        device_name: Camera device name

    Returns:
        dict: Black level property ranges and notes
    """
    results = {
        'black_level_ranges': {},
        'notes': []
    }

    black_level_props = ['BlackLevel_DigitalAll', 'BlackLevel_DigitalRed', 'BlackLevel_DigitalBlue']

    for prop in black_level_props:
        try:
            results['black_level_ranges'][prop] = {
                'current': float(core.getProperty(device_name, prop)),
                'min': core.getPropertyLowerLimit(device_name, prop),
                'max': core.getPropertyUpperLimit(device_name, prop)
            }
        except Exception as e:
            results['notes'].append(f"Black level {prop} not available: {e}")

    return results


def test_image_capture_with_individual_exposure(core, device_name: str = "JAICamera") -> dict:
    """
    Test actual image capture with individual exposure mode enabled.

    This verifies that per-channel exposures actually affect the captured image.

    Args:
        core: Pycromanager Core instance (wraps MMCore Java object)
        device_name: Camera device name

    Returns:
        dict: Capture test results including per-channel intensity measurements
    """
    import numpy as np

    results = {
        'capture_successful': False,
        'image_shape': None,
        'per_channel_means': {},
        'notes': []
    }

    try:
        # Enable individual exposure mode
        core.setProperty(device_name, "ExposureIsIndividual", "On")
        core.waitForDevice(device_name)

        # Set known different exposures
        core.setProperty(device_name, "Exposure_Red", "30.0")
        core.setProperty(device_name, "Exposure_Green", "50.0")
        core.setProperty(device_name, "Exposure_Blue", "80.0")
        core.waitForDevice(device_name)

        # Capture image
        core.snapImage()
        img = core.getImage()

        # Get image dimensions
        width = core.getImageWidth()
        height = core.getImageHeight()
        bytes_per_pixel = core.getBytesPerPixel()

        results['image_shape'] = {
            'width': width,
            'height': height,
            'bytes_per_pixel': bytes_per_pixel
        }

        # Reshape based on format (assuming RGB interleaved)
        if bytes_per_pixel == 3:
            img_array = np.frombuffer(img, dtype=np.uint8).reshape((height, width, 3))
            results['per_channel_means'] = {
                'red': float(img_array[:, :, 0].mean()),
                'green': float(img_array[:, :, 1].mean()),
                'blue': float(img_array[:, :, 2].mean())
            }
        elif bytes_per_pixel == 6:
            img_array = np.frombuffer(img, dtype=np.uint16).reshape((height, width, 3))
            results['per_channel_means'] = {
                'red': float(img_array[:, :, 0].mean()),
                'green': float(img_array[:, :, 1].mean()),
                'blue': float(img_array[:, :, 2].mean())
            }
        else:
            results['notes'].append(f"Unexpected bytes_per_pixel: {bytes_per_pixel}")

        results['capture_successful'] = True

        # Restore unified exposure mode
        core.setProperty(device_name, "ExposureIsIndividual", "Off")

    except Exception as e:
        results['notes'].append(f"Capture test failed: {e}")

    return results


def generate_report(all_properties: dict, test_results: dict, output_path: Path) -> None:
    """
    Generate YAML report with all discovered properties and test results.

    Args:
        all_properties: Dict of all device properties
        test_results: Dict of test results from various tests
        output_path: Path to save the YAML report
    """
    report = {
        'metadata': {
            'generated': datetime.now().isoformat(),
            'device': 'JAICamera',
            'description': 'JAI AP-3200T-USB property discovery report',
            'related_pr': 'https://github.com/micro-manager/mmCoreAndDevices/pull/781'
        },
        'all_properties': all_properties,
        'exposure_mode_tests': test_results.get('exposure', {}),
        'gain_mode_tests': test_results.get('gain', {}),
        'black_level_tests': test_results.get('black_level', {}),
        'capture_tests': test_results.get('capture', {}),
        'summary': {
            'total_properties': len(all_properties),
            'writable_properties': sum(1 for p in all_properties.values() if not p.get('is_read_only')),
            'numeric_properties': sum(1 for p in all_properties.values() if p.get('property_type') == 'numeric'),
            'enum_properties': sum(1 for p in all_properties.values() if p.get('property_type') == 'enum')
        },
        'white_balance_relevant_properties': {
            'exposure': ['ExposureIsIndividual', 'Exposure_Red', 'Exposure_Green', 'Exposure_Blue'],
            'gain': ['GainIsIndividual', 'Gain_AnalogRed', 'Gain_AnalogGreen', 'Gain_AnalogBlue',
                     'Gain_DigitalRed', 'Gain_DigitalBlue'],
            'black_level': ['BlackLevel_DigitalAll', 'BlackLevel_DigitalRed', 'BlackLevel_DigitalBlue']
        }
    }

    with open(output_path, 'w') as f:
        yaml.dump(report, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Report saved to {output_path}")


def print_summary(all_properties: dict, test_results: dict) -> None:
    """Print a human-readable summary to console."""
    exposure_results = test_results.get('exposure', {})
    gain_results = test_results.get('gain', {})
    black_level_results = test_results.get('black_level', {})
    capture_results = test_results.get('capture', {})

    print("\n" + "=" * 70)
    print("JAI CAMERA PROPERTY DISCOVERY SUMMARY")
    print("=" * 70)
    print(f"Total properties discovered: {len(all_properties)}")

    print("\n--- Exposure Individual Mode ---")
    print(f"  Toggle works: {exposure_results.get('exposure_individual_toggle') is not None}")
    print(f"  Per-channel works: {exposure_results.get('per_channel_exposure_works')}")
    print(f"  Channels independent: {exposure_results.get('channel_exposures_independent')}")
    if exposure_results.get('frame_rate_behavior'):
        fr = exposure_results['frame_rate_behavior']
        print(f"  Frame rate auto-adjusts: {fr.get('auto_adjusted')}")
        print(f"    Before: {fr.get('before')}, After long exp: {fr.get('after_long_exposure')}")

    if exposure_results.get('exposure_limits'):
        print("  Per-channel exposure limits:")
        for channel, limits in exposure_results['exposure_limits'].items():
            print(f"    {channel}: {limits.get('min')} - {limits.get('max')} ms")

    print("\n--- Gain Individual Mode ---")
    print(f"  Analog gain channels: {len(gain_results.get('analog_gain_ranges', {}))}")
    if gain_results.get('analog_gain_ranges'):
        for channel, limits in gain_results['analog_gain_ranges'].items():
            print(f"    {channel}: {limits.get('min')} - {limits.get('max')}")

    print(f"  Digital gain channels: {len(gain_results.get('digital_gain_ranges', {}))}")
    if gain_results.get('digital_gain_ranges'):
        for channel, limits in gain_results['digital_gain_ranges'].items():
            print(f"    {channel}: {limits.get('min')} - {limits.get('max')}")

    print("\n--- Black Level Properties ---")
    print(f"  Properties found: {len(black_level_results.get('black_level_ranges', {}))}")
    if black_level_results.get('black_level_ranges'):
        for prop, limits in black_level_results['black_level_ranges'].items():
            print(f"    {prop}: {limits.get('min')} - {limits.get('max')}")

    print("\n--- Image Capture Test ---")
    print(f"  Capture successful: {capture_results.get('capture_successful')}")
    if capture_results.get('image_shape'):
        shape = capture_results['image_shape']
        print(f"  Image size: {shape.get('width')}x{shape.get('height')}, {shape.get('bytes_per_pixel')} bytes/pixel")
    if capture_results.get('per_channel_means'):
        means = capture_results['per_channel_means']
        print(f"  Per-channel means (with different exposures):")
        print(f"    Red (30ms): {means.get('red', 'N/A'):.1f}")
        print(f"    Green (50ms): {means.get('green', 'N/A'):.1f}")
        print(f"    Blue (80ms): {means.get('blue', 'N/A'):.1f}")

    # Notes and warnings
    all_notes = []
    for key, results in test_results.items():
        if isinstance(results, dict) and 'notes' in results:
            all_notes.extend(results['notes'])

    if all_notes:
        print("\n--- Notes and Warnings ---")
        for note in all_notes:
            print(f"  - {note}")

    print("=" * 70)


def main():
    """Main entry point for property discovery."""
    from pycromanager import Core

    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    logger.info("Connecting to Micro-Manager...")
    core = Core()

    device_name = "JAICamera"

    # Verify JAI camera is active
    active_camera = core.getProperty("Core", "Camera")
    if active_camera != device_name:
        logger.error(f"JAICamera not active. Current camera: {active_camera}")
        logger.info("Please set JAICamera as the active camera in Micro-Manager and try again.")
        return

    logger.info("JAICamera detected. Beginning property discovery...")

    # Phase 1: Enumerate all properties
    logger.info("Enumerating device properties...")
    all_properties = discover_device_properties(core, device_name)
    logger.info(f"Found {len(all_properties)} properties")

    # Phase 2: Test specific functionality
    logger.info("Testing individual exposure mode...")
    exposure_results = test_individual_exposure_mode(core, device_name)

    logger.info("Testing individual gain mode...")
    gain_results = test_individual_gain_mode(core, device_name)

    logger.info("Testing black level properties...")
    black_level_results = test_black_level_properties(core, device_name)

    logger.info("Testing image capture with individual exposure...")
    capture_results = test_image_capture_with_individual_exposure(core, device_name)

    # Phase 3: Generate report
    test_results = {
        'exposure': exposure_results,
        'gain': gain_results,
        'black_level': black_level_results,
        'capture': capture_results
    }

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = output_dir / f"jai_camera_properties_report_{timestamp}.yml"
    generate_report(all_properties, test_results, output_path)

    # Print summary to console
    print_summary(all_properties, test_results)

    print(f"\nFull report saved to: {output_path}")


if __name__ == "__main__":
    main()
