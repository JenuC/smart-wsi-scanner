"""
Simple test for target intensity mapping logic.

This script tests the target intensity mapping without importing dependencies.
"""


def get_target_intensity_for_background(modality: str, angle: float) -> float:
    """
    Get target intensity for background acquisition based on modality and angle.

    (Copy of implementation for testing without dependencies)
    """
    # Normalize modality to lowercase for comparison
    modality_lower = modality.lower()

    # Brightfield modality
    if "brightfield" in modality_lower or "bf" in modality_lower:
        return 250.0

    # PPM modality - angle-specific targets
    if "ppm" in modality_lower:
        # Normalize angle to absolute value for symmetric angles
        abs_angle = abs(angle)

        if abs_angle == 90:
            return 245.0
        elif abs_angle in [7, 5]:
            # Distinguish between positive and negative angles
            if angle > 0:
                return 150.0  # +7 or +5 degrees
            else:
                return 155.0  # -7 or -5 degrees
        elif abs_angle == 0:
            return 125.0
        else:
            # Default for unknown PPM angles
            print(f"WARNING: Unknown PPM angle {angle}, using default target 150")
            return 150.0

    # Default fallback
    print(f"WARNING: Unknown modality {modality}, using default target 200")
    return 200.0


def test_target_intensity_mapping():
    """Test that target intensities are correctly mapped for all modality/angle combinations."""

    print("Testing target intensity mapping...")
    print("=" * 60)

    # Test brightfield
    test_cases = [
        # (modality, angle, expected_intensity)
        ("brightfield", 0, 250.0),
        ("Brightfield", 90, 250.0),  # Case insensitive
        ("bf", 0, 250.0),

        # Test PPM angles
        ("ppm", 90, 245.0),
        ("PPM", -90, 245.0),
        ("ppm", 7, 150.0),
        ("ppm", 5, 150.0),
        ("ppm", -7, 155.0),
        ("ppm", -5, 155.0),
        ("ppm", 0, 125.0),

        # Test with modality prefixes (common pattern)
        ("ppm_20x", 90, 245.0),
        ("ppm_20x", 5, 150.0),
        ("ppm_20x", -5, 155.0),
        ("ppm_20x", 0, 125.0),
    ]

    all_passed = True

    for modality, angle, expected in test_cases:
        result = get_target_intensity_for_background(modality, angle)
        passed = result == expected
        all_passed = all_passed and passed

        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: modality='{modality}', angle={angle:4.0f}° => "
              f"expected={expected:5.1f}, got={result:5.1f}")

    print("=" * 60)

    if all_passed:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed!")
        return 1


if __name__ == "__main__":
    import sys
    exit_code = test_target_intensity_mapping()
    sys.exit(exit_code)
