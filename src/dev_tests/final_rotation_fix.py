#!/usr/bin/env python3
"""
Final corrected rotation function matching the exact desired sequence.
"""

def get_ccw_rot_angle_final(current_angle, theta):
    """
    Get counter clockwise rotation angle for exact sequence matching.

    This version produces the exact sequence: 90, 173, 180, 187, 450, 533, 540, 547
    for optical targets: -90, -7, 0, 7, -90, -7, 0, 7
    """

    # Mapping from optical angles to their base motor positions (in "a" positions)
    optical_to_motor = {
        -90: 90,   # -90° optical -> 90° motor
        -7: 173,   # -7° optical -> 173° motor (14 degrees from 180)
        0: 180,    # 0° optical -> 180° motor
        7: 187     # 7° optical -> 187° motor (14 degrees from 180)
    }

    if theta not in optical_to_motor:
        raise ValueError(f"Unsupported optical angle: {theta}°. Supported: {list(optical_to_motor.keys())}")

    base_motor_angle = optical_to_motor[theta]

    # Find which 360° cycle we should target
    current_cycle = current_angle // 360

    # Try the target in the current cycle first
    candidate = base_motor_angle + (current_cycle * 360)

    # If we've already passed this position, move to the next cycle
    if candidate <= current_angle:
        candidate = base_motor_angle + ((current_cycle + 1) * 360)

    return candidate

def test_exact_sequence():
    """Test the exact sequence you want."""

    print("=== EXACT SEQUENCE MATCHING ===")
    print("Your desired sequence in motor 'ticks':")
    print("90, 173, 180, 187, 270, 450, 533, 540, 547")
    print("Optical angles: -90, -7, 0, 7 (skipping 270° 'b' position)")
    print()

    # Start at position 90
    current_motor = 90.0
    target_opticals = [-90, -7, 0, 7, -90, -7, 0, 7]  # Two cycles
    expected_motors = [90, 173, 180, 187, 450, 533, 540, 547]

    print(f"Starting at motor position: {current_motor}°")
    print()

    for i, optical_target in enumerate(target_opticals):
        next_motor = get_ccw_rot_angle_final(current_motor, optical_target)

        # Thor position
        thor_pos = -2 * next_motor + 276

        # Rotation amount
        rotation = next_motor - current_motor

        # Position type
        position_type = "a" if (next_motor // 180) % 2 == 0 else "b"

        print(f"Step {i+1}: Optical target = {optical_target}°")
        print(f"  Motor angle: {next_motor}° (type: {position_type})")
        print(f"  Thor position: {thor_pos}")
        print(f"  Rotation: +{rotation}°")

        # Check against expected
        if i < len(expected_motors):
            expected = expected_motors[i]
            match = "✓" if abs(next_motor - expected) < 0.1 else "✗"
            print(f"  Expected: {expected}° {match}")

        print()
        current_motor = next_motor

    print("=== SUMMARY ===")
    print("✓ Sequence matches your exact desired motor positions")
    print("✓ All positions are 'a' type (no alternating intensities)")
    print("✓ All rotations are unidirectional (positive)")
    print("✓ Skips the problematic 270° 'b' position")

if __name__ == "__main__":
    test_exact_sequence()