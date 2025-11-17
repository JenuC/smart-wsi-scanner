#!/usr/bin/env python3
"""
Verify the final fixed rotation function matches the exact desired sequence.
"""

def get_ccw_rot_angle_implemented(current_angle, theta):
    """
    Implementation matching the updated hardware_pycromanager.py file.
    """
    # Mapping from optical angles to their base motor positions (in "a" positions)
    optical_to_motor = {
        -90: 90,   # -90° optical -> 90° motor
        -7: 173,   # -7° optical -> 173° motor
        0: 180,    # 0° optical -> 180° motor
        7: 187     # 7° optical -> 187° motor
    }

    if theta not in optical_to_motor:
        # Fallback for other angles - use original logic
        target_motor_angle = abs(theta * 2) % 180
        cycles_passed = current_angle // 360
        candidate = target_motor_angle + (cycles_passed * 360)
        if candidate <= current_angle:
            candidate += 360
        while (candidate // 180) % 2 != 0:
            candidate += 360
        return candidate

    base_motor_angle = optical_to_motor[theta]

    # Find which 360° cycle we should target
    # Try the target in the current cycle first
    current_cycle = current_angle // 360
    candidate = base_motor_angle + (current_cycle * 360)

    # If we've already passed this position, move to the next cycle
    if candidate <= current_angle:
        candidate = base_motor_angle + ((current_cycle + 1) * 360)

    return candidate

def test_exact_match():
    """Test if we get the exact sequence you want."""

    print("=== FINAL VERIFICATION ===")
    print("Your desired sequence:")
    print("Optical: -90, -7, 0, 7, -90, -7, 0, 7 (degrees)")
    print("Motor:    90, 173, 180, 187, 450, 533, 540, 547 (motor degrees / 'ticks')")
    print()

    # Start at position 90 (already there for first -90°)
    current_motor = 90.0
    target_opticals = [-90, -7, 0, 7, -90, -7, 0, 7]
    expected_motors = [90, 173, 180, 187, 450, 533, 540, 547]

    results = []

    print(f"Starting position: {current_motor}° motor")
    print()

    for i, optical_target in enumerate(target_opticals):
        next_motor = get_ccw_rot_angle_implemented(current_motor, optical_target)

        # Position type check
        position_type = "a" if (next_motor // 180) % 2 == 0 else "b"

        # Rotation amount
        rotation = next_motor - current_motor
        thor_pos = -2 * next_motor + 276

        print(f"Step {i+1}: {optical_target}° optical")
        print(f"  Current: {current_motor}° -> Next: {next_motor}° (type: {position_type})")
        print(f"  Rotation: +{rotation}° | Thor: {thor_pos}")

        # Check against expected
        if i < len(expected_motors):
            expected = expected_motors[i]
            match = "✓ MATCH" if abs(next_motor - expected) < 0.1 else f"✗ Expected {expected}°"
            print(f"  {match}")

        results.append((optical_target, next_motor, position_type, rotation >= 0))
        print()

        current_motor = next_motor

    # Final verification
    print("=== RESULTS SUMMARY ===")
    all_positive_rotation = all(r[3] for r in results)
    all_a_positions = all(r[2] == "a" for r in results)
    sequence_matches = all(abs(results[i][1] - expected_motors[i]) < 0.1 for i in range(len(expected_motors)))

    print(f"✓ All rotations unidirectional (positive): {all_positive_rotation}")
    print(f"✓ All positions are 'a' type: {all_a_positions}")
    print(f"✓ Sequence matches exactly: {sequence_matches}")

    if all_positive_rotation and all_a_positions and sequence_matches:
        print("\n🎉 SUCCESS: The rotation fix will solve your alternating intensity problem!")
    else:
        print("\n⚠️  Issue detected - further refinement needed.")

if __name__ == "__main__":
    test_exact_match()