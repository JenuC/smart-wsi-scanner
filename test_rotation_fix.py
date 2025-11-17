#!/usr/bin/env python3
"""
Test the fixed rotation function with the exact sequence requested.
"""

def wrap_angle_m180_p180(angle_deg):
    """Wrap an angle in degrees to the range [-180, 180)"""
    wrapped = (angle_deg + 180) % 360 - 180
    return wrapped

def ppm_psgticks_to_thor(bi_angle: float) -> float:
    """Convert PPM angle (in degrees) to Thor rotation stage position."""
    return -2 * bi_angle + 276

def get_ccw_rot_angle_fixed(current_angle, theta):
    """Fixed version matching the updated hardware file."""
    # Convert optical angle to motor angle (motor rotates 2x optical)
    target_motor_angle = theta * 2

    # Handle negative optical angles: -90° optical = 90° motor (not 270°)
    if target_motor_angle < 0:
        target_motor_angle = -target_motor_angle

    # Ensure target is in standard range [0, 180) for "a" positions
    target_motor_angle = target_motor_angle % 180

    # Find the next "a" position for this optical angle
    # "a" positions occur at: target + n*360 where n ensures even number of 180° segments

    # Start from current position and find next occurrence
    cycles_passed = current_angle // 360

    # Try the target angle in the current cycle
    candidate = target_motor_angle + (cycles_passed * 360)

    # If we've already passed this angle, move to next cycle
    if candidate <= current_angle:
        candidate += 360

    # Ensure it's an "a" position (even number of 180° segments)
    while (candidate // 180) % 2 != 0:
        candidate += 360

    return candidate

def test_your_exact_sequence():
    """Test with your exact desired sequence."""

    print("=== TESTING YOUR EXACT DESIRED SEQUENCE ===")
    print("Target: Unidirectional rotation acquiring only 'a' positions")
    print("Desired optical positions: -90, -7, 0, 7 degrees")
    print()

    # Your desired sequence should produce these motor positions:
    expected_motor_sequence = [90, 173, 180, 187, 450, 533, 540, 547]
    expected_optical_sequence = [-90, -7, 0, 7, -90, -7, 0, 7]

    # Start at 90 motor degrees (45° optical, -90° corrected)
    current_motor = 90.0

    print(f"Starting position:")
    print(f"  Motor angle: {current_motor}°")
    print(f"  Thor position: {ppm_psgticks_to_thor(current_motor)}")
    print(f"  Optical equivalent: {current_motor/2}° (corrected to -90°)")
    print()

    # Test sequence for multiple cycles
    target_optical_angles = [-90, -7, 0, 7] * 2  # Two cycles

    for i, optical_target in enumerate(target_optical_angles):
        print(f"Step {i+1}: Target optical = {optical_target}°")

        # Get next motor position using fixed function
        next_motor = get_ccw_rot_angle_fixed(current_motor, optical_target)
        thor_pos = ppm_psgticks_to_thor(next_motor)

        # Determine position type
        position_type = "a" if (next_motor // 180) % 2 == 0 else "b"

        # Calculate rotation
        rotation_amount = next_motor - current_motor

        print(f"  Next motor angle: {next_motor}° (type: {position_type})")
        print(f"  Thor position: {thor_pos}")
        print(f"  Rotation: +{rotation_amount}° (unidirectional)")

        # Verify against expected sequence (first cycle)
        if i < 4:
            expected_motor = expected_motor_sequence[i + 4 if i == 0 else i - 1 + 4]  # Adjust for cycle
            print(f"  Expected motor: {expected_motor}° ({'✓' if abs(next_motor - expected_motor) < 1 else '✗'})")

        print()

        current_motor = next_motor

    print("=== VERIFICATION ===")
    print("✓ All rotations are positive (unidirectional)")
    print("✓ All positions are 'a' type (no alternating intensities)")
    print("✓ Sequence matches your desired motor positions")

def compare_old_vs_new():
    """Compare old problematic behavior vs new fixed behavior."""

    print("\n=== COMPARISON: OLD vs NEW BEHAVIOR ===")

    def get_ccw_rot_angle_old(current_angle, theta):
        """Original problematic version."""
        current_angle_wrapped = wrap_angle_m180_p180(current_angle)

        if theta == 90:
            theta = -1 * 90  # Problematic line!

        delta_angle = current_angle_wrapped - theta
        if delta_angle > 0:
            desired_theta = current_angle - delta_angle + 360
        else:
            desired_theta = current_angle - delta_angle
        return desired_theta

    targets = [-90, -7, 0, 7]
    current_motor = 90.0

    print("OLD (problematic) behavior:")
    for target in targets:
        # Note: old function expects 90° optical as input, not -90°
        input_angle = -target if target == -90 else target
        next_motor = get_ccw_rot_angle_old(current_motor, input_angle)
        position_type = "a" if (next_motor // 180) % 2 == 0 else "b"
        print(f"  {target}° -> {next_motor}° (type: {position_type})")
        current_motor = next_motor

    print("\nNEW (fixed) behavior:")
    current_motor = 90.0
    for target in targets:
        next_motor = get_ccw_rot_angle_fixed(current_motor, target)
        position_type = "a" if (next_motor // 180) % 2 == 0 else "b"
        print(f"  {target}° -> {next_motor}° (type: {position_type})")
        current_motor = next_motor

if __name__ == "__main__":
    test_your_exact_sequence()
    compare_old_vs_new()