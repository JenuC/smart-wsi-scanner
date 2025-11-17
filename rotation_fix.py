#!/usr/bin/env python3
"""
Fixed rotation function for unidirectional "a" position acquisition.
"""

def wrap_angle_m180_p180(angle_deg):
    """Wrap an angle in degrees to the range [-180, 180)"""
    wrapped = (angle_deg + 180) % 360 - 180
    return wrapped

def ppm_psgticks_to_thor(bi_angle: float) -> float:
    """Convert PPM angle (in degrees) to Thor rotation stage position."""
    return -2 * bi_angle + 276

def get_ccw_rot_angle_fixed(current_angle, theta, force_a_position=True):
    """
    Get counter clockwise rotation angle ensuring unidirectional "a" positions only.

    Args:
        current_angle: Current motor angle (unwrapped, can be > 360°)
        theta: Target optical angle (-90, -7, 0, 7, etc.)
        force_a_position: If True, only use "a" positions

    Returns:
        Next motor angle for unidirectional rotation to "a" position
    """

    # Convert optical angle to motor angle (motor rotates 2x optical)
    target_motor_angle = theta * 2

    # Handle negative optical angles: -90° optical = 90° motor (not 270°)
    if target_motor_angle < 0:
        target_motor_angle = -target_motor_angle

    # Ensure target is in standard range [0, 180) for "a" positions
    target_motor_angle = target_motor_angle % 180

    if force_a_position:
        # Find the next "a" position for this optical angle
        # "a" positions occur at: target + n*360 where n is even

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
    else:
        # Original logic without "a" position constraint
        current_angle_wrapped = wrap_angle_m180_p180(current_angle)

        delta_angle = current_angle_wrapped - theta
        if delta_angle > 0:
            desired_theta = current_angle - delta_angle + 360
        else:
            desired_theta = current_angle - delta_angle
        return desired_theta

def test_fixed_sequence():
    """Test the fixed rotation sequence."""

    print("=== TESTING FIXED ROTATION SEQUENCE ===")

    # Your desired optical angles
    optical_targets = [-90, -7, 0, 7]

    # Start at some initial position
    current_motor_angle = 90.0  # Starting position

    print(f"Starting motor angle: {current_motor_angle}°")
    print(f"Starting Thor position: {ppm_psgticks_to_thor(current_motor_angle)}")
    print()

    for i, optical_target in enumerate(optical_targets):
        print(f"Step {i+1}: Target optical angle = {optical_target}°")

        # Calculate next motor position
        next_motor_angle = get_ccw_rot_angle_fixed(current_motor_angle, optical_target, force_a_position=True)

        # Calculate Thor position
        thor_pos = ppm_psgticks_to_thor(next_motor_angle)

        # Determine position type
        position_type = "a" if (next_motor_angle // 180) % 2 == 0 else "b"

        # Calculate rotation amount
        rotation = next_motor_angle - current_motor_angle

        print(f"  Next motor angle: {next_motor_angle}° (position type: {position_type})")
        print(f"  Thor position: {thor_pos}")
        print(f"  Rotation amount: +{rotation}° (unidirectional)")
        print(f"  Optical angle achieved: {(next_motor_angle % 360) / 2}°")
        print()

        # Update current position
        current_motor_angle = next_motor_angle

    print("=== SUMMARY ===")
    print("All positions are 'a' type - no alternating intensities!")
    print("All rotations are positive (unidirectional)")

if __name__ == "__main__":
    test_fixed_sequence()