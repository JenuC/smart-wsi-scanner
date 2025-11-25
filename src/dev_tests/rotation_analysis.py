#!/usr/bin/env python3
"""
Analysis of PPM rotation patterns to understand alternating intensities.
"""

def ppm_psgticks_to_thor(bi_angle: float) -> float:
    """Convert PPM angle (in degrees) to Thor rotation stage position."""
    return -2 * bi_angle + 276

def ppm_thor_to_psgticks(kinesis_pos: float) -> float:
    """Convert Thor rotation stage position to PPM angle (in degrees)."""
    return (276 - kinesis_pos) / 2

def wrap_angle_m180_p180(angle_deg):
    """Wrap an angle in degrees to the range [-180, 180)"""
    wrapped = (angle_deg + 180) % 360 - 180
    return wrapped

def get_ccw_rot_angle(current_angle, theta):
    """Get counter clockwise rotation angle (simplified version)"""
    current_angle_wrapped = wrap_angle_m180_p180(current_angle)
    if theta == 90:
        theta = -1 * 90  # This is the problematic line!

    delta_angle = current_angle_wrapped - theta
    if delta_angle > 0:
        desired_theta = current_angle - delta_angle + 360
    else:
        desired_theta = current_angle - delta_angle
    return desired_theta

def analyze_rotation_sequence():
    """Analyze the rotation sequence to understand alternating intensities."""

    print("=== ROTATION ANALYSIS ===")
    print("Understanding the alternating intensity problem\n")

    # Your desired sequence in "ticks" (double angles)
    desired_ticks = [90, 173, 180, 187, 270, 450, 533, 540, 547]
    desired_optical = [angle/2 for angle in desired_ticks]  # Convert to optical degrees

    print("Desired sequence (ticks -> optical degrees):")
    for tick, opt in zip(desired_ticks, desired_optical):
        print(f"  {tick} ticks -> {opt}° optical")

    print("\n=== PROBLEM ANALYSIS ===")

    # Simulate what the current code does
    print("Current code behavior with get_ccw_rot_angle:")
    current_psg_angle = 90  # Start at 90 "ticks"

    target_angles = [-90, -7, 0, 7]  # Your desired optical angles

    for i, target in enumerate(target_angles):
        print(f"\nStep {i+1}: Target = {target}°")

        # Current code converts 90 to -90
        if target == 90:
            adjusted_target = -90
        else:
            adjusted_target = target

        print(f"  Adjusted target: {adjusted_target}°")

        # Calculate CCW rotation
        current_wrapped = wrap_angle_m180_p180(current_psg_angle)
        print(f"  Current angle (wrapped): {current_wrapped}°")

        delta = current_wrapped - adjusted_target
        print(f"  Delta: {delta}°")

        if delta > 0:
            new_angle = current_psg_angle - delta + 360
            print(f"  New angle: {current_psg_angle} - {delta} + 360 = {new_angle}°")
        else:
            new_angle = current_psg_angle - delta
            print(f"  New angle: {current_psg_angle} - {delta} = {new_angle}°")

        # Convert to Thor position
        thor_pos = ppm_psgticks_to_thor(new_angle)
        print(f"  Thor position: {thor_pos}")

        # Check if this causes "a" vs "b" position issue
        optical_angle = new_angle / 2
        position_type = "a" if (new_angle // 180) % 2 == 0 else "b"
        print(f"  Optical angle: {optical_angle}° (position type: {position_type})")

        current_psg_angle = new_angle

def analyze_alternating_pattern():
    """Analyze why we get alternating a/b positions."""

    print("\n=== ALTERNATING PATTERN ANALYSIS ===")
    print("Motor rotations and optical positions:")

    # The key insight: every 180° of motor rotation gives same optical angle
    # but from opposite sides of the birefringent element

    angles_to_test = [90, 270, 450, 630]  # Same optical angle (45°) at different motor positions

    for angle in angles_to_test:
        optical = angle / 2
        motor_revolutions = angle / 360
        position_type = "a" if (angle // 180) % 2 == 0 else "b"

        print(f"Motor angle: {angle}° -> Optical: {optical}° -> {motor_revolutions:.1f} revs -> Type: {position_type}")

def proposed_solution():
    """Propose a solution for unidirectional rotation."""

    print("\n=== PROPOSED SOLUTION ===")
    print("Modified rotation strategy for unidirectional movement:")

    # Your desired sequence: -90, -7, 0, 7 (all "a" positions)
    desired_optical = [-90, -7, 0, 7]

    # Convert to motor angles, ensuring we only get "a" positions
    # "a" positions occur when motor_angle // 180 is even
    # So we want motor angles: 90, 173, 180, 187 (first cycle)
    # Then: 450, 533, 540, 547 (second cycle, +360°)

    print("Corrected sequence for unidirectional 'a' positions only:")

    base_motor_angles = [90, 173, 180, 187]  # First cycle "a" positions

    for cycle in range(3):  # Show 3 cycles
        print(f"\nCycle {cycle + 1}:")
        for angle in base_motor_angles:
            motor_angle = angle + (cycle * 360)
            optical_angle = (motor_angle % 360) / 2
            if optical_angle > 90:
                optical_angle = optical_angle - 180  # Wrap to [-90, 90] range

            thor_pos = ppm_psgticks_to_thor(motor_angle)
            print(f"  Motor: {motor_angle}° -> Optical: {optical_angle}° -> Thor: {thor_pos}")

def fixed_rotation_function():
    """Provide corrected rotation function."""

    print("\n=== CORRECTED ROTATION FUNCTION ===")

    print("""
def get_ccw_rot_angle_fixed(current_angle, theta):
    '''
    Get counter clockwise rotation angle ensuring unidirectional "a" positions.
    '''
    current_angle_wrapped = wrap_angle_m180_p180(current_angle)

    # DON'T modify the target angle - this was the bug!
    # if theta == 90:
    #     theta = -1 * 90  # <-- REMOVE THIS LINE

    # Instead, ensure we move to the next "a" position for the target angle
    # "a" positions occur when motor_angle // 180 is even

    # Find the next "a" position for the target optical angle
    target_motor_base = theta * 2  # Convert optical to motor angle

    # Ensure we get an "a" position (even multiple of 180°)
    if target_motor_base < 0:
        target_motor_base += 180  # Map negative angles to positive

    # Find next "a" position that's greater than current position
    current_cycle = current_angle // 360
    next_target = target_motor_base + (current_cycle * 360)

    # If we've already passed this angle in current cycle, move to next cycle
    if next_target <= current_angle:
        next_target += 360

    # Ensure it's an "a" position
    while (next_target // 180) % 2 != 0:
        next_target += 180

    return next_target
""")

if __name__ == "__main__":
    analyze_rotation_sequence()
    analyze_alternating_pattern()
    proposed_solution()
    fixed_rotation_function()