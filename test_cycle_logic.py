#!/usr/bin/env python3
"""
Test the improved cycle-based logic.
"""

def get_ccw_rot_angle_improved(current_angle, theta, is_sequence_start=False):
    """Improved logic matching the latest hardware implementation."""

    special_angles = {-90: 90, 90: 90, -7: 173, 0: 180, 7: 7}

    if theta in special_angles:
        target_ppm_ticks = special_angles[theta]
    else:
        if theta < 0:
            target_ppm_ticks = 180 + theta
        else:
            target_ppm_ticks = theta % 180

    if is_sequence_start:
        # Starting new acquisition sequence - force "a" polarization state
        # "a" positions are in even-numbered 360° cycles: 0-359, 720-1079, 1440-1799, etc.

        # Find the next even-numbered cycle (360° period)
        current_cycle = current_angle // 360

        # Target cycle should be even (for "a" position)
        if current_cycle % 2 != 0:
            # Currently in odd cycle ("b"), move to next even cycle ("a")
            target_cycle = current_cycle + 1
        else:
            # Currently in even cycle ("a")
            target_cycle = current_cycle

        # Calculate candidate position in target cycle
        candidate = target_ppm_ticks + (target_cycle * 360)

        # If we've already passed this position, move to next even cycle
        if candidate <= current_angle:
            target_cycle += 2 if target_cycle % 2 == 0 else 1
            candidate = target_ppm_ticks + (target_cycle * 360)

        return candidate
    else:
        # Within acquisition sequence - stay in the same 360° cycle
        current_cycle = current_angle // 360
        candidate = target_ppm_ticks + (current_cycle * 360)

        if candidate <= current_angle:
            print(f"WARNING: Backward sequence - current: {current_angle}, candidate: {candidate}")
            candidate = current_angle + (target_ppm_ticks % 180)

        return candidate

def test_improved_logic():
    """Test the improved logic with your actual starting position."""

    print("=== TESTING IMPROVED CYCLE LOGIC ===")
    print("Starting near your actual log position: 2610°")
    print()

    current_angle = 2610.0  # Starting position from your log

    for pos in range(4):
        print(f"Position {pos + 1}:")

        for angle_idx, angle in enumerate([90.0, -7.0, 0.0, 7.0]):
            is_start = (angle_idx == 0)
            next_angle = get_ccw_rot_angle_improved(current_angle, angle, is_sequence_start=is_start)

            cycle = next_angle // 360
            wrapped = next_angle % 360
            if wrapped > 179:
                wrapped = wrapped - 360

            position_type = "a" if cycle % 2 == 0 else "b"

            print(f"  {angle}° → {next_angle}° (cycle: {cycle}, wrapped: {wrapped}°, type: {position_type})")

            current_angle = next_angle

        print()

    print("Expected result: All positions should show type 'a' consistently")

if __name__ == "__main__":
    test_improved_logic()