#!/usr/bin/env python3
"""
Test the final corrected rotation logic.
"""

def get_ccw_rot_angle_final(current_angle, theta, is_sequence_start=False):
    """Final corrected version matching the hardware file."""

    # Convert optical angle to PPM ticks (base positions)
    special_angles = {
        -90: 90, 90: 90, -7: 173, 0: 180, 7: 7
    }

    if theta in special_angles:
        target_ppm_ticks = special_angles[theta]
    else:
        if theta < 0:
            target_ppm_ticks = 180 + theta
        else:
            target_ppm_ticks = theta % 180

    if is_sequence_start:
        # Starting new acquisition sequence - ensure we're in "a" polarization state
        # Calculate how many complete cycles we've done (each cycle = 360° motor rotation)
        current_cycle = current_angle // 360

        # Find the next "a" position for this target
        candidate = target_ppm_ticks + (current_cycle * 360)

        # If we've passed this position in current cycle, move to next cycle
        if candidate <= current_angle:
            candidate = target_pmp_ticks + ((current_cycle + 1) * 360)

        # Ensure this is an "a" position using the specific pattern
        # "a" positions: 0-179, 360-539, 720-899, 1080-1259, etc.
        # This means: cycle_number % 2 == 0
        target_cycle = candidate // 360
        if target_cycle % 2 != 0:
            # We're in a "b" cycle, move to next "a" cycle
            candidate += 360

        return candidate
    else:
        # Within acquisition sequence - maintain current polarization state
        current_state = "a" if (current_angle // 360) % 2 == 0 else "b"
        current_cycle = current_angle // 360
        candidate = target_ppm_ticks + (current_cycle * 360)
        candidate_state = "a" if (candidate // 360) % 2 == 0 else "b"

        if candidate_state != current_state:
            if current_state == "a":
                while (candidate // 360) % 2 != 0:
                    candidate += 360
            else:
                while (candidate // 360) % 2 == 0:
                    candidate += 360

        if candidate <= current_angle:
            candidate += 360

        return candidate

def test_corrected_sequence():
    """Test the corrected sequence based on the actual log data."""

    print("=== TESTING CORRECTED SEQUENCE ===")
    print("Based on your actual log starting around 2610°")
    print()

    # Start near where your log started
    current_angle = 2610.0

    angles = [90.0, -7.0, 0.0, 7.0] * 4  # Four positions

    print("Expected pattern:")
    print("Position 1: All angles → 'a' state (like 0.0, 173.0, 180.0, 7.0)")
    print("Position 2: All angles → 'a' state (like 0.0, 173.0, 180.0, 7.0)")
    print("Position 3: All angles → 'a' state (like 0.0, 173.0, 180.0, 7.0)")
    print("Position 4: All angles → 'a' state (like 0.0, 173.0, 180.0, 7.0)")
    print()

    for pos in range(4):
        print(f"Position {pos + 1}:")

        for angle_idx, angle in enumerate([90.0, -7.0, 0.0, 7.0]):
            is_start = (angle_idx == 0)
            next_angle = get_ccw_rot_angle_final(current_angle, angle, is_sequence_start=is_start)

            # Determine what the wrapped angle would be
            wrapped = next_angle % 360
            if wrapped > 180:
                wrapped = wrapped - 360

            cycle = next_angle // 360
            position_type = "a" if cycle % 2 == 0 else "b"

            print(f"  {angle}° → {next_angle}° (wrapped: {wrapped}°, cycle: {cycle}, type: {position_type})")

            current_angle = next_angle

        print()

if __name__ == "__main__":
    # Fix the typo first
    def get_ccw_rot_angle_final_corrected(current_angle, theta, is_sequence_start=False):
        special_angles = {-90: 90, 90: 90, -7: 173, 0: 180, 7: 7}

        if theta in special_angles:
            target_ppm_ticks = special_angles[theta]
        else:
            if theta < 0:
                target_ppm_ticks = 180 + theta
            else:
                target_ppm_ticks = theta % 180

        if is_sequence_start:
            current_cycle = current_angle // 360
            candidate = target_ppm_ticks + (current_cycle * 360)

            if candidate <= current_angle:
                candidate = target_ppm_ticks + ((current_cycle + 1) * 360)

            target_cycle = candidate // 360
            if target_cycle % 2 != 0:
                candidate += 360

            return candidate
        else:
            current_state = "a" if (current_angle // 360) % 2 == 0 else "b"
            current_cycle = current_angle // 360
            candidate = target_ppm_ticks + (current_cycle * 360)
            candidate_state = "a" if (candidate // 360) % 2 == 0 else "b"

            if candidate_state != current_state:
                if current_state == "a":
                    while (candidate // 360) % 2 != 0:
                        candidate += 360
                else:
                    while (candidate // 360) % 2 == 0:
                        candidate += 360

            if candidate <= current_angle:
                candidate += 360

            return candidate

    # Replace function with corrected version
    get_ccw_rot_angle_final = get_ccw_rot_angle_final_corrected

    test_corrected_sequence()