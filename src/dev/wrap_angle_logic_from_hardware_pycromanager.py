# def wrap_angle_m180_p180(self, angle_deg):
#     """
#     Wrap an angle in degrees to the range [-180, 180)
#     """
#     wrapped = (angle_deg + 180) % 360 - 180
#     return wrapped

# def get_ccw_rot_angle(self, theta, is_sequence_start=False):
#     """
#     Get counter clockwise rotation angle maintaining polarization state consistency.

#     CRITICAL: Within an acquisition sequence (e.g., -90°, -7°, 0°, 7° for one tile),
#     ALL angles must maintain the same optical polarization state to avoid
#     alternating light/dark intensities. Large rotations that flip the optical
#     element are ONLY allowed between acquisition sequences.

#     Args:
#         theta: Target optical angle (e.g., -90, -7, 0, 7)
#         is_sequence_start: True if starting new acquisition sequence (allows large rotation)

#     Returns:
#         Next PPM tick value for motor positioning
#     """
#     current_angle_wrapped = self.get_psg_ticks()
#     current_angle = self.psg_angle

#     # Convert optical angle to PPM ticks (base positions)
#     # Handle special known angles first
#     special_angles = {
#         -90: 90,  # -90° optical -> 90 PPM ticks
#         90: 90,  # 90° optical -> 90 PPM ticks (same as -90°, 90° away from 0°)
#         -7: 173,  # -7° optical -> 173 PPM ticks (180 - 7)
#         0: 180,  # 0° optical -> 180 PPM ticks (or 0, but using 180)
#         7: 7,  # 7° optical -> 7 PPM ticks
#     }

#     if theta in special_angles:
#         target_ppm_ticks = special_angles[theta]
#     else:
#         # For other angles, calculate PPM ticks
#         # Convert angle to equivalent PPM ticks (0-179 range)
#         if theta < 0:
#             # Negative angles: convert to positive equivalent
#             target_ppm_ticks = 180 + theta  # e.g., -5 -> 175
#         else:
#             # Positive angles: use as-is but ensure within range
#             target_ppm_ticks = theta % 180

#     if is_sequence_start:
#         # Starting new acquisition sequence - force "a" polarization state
#         # "a" positions are in even-numbered 360° cycles: 0-359, 720-1079, 1440-1799, etc.

#         # Find the next even-numbered cycle (360° period)
#         current_cycle = current_angle // 360

#         # Target cycle should be even (for "a" position)
#         if current_cycle % 2 != 0:
#             # Currently in odd cycle ("b"), move to next even cycle ("a")
#             target_cycle = current_cycle + 1
#         else:
#             # Currently in even cycle ("a")
#             target_cycle = current_cycle

#         # Calculate candidate position in target cycle
#         candidate = target_ppm_ticks + (target_cycle * 360)

#         # If we've already passed this position, move to next even cycle
#         if candidate <= current_angle:
#             target_cycle += 2 if target_cycle % 2 == 0 else 1
#             candidate = target_ppm_ticks + (target_cycle * 360)

#         return candidate

#     else:
#         # Within acquisition sequence - stay in the same 360° cycle
#         # This ensures all angles in the sequence maintain the same polarization state

#         current_cycle = current_angle // 360
#         candidate = target_ppm_ticks + (current_cycle * 360)

#         # If we can't reach this angle in the current cycle (already passed it),
#         # we have a problem - the sequence should be designed to avoid this
#         if candidate <= current_angle:
#             # This should not happen in a well-designed sequence, but handle it
#             logger.warning(
#                 f"Angle sequence issue: target {target_ppm_ticks} in cycle {current_cycle} "
#                 f"would go backwards from {current_angle} to {candidate}"
#             )
#             # Stay in same cycle but move to next logical position
#             candidate = current_angle + (target_ppm_ticks % 180)

#         return candidate

# def _ppm_set_psgticks(self, theta: float, is_sequence_start: bool = False) -> None:
#     """Set the PPM rotation stage to a specific angle."""
#     # Try to get rotation stage device from settings
#     rotation_device = self.rotation_device
#     new_theta = self.get_ccw_rot_angle(theta, is_sequence_start=is_sequence_start)
#     theta_thor = ppm_psgticks_to_thor(new_theta)
#     current_pos_thor = self.core.get_position(rotation_device)

#     self.core.set_position(rotation_device, theta_thor)
#     self.core.wait_for_device(rotation_device)
#     # time.sleep(0.15)
#     # assert theta_thor < current_pos_thor
#     n_repeats = 5
#     while int(self.core.get_position(rotation_device)) != int(theta_thor):
#         logger.warning(
#             f"Rotation stage failed to reach target: requested {theta_thor}, "
#             f"current {self.core.get_position(rotation_device)}"
#         )
#         self.core.set_position(rotation_device, theta_thor)
#         self.core.wait_for_device(rotation_device)
#         time.sleep(0.15)
#         n_repeats -= 1
#         if n_repeats <= 0:
#             # raise RuntimeError("Rotation stage failed to reach target position after retries")
#             break

#     logger.info(f"Set rotation angle to {theta}° (Thor position: {theta_thor})")
#     print(
#         f"[PPM Rotation Stage] Requested: {theta}°, "
#         f"CCW-adjusted: {new_theta}°, "
#         f"Current (Thor): {current_pos_thor}, "
#         f"Target (Thor): {theta_thor}"
#     )

# def _ppm_get_psgticks(self) -> float:
#     """Get the current PPM rotation angle."""
#     rotation_device = self.rotation_device
#     thor_pos = self.core.get_position(rotation_device)
#     self.psg_angle = ppm_thor_to_psgticks(thor_pos)
#     angle_wrapped = self.wrap_angle_m180_p180(self.psg_angle)
#     return angle_wrapped
