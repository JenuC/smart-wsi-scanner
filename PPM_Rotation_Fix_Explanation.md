# PPM Polarizer Rotation Fix: Technical Explanation

**Date:** 2025-09-25
**Issue:** Alternating light/dark tiles during polarized light microscopy acquisition
**Root Cause:** Unintentional optical element state changes during rotation sequence
**Solution:** Modified rotation logic to maintain polarization state consistency

## Problem Description

### Symptoms
- Alternating bright and dark tiles during multi-angle PPM acquisition
- Inconsistent image intensities even when acquiring the same optical angles
- Pattern described as "a" (normal) and "b" (dimmer) position variations

### Physical Setup
- **Thorlabs motor** drives gear system connected to **Polarization State Generator (PSG)**
- **Motor-to-optical ratio**: 2:1 (360° motor rotation = 180° optical element rotation)
- **PPM tick system**: 0-179 ticks represent full 180° optical rotation
- **Coordinate systems**:
  - PPM ticks: 0-179 (displayed in software)
  - Motor positions: Thor Kinesis positions (inverse relationship)
  - Optical angles: Actual polarization angles (-90°, -7°, 0°, 7°)

### Conversion Relationships
```python
# PPM ticks to Thor motor position
thor_pos = -2 * ppm_ticks + 276

# Thor motor position to PPM ticks
ppm_ticks = (276 - thor_pos) / 2
```

## Root Cause Analysis

### The "a" vs "b" Position Problem

The optical element (PSG) can achieve the same optical polarization angle at **two different physical orientations**:

1. **"a" positions**: Element in "forward" orientation
   - Occurs when: `(ppm_ticks // 180) % 2 == 0`
   - Examples: 0-179 ticks, 360-539 ticks, 720-899 ticks...

2. **"b" positions**: Element in "backward" orientation
   - Occurs when: `(pmp_ticks // 180) % 2 != 0`
   - Examples: 180-359 ticks, 540-719 ticks, 900-1079 ticks...

**Key Insight**: Even though both positions achieve the same optical angle reading, they have different optical properties due to the physical orientation of the birefringent element.

### Original Problematic Code

In `hardware_pycromanager.py`, line 549-550:
```python
if theta == 90:
    theta = -1 * 90  # This line caused alternating a/b positions!
```

This line caused the rotation algorithm to alternate between "a" and "b" positions, creating the intensity variations observed in acquired images.

## Solution Implementation

### Key Requirements

1. **Polarization State Consistency**: Within a single acquisition sequence (e.g., -90°, -7°, 0°, 7° for one tile), ALL angles must maintain the same optical polarization state.

2. **Unidirectional Rotation**: Always rotate in the same direction to avoid gear backlash and ensure repeatability.

3. **Sequence Flexibility**: Support both full sequences (-90°, -7°, 0°, 7°) and partial sequences (-7°, 7°) while maintaining state consistency.

### Modified Function Signature

```python
def get_ccw_rot_angle(self, theta, is_sequence_start=False):
    """
    Args:
        theta: Target optical angle (-90, -7, 0, 7)
        is_sequence_start: True when starting new acquisition sequence
    """
```

### Logic Flow

#### 1. Optical Angle to PPM Tick Conversion
```python
optical_to_ppm_ticks = {
    -90: 90,   # -90° optical -> 90 PPM ticks
    -7: 173,   # -7° optical -> 173 PPM ticks (180 - 7)
    0: 180,    # 0° optical -> 180 PPM ticks
    7: 7       # 7° optical -> 7 PPM ticks
}
```

#### 2. Sequence Start Behavior (`is_sequence_start=True`)
- **Purpose**: Moving to new tile/position - can do large rotation
- **Action**: Find next "a" position for target angle
- **Result**: Ensures consistent "a" state for entire upcoming sequence

#### 3. Within-Sequence Behavior (`is_sequence_start=False`)
- **Purpose**: Within single tile acquisition - maintain current state
- **Action**: Find target angle while preserving current polarization state
- **Result**: Small rotations only (no optical element flipping)

### Expected Rotation Sequences

#### Scenario 1: Full Sequence Acquisition
**Tile 1**: 90 → 173 → 180 → 7 (small rotations, same "a" state)
**Move to Tile 2**: 7 → 450 (large rotation, reset to "a" state)
**Tile 2**: 450 → 533 → 540 → 367 (small rotations, same "a" state)

#### Scenario 2: Partial Sequence (-7°, 7° only)
**Tile 1**: 173 → 7 (small rotation, same "a" state)
**Move to Tile 2**: 7 → 533 (large rotation, reset to "a" state)
**Tile 2**: 533 → 367 (small rotation, same "a" state)

## Implementation Details

### Modified Function Structure

1. **Input Validation**: Ensure only supported optical angles
2. **Current State Detection**: Determine if currently in "a" or "b" position
3. **Branch Logic**:
   - Sequence start: Force "a" position
   - Within sequence: Maintain current state
4. **Forward Motion**: Ensure unidirectional rotation

### Usage in Acquisition Code

```python
# Starting new tile acquisition
first_angle = get_ccw_rot_angle(-90, is_sequence_start=True)  # Large rotation OK

# Subsequent angles in same sequence
second_angle = get_ccw_rot_angle(-7, is_sequence_start=False)  # Small rotation only
third_angle = get_ccw_rot_angle(0, is_sequence_start=False)   # Small rotation only
fourth_angle = get_ccw_rot_angle(7, is_sequence_start=False)  # Small rotation only
```

## Expected Outcomes

### ✅ Eliminated Issues
- **No more alternating intensities**: All acquisitions use same optical state
- **Consistent image quality**: Predictable optical properties across tiles
- **Reduced gear backlash**: Unidirectional rotation only

### ✅ Maintained Features
- **Flexible angle selection**: Supports partial angle sequences
- **Unidirectional rotation**: Maintains existing backlash avoidance
- **Existing coordinate systems**: No changes to PPM tick or Thor position handling

## Testing and Validation

### Test Cases
1. **Full sequence**: Verify 90→173→180→7→450→533→540→367 pattern
2. **Partial sequence**: Verify proper state maintenance with subset of angles
3. **State consistency**: Confirm no large rotations within acquisition sequence
4. **Image quality**: Validate elimination of alternating bright/dark tiles

### Verification Points
- All rotations within sequence are <97° (largest gap: 173→180→7)
- Large rotations (>97°) only occur between sequences
- All positions maintain consistent "a" polarization state
- Motor positions decrease monotonically (due to inverse PPM→Thor relationship)

## Files Modified

- **`src/smart_wsi_scanner/hardware_pycromanager.py`**: Lines 546-626
  - Modified `get_ccw_rot_angle()` function
  - Added `is_sequence_start` parameter
  - Implemented polarization state preservation logic

## Future Considerations

### Potential Enhancements
1. **Automatic sequence detection**: Infer sequence boundaries from acquisition patterns
2. **State optimization**: Choose optimal starting state based on acquisition pattern
3. **Angle validation**: Runtime verification of small rotation constraint

### Monitoring
- **Image intensity consistency**: Verify elimination of alternating patterns
- **Motor performance**: Monitor for any unexpected rotation behaviors
- **Acquisition timing**: Ensure no significant speed impact from logic changes

---

**Summary**: This fix eliminates alternating light/dark tiles by ensuring all angles within an acquisition sequence maintain the same optical polarization state, while still allowing unidirectional rotation and flexible angle selection.