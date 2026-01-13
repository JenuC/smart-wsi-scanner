# Quick Fix: cv2.cvtColor 16-bit Data Loss

## Problem

**File**: `src/dev_tests/ppm_birefringence_maximization_test.py`
**Lines**: 578-580 and 399-400
**Issue**: `cv2.cvtColor()` converts uint16 RGB to uint8 grayscale, causing quantization artifacts

## Symptoms

- Stepwise curves in angle vs. signal plots
- Normalized difference values clustered at whole numbers or X.5 values
- Loss of precision in birefringence angle optimization

## Root Cause

```python
# This line converts uint16 to uint8 internally!
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

OpenCV's cvtColor scales 16-bit RGB down to 8-bit before grayscale conversion, losing 99.6% of the dynamic range (65536 -> 256 levels).

## Fix

### Location 1: Lines 578-580

**BEFORE**:
```python
if len(pos_img.shape) == 3:
    pos_img = cv2.cvtColor(pos_img, cv2.COLOR_BGR2GRAY)
if len(neg_img.shape) == 3:
    neg_img = cv2.cvtColor(neg_img, cv2.COLOR_BGR2GRAY)
```

**AFTER**:
```python
if len(pos_img.shape) == 3:
    # Manual grayscale conversion preserving 16-bit precision
    # OpenCV uses: 0.299*R + 0.587*G + 0.114*B
    pos_img = (0.299 * pos_img[:,:,2].astype(np.float32) +
               0.587 * pos_img[:,:,1].astype(np.float32) +
               0.114 * pos_img[:,:,0].astype(np.float32))
    pos_img = np.clip(pos_img, 0, 65535).astype(np.uint16)

if len(neg_img.shape) == 3:
    neg_img = (0.299 * neg_img[:,:,2].astype(np.float32) +
               0.587 * neg_img[:,:,1].astype(np.float32) +
               0.114 * neg_img[:,:,0].astype(np.float32))
    neg_img = np.clip(neg_img, 0, 65535).astype(np.uint16)
```

### Location 2: Line 400

**BEFORE**:
```python
if len(img.shape) == 3:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

**AFTER**:
```python
if len(img.shape) == 3:
    # Manual grayscale conversion preserving 16-bit precision
    img = (0.299 * img[:,:,2].astype(np.float32) +
           0.587 * img[:,:,1].astype(np.float32) +
           0.114 * img[:,:,0].astype(np.float32))
    img = np.clip(img, 0, 65535).astype(np.uint16)
```

## Expected Results

After fixing:
- Smooth angle vs. signal curves
- Full 16-bit precision in normalized differences
- More accurate optimal angle detection
- Eliminated quantization artifacts

## Testing

```bash
# Re-run the test
cd /home/msnelson/QPSC_Project/smart-wsi-scanner
python src/dev_tests/ppm_birefringence_maximization_test.py

# Check output plots for smooth curves
# Verify normalized difference images have full dynamic range
```

## Other Files to Check

The following files also use `cv2.cvtColor()` and should be audited:

- `src/smart_wsi_scanner/qp_utils.py`
- `src/dev_tests/ppm_rotation_sensitivity_analysis.py`
- `src/dev_tests/check_background_intensities.py`
- `src/dev_tests/diagnose_biref_issue.py`
- `src/dev_tests/test_tissue_batch.py`
- `src/dev_tests/test_tissue_detection_debug.py`
- `src/dev_tests/test_tissue_detection.py`
- `src/smart_wsi_scanner/swsi_empty_region_detection.py`
- `src/smart_wsi_scanner/debayering/src/main_cpu.py`
- `src/smart_wsi_scanner/qp_text_pipeline.py`

For each, check:
1. Are images uint16? (not uint8)
2. Is cvtColor used for grayscale conversion?
3. If yes to both, apply the same fix

## References

- Full analysis: `BIT_DEPTH_ANALYSIS_REPORT.md`
- OpenCV GitHub Issue #9742: "cvtColor converts 16-bit to 8-bit"
- Known limitation of cv2.cvtColor with high bit depth images
