# Tissue Detection Tuning Guide

## Quick Start

When adaptive autofocus is failing with `texture=0.0000`, you need to tune the tissue detection thresholds.

### Step 1: Collect Test Image

From your failed acquisition logs, find an image that was rejected:
```
2025-10-01 10:57:22,080 - __main__ - WARNING - Insufficient tissue at position 8 - deferring autofocus
2025-10-01 10:57:22,080 - __main__ - WARNING -   Tissue stats: texture=0.0000 (threshold=0.0150), area=0.000 (threshold=0.150)
```

The image is likely saved in your acquisition directory at:
```
D:\2025QPSC\data\ExistingImageTrial\ppm_10x_1\Raw\<position>\90.0\<image>.tif
```

### Step 2: Run Debug Script

```bash
# On Windows system (where the data is):
cd smart-wsi-scanner
python test_tissue_detection_debug.py "D:\2025QPSC\data\ExistingImageTrial\ppm_10x_1" --find

# Or test specific image:
python test_tissue_detection_debug.py "D:\path\to\failed\image.tif"
```

### Step 3: Interpret Results

The script will show you:

1. **Visual Analysis**
   - Original image vs what the algorithm sees
   - Gradient maps showing detected texture
   - Different tissue masks at various intensity ranges
   - Histograms of intensity and gradient distributions

2. **Threshold Test Matrix**
   - Tests multiple combinations of texture and area thresholds
   - Shows which combinations PASS vs FAIL
   - Helps you understand the sensitivity

3. **Recommendations**
   - Suggested texture threshold
   - Suggested area threshold
   - Best tissue mask strategy for your modality

### Step 4: Apply Settings

The script will output code you can paste into `qp_acquisition.py`. For example:

```python
# In qp_acquisition.py around line 474:
has_tissue, tissue_stats = AutofocusUtils.has_sufficient_tissue(
    test_img,
    texture_threshold=0.008,  # <-- Recommended value
    tissue_area_threshold=0.12,  # <-- Recommended value
    modality=modality,
    logger=logger,
    return_stats=True
)
```

## Understanding the Parameters

### `texture_threshold` (default: 0.015 for PPM, 0.02 for others)

Controls the minimum texture variance required. Lower = more sensitive (detects less texture).

**What it measures:** Standard deviation of the gradient magnitude within tissue regions.

**Symptoms:**
- Too high: Rejects images that actually have tissue → autofocus never runs
- Too low: Accepts blank/background images → poor autofocus

**Typical ranges:**
- PPM (polarized): 0.005 - 0.020
- Brightfield: 0.010 - 0.030
- High contrast (SHG): 0.015 - 0.040

### `tissue_area_threshold` (default: 0.15)

Controls the minimum fraction of the image that must contain tissue. Lower = accepts images with less coverage.

**What it measures:** Fraction of pixels within the tissue intensity range (e.g., 0.05-0.95 for PPM).

**Symptoms:**
- Too high: Rejects images where tissue only covers part of FOV
- Too low: Accepts images with tiny tissue fragments

**Typical ranges:**
- Sparse tissue: 0.05 - 0.15
- Moderate coverage: 0.15 - 0.25
- Dense tissue: 0.25 - 0.40

## Common Issues

### Issue 1: `texture=0.0000, area=0.000`

**Cause:** The image is either:
1. Completely out of focus (blurred to uniform gray)
2. Exposure too dark (all black) or bright (all white)
3. Background only (no sample in FOV)
4. Wrong modality parameters (e.g., using brightfield settings for PPM)

**Solution:**
- Check exposure settings at 90° (autofocus angle)
- Verify rotation stage is actually moving to 90°
- Test with much lower thresholds (texture=0.005, area=0.05)
- Capture image at autofocus position and run debug script

### Issue 2: Texture detected but area too low

**Symptom:** `texture=0.0250 (threshold=0.0150), area=0.08 (threshold=0.150)`

**Cause:** Tissue is present but doesn't cover enough of the FOV, or the intensity mask is too narrow.

**Solution:**
- Lower `tissue_area_threshold` to 0.10 or 0.05
- Check if modality is correctly detected (PPM uses wider intensity range)
- Verify tissue is actually in the FOV at this position

### Issue 3: Detection works for some tiles but not others

**Symptom:** Autofocus runs on some tiles, defers on others, even though all have tissue.

**Cause:** Variable tissue coverage or staining intensity across the sample.

**Solution:**
- Use more conservative thresholds (lower both texture and area)
- Consider position-specific logic if edges have less tissue
- Check if background correction is affecting texture detection

## Advanced: Modality-Specific Adjustments

The tissue detection automatically adjusts for different modalities via the `modality` parameter. Current settings:

### PPM (Polarized Light)
```python
tissue_mask_range = (0.05, 0.95)  # Wide range for birefringent structures
texture_threshold = 0.015  # Slightly more sensitive than default
```

### Brightfield
```python
tissue_mask_range = (0.15, 0.85)  # Typical tissue staining intensity
texture_threshold = 0.02  # Standard setting
```

### SHG / Multiphoton
```python
tissue_mask_range = (0.1, 0.9)
texture_threshold = 0.025  # Less sensitive (sparse features)
```

## Testing Without Acquisition

You can test threshold changes without running a full acquisition:

```python
from smart_wsi_scanner.qp_utils import AutofocusUtils
import tifffile as tf

# Load your test image
img = tf.imread("path/to/test/image.tif")

# Test different thresholds
for tex in [0.005, 0.01, 0.015, 0.02]:
    for area in [0.05, 0.10, 0.15, 0.20]:
        result = AutofocusUtils.has_sufficient_tissue(
            img,
            texture_threshold=tex,
            tissue_area_threshold=area,
            modality="ppm"
        )
        print(f"tex={tex:.3f}, area={area:.2f} → {'PASS' if result else 'FAIL'}")
```

## Configuration File (Future Enhancement)

Consider adding these to your YAML config:

```yaml
autofocus:
  tissue_detection:
    modality_overrides:
      ppm_10x:
        texture_threshold: 0.012
        area_threshold: 0.10
      ppm_40x:
        texture_threshold: 0.015
        area_threshold: 0.15
```

## Debug Output During Acquisition

To enable detailed tissue detection logging during acquisition, the system already logs:

```
INFO - Checking for autofocus at position X: ...
WARNING - Insufficient tissue at position X - deferring autofocus
WARNING -   Tissue stats: texture=X.XXXX (threshold=X.XXXX), area=X.XXX (threshold=X.XXX)
INFO - Deferring autofocus from tile X to tile Y
```

If you need MORE detail, add `return_stats=True` to the `has_sufficient_tissue()` call (already present).

## Contact

If tuning still doesn't work after trying the debug script, check:

1. Is the microscope actually acquiring images? (Check raw file output)
2. Are images completely blank or uniform?
3. Is background correction removing all texture?
4. Is the rotation stage moving to 90° for autofocus?
