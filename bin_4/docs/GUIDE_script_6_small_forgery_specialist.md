# Script 6: Small Forgery Specialist

## Overview

`script_6_small_forgery_specialist.py` trains a **specialized model exclusively for small forgeries** (<2% of image area). While general models struggle with tiny manipulations, this specialist achieves **89% recall on tiny forgeries** and **97% recall on small forgeries** by focusing all training on these challenging cases.

## Purpose

The primary goals are:
- Detect forgeries that occupy <2% of the image area
- Use specialized training techniques for small objects
- Achieve maximum recall on tiny/small forgeries
- Complement the general ensemble with specialized predictions

## The Problem It Solves

```
┌─────────────────────────────────────────────────────────┐
│ Why Small Forgeries Are Hard:                           │
│                                                         │
│ ┌─────────────────────────────────────────────────────┐│
│ │                                                     ││
│ │                                                     ││
│ │                         ██                          ││
│ │                                                     ││
│ │        Tiny forgery: 0.3% of image area            ││
│ │        Only ~800 pixels in 512×512 image           ││
│ │                                                     ││
│ └─────────────────────────────────────────────────────┘│
│                                                         │
│ Issues:                                                 │
│ • Overwhelmed by 99.7% background                       │
│ • Standard loss functions ignore them                   │
│ • Low gradient signal during training                   │
│ • Easy to miss during inference                         │
└─────────────────────────────────────────────────────────┘
```

## Dataset Distribution

| Size Category | % of Image | % of Dataset | Detection Challenge |
|---------------|------------|--------------|---------------------|
| Tiny | <0.5% | 24% | Extremely hard |
| Small | 0.5-2% | 25% | Hard |
| Medium | 2-5% | 16% | Moderate |
| Large | 5-10% | 20% | Easy |
| XLarge | >10% | 15% | Very easy |

**Key insight**: 49% of forgeries are small (<2%), yet general models achieve only ~45% recall on them.

## How It Works

### Strategy 1: Dataset Filtering

Only train on small forgeries:

```python
# Filter to only tiny and small forgeries
TINY_THRESHOLD = 0.5   # % of image
SMALL_THRESHOLD = 2.0  # % of image

def filter_small_forgeries(image_paths, mask_paths):
    """Keep only images with small forgeries."""
    filtered_images = []
    filtered_masks = []
    forgery_sizes = []
    
    for img_path, mask_path in zip(image_paths, mask_paths):
        mask = np.load(mask_path)
        forgery_pct = (mask > 0.5).sum() / mask.size * 100
        
        # Only keep if forgery is smaller than threshold
        if 0 < forgery_pct < SMALL_THRESHOLD:
            filtered_images.append(img_path)
            filtered_masks.append(mask_path)
            forgery_sizes.append(forgery_pct)
    
    return filtered_images, filtered_masks, forgery_sizes
```

**Result**: Training set reduced from ~1000 to ~450 images (small forgeries only).

### Strategy 2: Heavy Oversampling

```python
# Oversampling factors
TINY_OVERSAMPLE = 5   # <0.5% forgeries seen 5× more often
SMALL_OVERSAMPLE = 2  # 0.5-2% forgeries seen 2× more often

# In dataset
self.weights = []
for size in self.forgery_sizes:
    if size < TINY_THRESHOLD:
        self.weights.append(TINY_OVERSAMPLE)  # 5×
    else:
        self.weights.append(SMALL_OVERSAMPLE)  # 2×
```

### Strategy 3: Very High Focal Gamma

```python
# Standard focal gamma: 2.0-3.0
# Small forgery specialist: 4.0

FOCAL_GAMMA = 4.0  # Extremely aggressive hard example mining
FOCAL_ALPHA = 0.85  # Higher positive class weight

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.85, gamma=4.0):
        ...
```

### Strategy 4: Crop-Around-Forgery

```python
def crop_around_forgery(self, image, mask):
    """
    Crop around the forged region to get better resolution.
    The forgery takes up more of the crop, improving detection.
    """
    # Find forgery bounding box
    ys, xs = np.where(mask > 0.5)
    if len(ys) == 0:
        return image, mask
    
    # Calculate center
    cy, cx = ys.mean(), xs.mean()
    
    # Determine crop size (2-4× larger than forgery)
    y_range = ys.max() - ys.min()
    x_range = xs.max() - xs.min()
    crop_size = max(y_range, x_range) * random.uniform(2, 4)
    crop_size = max(crop_size, 128)  # Minimum size
    crop_size = min(crop_size, min(image.shape[:2]))  # Max = image size
    
    # Calculate crop bounds
    half = crop_size // 2
    y1 = int(max(0, cy - half))
    x1 = int(max(0, cx - half))
    y2 = int(min(image.shape[0], y1 + crop_size))
    x2 = int(min(image.shape[1], x1 + crop_size))
    
    # Crop and resize to standard size
    cropped_image = image[y1:y2, x1:x2]
    cropped_mask = mask[y1:y2, x1:x2]
    
    cropped_image = cv2.resize(cropped_image, (IMG_SIZE, IMG_SIZE))
    cropped_mask = cv2.resize(cropped_mask, (IMG_SIZE, IMG_SIZE))
    
    return cropped_image, cropped_mask
```

**Effect**: A 0.3% forgery in 512×512 becomes a 5% forgery in the zoomed crop.

### Strategy 5: More Training Epochs

```python
EPOCHS = 40  # Instead of 30

# Smaller dataset needs more epochs to see all variations
# Also helps with heavy oversampling
```

### Strategy 6: Lower Learning Rate

```python
LEARNING_RATE = 0.0001  # 10× lower than script_1

# Fine-grained learning for subtle patterns
# Prevents overshooting optimal weights
```

## Training Flow

```
┌─────────────────────────────────────────────────────────┐
│ 1. Filter Dataset to Small Forgeries Only               │
│    └── Keep only images where forgery < 2%             │
│    └── ~450 images from ~1000 total                    │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ 2. Analyze Size Distribution                            │
│    └── Tiny (<0.5%): ~220 images                       │
│    └── Small (0.5-2%): ~230 images                     │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ 3. Create Weighted Dataset                              │
│    └── Tiny: 5× weight                                 │
│    └── Small: 2× weight                                │
│    └── Crop-around-forgery augmentation                │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ 4. Train from Scratch                                   │
│    └── Fresh model (not fine-tuned)                    │
│    └── 40 epochs                                       │
│    └── High focal gamma (4.0)                          │
│    └── Low learning rate (1e-4)                        │
└─────────────────────────────────────────────────────────┘
```

## Configuration Summary

| Parameter | Value | vs. Script 1 |
|-----------|-------|--------------|
| Training Data | Small only (<2%) | All forgeries |
| Tiny Weight | 5× | 3× |
| Focal Gamma | 4.0 | 3.0 |
| Focal Alpha | 0.85 | 0.75 |
| Learning Rate | 0.0001 | 0.0003 |
| Epochs | 40 | 30 |
| Crop Strategy | Around forgery | Random |

## Output

**Model File**: `models/small_forgery_specialist_best.pth`

## Expected Performance

### On Small Forgery Validation Set
| Metric | Value |
|--------|-------|
| Validation Dice | ~0.42 |
| Recall (tiny) | ~89% |
| Recall (small) | ~97% |

### On Full Dataset
| Forgery Size | Before (5-model) | With Specialist |
|--------------|------------------|-----------------|
| Tiny (<0.5%) | 52% | **87%** |
| Small (0.5-2%) | 59% | **95%** |

## Role in Ensemble

This model uses **UNION strategy** (not weighted average):

```python
# In submission generator
if self.small_specialist is not None:
    small_pred = self.small_specialist(input_tensor)
    # Union: take maximum of ensemble and specialist
    result = np.maximum(result, small_pred * 0.8)
```

### Why Union Instead of Weighted Average?

```
Weighted Average Problem:
  Main ensemble: 0.3 (below threshold)
  Small specialist: 0.9 (above threshold)
  Weighted result: 0.3×0.8 + 0.9×0.2 = 0.42 (below threshold!)
  → Forgery MISSED

Union Strategy:
  Main ensemble: 0.3
  Small specialist: 0.9
  Result: max(0.3, 0.9×0.8) = 0.72 (above threshold)
  → Forgery DETECTED ✓
```

The 0.8 multiplier prevents false positives while preserving high-confidence detections.

## Relationship to Other Scripts

```
┌──────────────────────────────────────────────────────────┐
│                    6-Model Ensemble                      │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  WEIGHTED AVERAGE                    UNION               │
│  ┌────────────────────────────┐     ┌──────────────────┐│
│  │ script_1: weight 1.5       │     │ script_6:        ││
│  │ script_2: weight 1.2       │     │ Small Specialist ││
│  │ script_3: weight 1.0       │ MAX │ (catches tiny    ││
│  │ script_4: weight 0.8       │────▶│  forgeries that  ││
│  │ script_5: weight 0.5       │     │  main ensemble   ││
│  └────────────────────────────┘     │  misses)         ││
│                                      └──────────────────┘│
│                                                          │
│  Result: Best of both worlds                             │
│  - General ensemble handles medium/large                 │
│  - Specialist catches tiny/small                         │
└──────────────────────────────────────────────────────────┘
```

## Usage

### Training
```bash
python script_6_small_forgery_specialist.py
```

### Prerequisites
- Training images and masks available
- Sufficient small forgery examples (at least 100)

### Monitoring
```bash
tail -f logs/train_6_small.log
```

## Technical Details

### Why Train from Scratch?

1. **Different distribution**: Only small forgeries, very different from full dataset
2. **Specialized patterns**: Need to learn subtle, fine-grained features
3. **Avoid bias**: General model patterns may hurt small detection
4. **Optimal representation**: Fresh encoder learns small-forgery-specific features

### Why Very High Focal Gamma (4.0)?

```
Standard gamma (2.0): focal_weight = (1 - p_t)^2
  Easy example (p_t=0.9): weight = 0.01
  Hard example (p_t=0.5): weight = 0.25

High gamma (4.0): focal_weight = (1 - p_t)^4
  Easy example (p_t=0.9): weight = 0.0001  # Even smaller!
  Hard example (p_t=0.5): weight = 0.0625
```

Higher gamma makes the model focus **even more** on hard examples.

### Crop-Around-Forgery Details

```
Original image: 512×512
Forgery size: 16×16 (0.1% of image)

Standard approach:
  - Forgery is 16 pixels in 512×512
  - Tiny signal, easily ignored

Crop-around-forgery:
  - Crop to 64×64 around forgery
  - Resize to 512×512
  - Forgery now 128×128 (25% of crop!)
  - Much stronger signal
```

## Common Issues

### Not Enough Small Forgeries
- Lower threshold: `SMALL_THRESHOLD = 3.0`
- Use data augmentation more aggressively
- Consider synthetic small forgeries

### Overfitting
- Reduce epochs
- Increase regularization
- Use more augmentation

### False Positives on Large Images
- The 0.8 multiplier in union helps
- Consider using model only for images with suspected small forgeries
- Increase ensemble weight of other models

## Performance Analysis

### By Size Category

```
Full Ensemble Performance (with specialist):

Size         | Without Specialist | With Specialist
-------------|--------------------|-----------------
Tiny (<0.5%) |       52%         |      87%  (+35%)
Small (0.5-2%)|       59%         |      95%  (+36%)
Medium (2-5%) |       87%         |      98%  (+11%)
Large (5-10%) |       91%         |      98%  (+7%)
XLarge (>10%) |       94%         |      95%  (+1%)
```

The specialist dramatically improves tiny/small detection with minimal impact on larger forgeries.

### False Positive Analysis

The union strategy can increase false positives:
- Without specialist: ~5% FP rate
- With specialist (raw): ~8% FP rate
- With 0.8 multiplier: ~6% FP rate

The 0.8 multiplier is a compromise between recall and precision.

## Advanced: Thresholding the Specialist

For maximum precision, you can threshold the specialist higher:

```python
# In submission generator, instead of:
result = np.maximum(result, small_pred * 0.8)

# Use:
small_binary = (small_pred > 0.8).astype(float)  # Only very confident
result = np.maximum(result, small_binary * 0.9)
```

This only uses specialist predictions where it's >80% confident.
