# script_3_two_stage_submission.py - Detailed Reference

## Overview

This script generates the final submission CSV using the **two-stage pipeline**:
1. **Stage 1**: Binary classifier filters likely authentic images
2. **Stage 2**: 4-model V4 ensemble generates forgery masks

## Purpose

The two-stage approach optimizes the speed/accuracy trade-off:
- Fast classifier rejects ~30% of images immediately
- Slower but accurate ensemble only runs on suspicious images
- Results in better net score (TP - FP) than single-stage approaches

## Architecture

### Two-Stage Pipeline Flow

```
Input Image
    │
    ▼
┌─────────────────────────────────────────┐
│  Stage 1: Binary Classifier             │
│  - EfficientNet-B2 backbone             │
│  - Input: 384×384                       │
│  - Output: probability of being forged  │
│  - Speed: ~10ms                         │
└─────────────────────────────────────────┘
    │
    ├── prob < 0.25 ──► Return "authentic" (filtered)
    │
    ▼ prob ≥ 0.25
┌─────────────────────────────────────────┐
│  Stage 2: Segmentation Ensemble         │
│  - 4× AttentionFPN models               │
│  - Input: 512×512                       │
│  - 4× TTA (flips)                       │
│  - Mean aggregation                     │
│  - Speed: ~200ms                        │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Post-processing                        │
│  - Adaptive thresholding                │
│  - Small region removal (min_area)      │
│  - Resize to original resolution        │
└─────────────────────────────────────────┘
    │
    ├── mask empty ──► Return "authentic"
    │
    ▼ mask has content
    Return RLE-encoded mask
```

## Configuration

### Default Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CLASSIFIER_THRESHOLD` | 0.25 | Probability threshold for classifier |
| `SEG_THRESHOLD` | 0.35 | Base segmentation threshold |
| `MIN_AREA` | 300 | Minimum region size in pixels |
| `CLASSIFIER_SIZE` | 384 | Classifier input resolution |
| `SEG_SIZE` | 512 | Segmentation input resolution |

### V4 Ensemble Models

```python
V4_MODELS = [
    'highres_no_ela_v4_best.pth',
    'hard_negative_v4_best.pth',
    'high_recall_v4_best.pth',
    'enhanced_aug_v4_best.pth',
]
```

## Classes and Functions

### TwoStageGenerator

Main class orchestrating the pipeline.

```python
class TwoStageGenerator:
    """Two-stage pipeline: classifier + 4-model ensemble."""
    
    def __init__(self, model_dir=None, classifier_threshold=0.25):
        """
        Initialize pipeline.
        
        Args:
            model_dir: Path to directory containing model files
            classifier_threshold: Probability threshold for Stage 1
        """
        # Load classifier
        self.classifier = load_classifier(model_dir, DEVICE)
        
        # Load 4 ensemble models
        self.seg_models = load_ensemble(model_dir, DEVICE)
    
    def classify(self, image_bgr) -> float:
        """
        Run binary classifier.
        
        Returns: Probability of image being forged (0.0 - 1.0)
        """
        ...
    
    def segment(self, image_bgr, seg_threshold, min_area, use_tta, adaptive):
        """
        Run segmentation ensemble.
        
        Returns: (mask, is_forged)
        """
        ...
    
    def generate_mask(self, image_path, seg_threshold, min_area, use_tta, adaptive):
        """
        Full two-stage pipeline.
        
        Returns: (mask, is_forged, was_filtered)
        """
        ...
```

### Preprocessing Functions

```python
def preprocess_for_classifier(image_bgr, device):
    """
    Preprocess for classifier (384×384).
    
    Steps:
    1. Resize to 384×384
    2. BGR → RGB
    3. Normalize to [0,1]
    4. ImageNet normalization
    5. Convert to tensor
    """
    img_resized = cv2.resize(image_bgr, (384, 384))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_norm = (img_norm - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    return tensor


def preprocess_for_segmentation(image_bgr, device):
    """Same as above but 512×512."""
    ...
```

### Test-Time Augmentation

```python
def apply_tta(model, img_tensor):
    """
    Apply 4× TTA with flip augmentations.
    
    Augmentations:
    1. Original
    2. Horizontal flip
    3. Vertical flip
    4. Both flips
    
    Returns: MAX aggregation of all predictions
    """
    preds = []
    
    # Original
    preds.append(torch.sigmoid(model(img_tensor)))
    
    # Horizontal flip
    pred = torch.sigmoid(model(torch.flip(img_tensor, dims=[3])))
    preds.append(torch.flip(pred, dims=[3]))  # flip back
    
    # Vertical flip
    pred = torch.sigmoid(model(torch.flip(img_tensor, dims=[2])))
    preds.append(torch.flip(pred, dims=[2]))
    
    # Both flips
    pred = torch.sigmoid(model(torch.flip(img_tensor, dims=[2, 3])))
    preds.append(torch.flip(pred, dims=[2, 3]))
    
    return torch.stack(preds).max(dim=0)[0]
```

### Adaptive Thresholding

```python
def get_adaptive_threshold(image_bgr, base_threshold=0.35):
    """
    Adjust threshold based on image brightness.
    
    Dark images: Lower threshold (easier to detect)
    Bright images: Higher threshold (stricter)
    """
    brightness = np.mean(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY))
    
    if brightness < 50:      # Very dark
        return max(0.20, base_threshold - 0.10)
    elif brightness < 80:    # Dark
        return max(0.25, base_threshold - 0.05)
    elif brightness > 200:   # Very bright
        return min(0.50, base_threshold + 0.05)
    else:
        return base_threshold
```

### RLE Encoding

```python
def mask_to_rle(mask):
    """
    Convert binary mask to Run-Length Encoding.
    
    Format: "count1 value1 count2 value2 ..."
    
    Example:
    mask = [0,0,0,1,1,0,0]
    RLE  = "3 0 2 1 2 0"
    """
    flat = mask.flatten()
    runs = []
    current_val = flat[0]
    count = 1
    
    for i in range(1, len(flat)):
        if flat[i] == current_val:
            count += 1
        else:
            runs.append(f"{count} {int(current_val)}")
            current_val = flat[i]
            count = 1
    runs.append(f"{count} {int(current_val)}")
    
    return " ".join(runs)
```

## Command Line Interface

### Basic Usage

```bash
python script_3_two_stage_submission.py --input test_images/ --output submission.csv
```

### All Options

```bash
python script_3_two_stage_submission.py \
    --input test_images/ \            # Input directory (required)
    --output submission.csv \          # Output CSV file
    --classifier-threshold 0.25 \      # Stage 1 threshold
    --seg-threshold 0.35 \             # Segmentation threshold
    --min-area 300 \                   # Minimum region size
    --no-tta \                         # Disable TTA
    --no-adaptive \                    # Disable adaptive threshold
    --model-dir /path/to/models/       # Custom model directory
```

### Option Details

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--input, -i` | str | (required) | Directory containing test images |
| `--output, -o` | str | `submission.csv` | Output CSV file path |
| `--classifier-threshold, -c` | float | 0.25 | Classifier probability threshold |
| `--seg-threshold, -s` | float | 0.35 | Base segmentation threshold |
| `--min-area` | int | 300 | Minimum forgery area in pixels |
| `--no-tta` | flag | False | Disable test-time augmentation |
| `--no-adaptive` | flag | False | Disable adaptive thresholding |
| `--model-dir` | str | `models/` | Path to model files |

## Output Format

### Submission CSV

```csv
case_id,annotation
10001,authentic
10002,100 0 50 1 200 0 30 1
10003,authentic
...
```

- `case_id`: Image filename without extension
- `annotation`: Either `"authentic"` or RLE-encoded mask

### Console Output

```
Found 934 images in test_images/
Classifier threshold: 0.25
Seg threshold: 0.35, Min area: 300
TTA: True, Adaptive: True

Loading two-stage pipeline...
  Classifier threshold: 0.25
  ✓ Binary classifier loaded
  Loading 4-model ensemble:
    ✓ highres_no_ela_v4_best.pth
    ✓ hard_negative_v4_best.pth
    ✓ high_recall_v4_best.pth
    ✓ enhanced_aug_v4_best.pth
  ✓ Loaded 4 ensemble models

Processing: 100%|██████████| 934/934 [03:22<00:00, 4.62it/s]

============================================================
TWO-STAGE SUBMISSION GENERATED
============================================================
Total images: 934
Filtered by classifier: 285 (30.5%)
Passed to segmentation: 649
Forged detected: 450 (48.2%)
Authentic: 484
Saved to: submission.csv
```

## Processing Details

### Per-Image Processing

```python
def process_one_image(image_path, generator, seg_threshold, min_area, use_tta, adaptive):
    # 1. Load image
    image = cv2.imread(str(image_path))
    
    # 2. Stage 1: Classification
    prob = generator.classify(image)
    
    if prob < generator.classifier_threshold:
        # Filtered by classifier
        return 'authentic', True  # (annotation, was_filtered)
    
    # 3. Stage 2: Segmentation
    mask, is_forged = generator.segment(
        image, seg_threshold, min_area, use_tta, adaptive
    )
    
    if not is_forged:
        return 'authentic', False
    
    # 4. Encode mask
    return mask_to_rle(mask), False
```

### Ensemble Aggregation

```python
# For each image that passes classifier:

# 1. Run each model with TTA
all_preds = []
for model in seg_models:
    pred = apply_tta(model, img_tensor)  # 4× TTA, max aggregation
    all_preds.append(pred)

# 2. Mean across ensemble
avg_pred = torch.stack(all_preds).mean(dim=0)

# 3. Apply threshold
binary = (avg_pred > threshold).astype(np.uint8)

# 4. Post-process
# - Remove small regions
# - Resize to original resolution
```

### Post-Processing Pipeline

```python
# 1. Adaptive threshold
if adaptive:
    threshold = get_adaptive_threshold(image, base_threshold)

# 2. Binary thresholding
binary = (pred > threshold).astype(np.uint8)

# 3. Remove small connected components
if min_area > 0:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            binary[labels == i] = 0

# 4. Resize to original resolution
mask = cv2.resize(binary, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
```

## Input Requirements

### Test Images

- Location: Specified via `--input` argument
- Formats: PNG, JPG, JPEG, TIF, TIFF
- Size: Any (resized internally)

### Model Files

Required in `models/` directory (or `--model-dir`):

```
models/
├── binary_classifier_best.pth     # Stage 1
├── highres_no_ela_v4_best.pth     # Stage 2
├── hard_negative_v4_best.pth      # Stage 2
├── high_recall_v4_best.pth        # Stage 2
└── enhanced_aug_v4_best.pth       # Stage 2
```

## Performance

### Speed

| Component | Time per Image | Notes |
|-----------|----------------|-------|
| Classifier | ~10ms | Always runs |
| Segmentation | ~200ms | 4 models × 4 TTA |
| Total (filtered) | ~10ms | ~30% of images |
| Total (not filtered) | ~210ms | ~70% of images |

For 934 images:
- With TTA: ~3-4 minutes
- Without TTA: ~1-2 minutes

### Memory Usage

- GPU: ~4-6 GB (all 5 models loaded)
- CPU: ~8-10 GB

### Accuracy (Validated)

| Metric | Value |
|--------|-------|
| Net Score (TP - FP) | 2029 |
| True Positives | 2162 |
| False Positives | 133 |
| Recall | 78.6% |
| FP Rate | 5.6% |

## Threshold Tuning

### Classifier Threshold

| Value | Effect | Use Case |
|-------|--------|----------|
| 0.15 | Very high recall, minimal filtering | When missing forgeries is costly |
| 0.25 | **Optimal** - balanced filtering | Production default |
| 0.35 | More aggressive filtering | When FP reduction is priority |
| 0.50 | Aggressive filtering | Speed optimization |

### Segmentation Threshold

| Value | Effect | Use Case |
|-------|--------|----------|
| 0.25 | Larger masks, more FP | High recall needed |
| 0.35 | **Optimal** - balanced | Production default |
| 0.45 | Smaller, precise masks | High precision needed |
| 0.55 | Very conservative | Minimal FP |

### Min Area

| Value | Effect | Use Case |
|-------|--------|----------|
| 100 | Keep small detections | Detect small forgeries |
| 300 | **Optimal** - remove noise | Production default |
| 500 | Aggressive filtering | Very clean masks |
| 1000 | Very aggressive | Only large forgeries |

## Troubleshooting

### "Model not found"

```
Error: Model not found: models/binary_classifier_best.pth
```

Solution: Ensure all 5 model files exist in `models/` directory.

### "Could not read image"

```
Error: Could not read image: test_images/broken.png
```

Solution: Check image file is valid and not corrupted.

### CUDA Out of Memory

All 5 models are loaded simultaneously. Solutions:
1. Use smaller batch (already 1)
2. Reduce ensemble to 2 models
3. Use CPU: set `DEVICE = torch.device('cpu')`

### Slow Processing

1. Disable TTA: `--no-tta` (2-3× speedup)
2. Reduce ensemble models (edit V4_MODELS list)
3. Use GPU if not already

## Integration Example

```python
from script_3_two_stage_submission import TwoStageGenerator

# Initialize
generator = TwoStageGenerator(
    model_dir='models/',
    classifier_threshold=0.25
)

# Process single image
image_path = 'test_images/sample.png'
mask, is_forged, was_filtered = generator.generate_mask(
    image_path,
    seg_threshold=0.35,
    min_area=300,
    use_tta=True,
    adaptive=True
)

if was_filtered:
    print("Filtered by classifier (authentic)")
elif not is_forged:
    print("Segmentation found no forgery (authentic)")
else:
    print(f"Forgery detected! Mask has {mask.sum()} positive pixels")
```
