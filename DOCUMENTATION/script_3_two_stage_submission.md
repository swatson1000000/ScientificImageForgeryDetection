# Script 3: Two-Stage Submission Generator

**File:** `bin/script_3_two_stage_submission.py`

## Overview

This script generates Kaggle competition submissions using a sophisticated two-stage detection pipeline. Stage 1 uses a binary classifier to filter likely authentic images, and Stage 2 uses a 4-model ensemble to generate precise forgery masks for the remaining images.

## Purpose

Create optimized submission files that:
- Maximize true positive detections (correctly identify forgeries)
- Minimize false positives (incorrectly flagging authentic images)
- Balance speed and accuracy through intelligent filtering

## Two-Stage Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INPUT IMAGE                                   │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    STAGE 1: BINARY CLASSIFIER                        │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  ForgeryClassifier (EfficientNet-B2)                        │    │
│  │  Input: 384×384 RGB image                                   │    │
│  │  Output: P(forged) probability                              │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  if P(forged) < 0.25:  ──────────────────────┐                      │
│      return "authentic"                       │                      │
│  else:                                        │                      │
│      proceed to Stage 2                       │                      │
└───────────────────────┬───────────────────────┼─────────────────────┘
                        │                       │
                        ▼                       ▼
┌─────────────────────────────────────────┐  ┌──────────────────────┐
│        STAGE 2: 4-MODEL ENSEMBLE        │  │    FAST PATH         │
│  ┌───────────────────────────────────┐  │  │    (Filtered)        │
│  │  V4 Models (×4):                  │  │  │                      │
│  │  • highres_no_ela_v4              │  │  │  annotation:         │
│  │  • hard_negative_v4               │  │  │  "authentic"         │
│  │  • high_recall_v4                 │  │  │                      │
│  │  • enhanced_aug_v4                │  │  └──────────────────────┘
│  └───────────────────────────────────┘  │
│                    │                     │
│                    ▼                     │
│  ┌───────────────────────────────────┐  │
│  │  4× TTA per model:                │  │
│  │  • Original                       │  │
│  │  • Horizontal flip                │  │
│  │  • Vertical flip                  │  │
│  │  • Both flips                     │  │
│  │  Aggregation: MAX                 │  │
│  └───────────────────────────────────┘  │
│                    │                     │
│                    ▼                     │
│  ┌───────────────────────────────────┐  │
│  │  Ensemble: MEAN of 4 models       │  │
│  │  Threshold: Adaptive (0.35 base)  │  │
│  │  Post-process: Remove small blobs │  │
│  └───────────────────────────────────┘  │
└─────────────────────┬───────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        OUTPUT                                        │
│  • Forgery detected: RLE-encoded mask                               │
│  • No forgery: "authentic"                                          │
└─────────────────────────────────────────────────────────────────────┘
```

## Optimal Configuration

Based on validation experiments:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Classifier threshold | 0.25 | Low to maintain high recall |
| Segmentation threshold | 0.35 | Base threshold (adaptive) |
| Minimum area | 300 | Remove small false positives |
| TTA | Enabled | 4× augmentation per model |
| Adaptive threshold | Enabled | Adjusts based on image brightness |

### Performance Comparison (Validation Set)

| Configuration | Net Score | TP | FP |
|---------------|-----------|----|----|
| 4-model V4 ensemble | 2033 | 2173 | 140 |
| Single V4 model | 1934 | 2079 | 145 |
| **Ensemble improvement** | **+99** | - | - |

## Key Components

### ForgeryClassifier
```python
class ForgeryClassifier(nn.Module):
    """Binary classifier for Stage 1 filtering"""
    backbone: EfficientNet-B2 (pretrained=False, loaded from checkpoint)
    classifier: Linear(1408→256) → ReLU → Dropout(0.3) → Linear(256→1)
```

### AttentionFPN (×4)
```python
class AttentionFPN(nn.Module):
    """Segmentation model for Stage 2"""
    base: FPN with EfficientNet-B2 encoder
    attention: AttentionGate for refined predictions
```

### Test-Time Augmentation (TTA)

Each model processes 4 variations with MAX aggregation:

| Augmentation | Transform | Inverse |
|--------------|-----------|---------|
| Original | None | None |
| H-flip | flip(dims=[3]) | flip(dims=[3]) |
| V-flip | flip(dims=[2]) | flip(dims=[2]) |
| Both | flip(dims=[2,3]) | flip(dims=[2,3]) |

```python
final_pred = MAX(pred_original, pred_hflip, pred_vflip, pred_both)
```

### Adaptive Thresholding

Threshold adjusts based on image brightness:

| Brightness | Threshold | Rationale |
|------------|-----------|-----------|
| < 50 (very dark) | 0.25 (-0.10) | Dark images need lower threshold |
| 50-80 (dark) | 0.30 (-0.05) | Slightly lower threshold |
| 80-200 (normal) | 0.35 (base) | Standard threshold |
| > 200 (bright) | 0.40 (+0.05) | Bright images can use higher |

### RLE Encoding

Masks are encoded as Run-Length Encoding (RLE) strings:

```python
def mask_to_rle(mask):
    """Convert binary mask to 'count value count value ...' format"""
    # Example: "100 0 50 1 100 0" means:
    # 100 pixels of 0, then 50 pixels of 1, then 100 pixels of 0
```

## Usage

### Basic Usage
```bash
cd bin
python script_3_two_stage_submission.py \
    --input /path/to/test_images/ \
    --output submission.csv
```

### Full Options
```bash
python script_3_two_stage_submission.py \
    --input test_images/ \
    --output submission.csv \
    --classifier-threshold 0.25 \
    --seg-threshold 0.35 \
    --min-area 300 \
    --model-dir ../models
```

### Disable TTA (Faster)
```bash
python script_3_two_stage_submission.py \
    --input test_images/ \
    --output submission.csv \
    --no-tta
```

### Disable Adaptive Threshold
```bash
python script_3_two_stage_submission.py \
    --input test_images/ \
    --output submission.csv \
    --no-adaptive
```

## Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input, -i` | Required | Input directory with test images |
| `--output, -o` | `submission.csv` | Output CSV file path |
| `--classifier-threshold, -c` | 0.25 | Classifier threshold for filtering |
| `--seg-threshold, -s` | 0.35 | Base segmentation threshold |
| `--min-area` | 300 | Minimum forgery area in pixels |
| `--no-tta` | False | Disable test-time augmentation |
| `--no-adaptive` | False | Disable adaptive thresholding |
| `--model-dir` | `../models` | Directory containing model files |

## Required Model Files

The script requires these files in the models directory:

```
models/
├── binary_classifier_best.pth      # Stage 1 classifier
├── highres_no_ela_v4_best.pth      # Ensemble model 1
├── hard_negative_v4_best.pth       # Ensemble model 2
├── high_recall_v4_best.pth         # Ensemble model 3
└── enhanced_aug_v4_best.pth        # Ensemble model 4
```

## Output Format

CSV file with two columns:

| Column | Description |
|--------|-------------|
| `case_id` | Image filename (without extension) |
| `annotation` | "authentic" or RLE-encoded mask |

Example:
```csv
case_id,annotation
12345,authentic
67890,1000 0 500 1 200 0 300 1 ...
```

## Expected Console Output

```
Found 1000 images in test_images/
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

============================================================
TWO-STAGE SUBMISSION GENERATED
============================================================
Total images: 1000
Filtered by classifier: 423 (42.3%)
Passed to segmentation: 577
Forged detected: 312 (31.2%)
Authentic: 688
Saved to: submission.csv
```

## Python API

The script can also be used programmatically:

```python
from script_3_two_stage_submission import TwoStageGenerator

# Initialize generator
generator = TwoStageGenerator(
    model_dir='../models',
    classifier_threshold=0.25
)

# Process single image
mask, is_forged, was_filtered = generator.generate_mask(
    image_path='test.png',
    seg_threshold=0.35,
    min_area=300,
    use_tta=True,
    adaptive=True
)

# Generate full submission
from script_3_two_stage_submission import generate_submission

df = generate_submission(
    input_dir='test_images/',
    output_path='submission.csv',
    classifier_threshold=0.25
)
```

## Processing Time

| Stage | Time per Image | Notes |
|-------|----------------|-------|
| Stage 1 (Classifier) | ~10ms | Fast filtering |
| Stage 2 (Ensemble + TTA) | ~200ms | 4 models × 4 TTA |
| **Total (filtered)** | ~10ms | If classifier rejects |
| **Total (full pipeline)** | ~210ms | If passes to segmentation |

For 1000 images with 40% filtering: ~140 seconds total

## Dependencies

- PyTorch
- timm
- segmentation-models-pytorch
- OpenCV
- NumPy
- pandas

## Troubleshooting

### Missing Model Files
```
Error: Model file not found: models/binary_classifier_best.pth
```
**Solution:** Run `script_2_train_binary_classifier.py` first

### GPU Memory Issues
```
RuntimeError: CUDA out of memory
```
**Solutions:**
- Disable TTA with `--no-tta`
- Process images in smaller batches
- Use CPU with `CUDA_VISIBLE_DEVICES=""`

### Low Detection Rate
- Try lower classifier threshold (e.g., 0.15)
- Try lower segmentation threshold (e.g., 0.25)
- Disable adaptive thresholding

### High False Positive Rate
- Try higher classifier threshold (e.g., 0.35)
- Increase minimum area (e.g., 500)
- Try higher segmentation threshold (e.g., 0.45)
