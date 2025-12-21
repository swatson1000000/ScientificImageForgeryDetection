# Script 1: Train V4 Models (Hard Negative Mining)

**File:** `bin/script_1_train_v4.py`

## Overview

This script fine-tunes the V3 models to create improved V4 models by incorporating a much larger set of hard negative authentic images. It specifically targets the false positive problem by training on all authentic images that produce false positives at standard detection thresholds.

## Purpose

Reduce false positive rates by fine-tuning pre-trained V3 models with:
- All authentic images that produce false positives at threshold=0.75, min-area=500
- Includes both dark images and normal images that cause FPs
- Uses a stronger FP penalty during training

## Key Improvements Over V3

| Aspect | V3 | V4 |
|--------|----|----|
| Hard negatives | 73 images | 677 images |
| FP penalty | 2.0 | 4.0 |
| Starting point | ImageNet | Pre-trained V3 |
| Dark image handling | Limited | 173 dark images included |

## Hard Negative Dataset

The hard negatives are loaded from `/tmp/all_fp_images.txt`:

| Category | Count | Description |
|----------|-------|-------------|
| Dark images | 173 | Mean brightness < 30 |
| Normal FP images | 504 | Standard images that produce FPs |
| **Total** | **677** | All FP-producing authentic images |

## Models Fine-Tuned

| Model Name | Base Model | Output File |
|------------|------------|-------------|
| `highres_no_ela` | `highres_no_ela_v3_best.pth` | `highres_no_ela_v4_best.pth` |
| `hard_negative` | `hard_negative_v3_best.pth` | `hard_negative_v4_best.pth` |
| `high_recall` | `high_recall_v3_best.pth` | `high_recall_v4_best.pth` |
| `enhanced_aug` | `enhanced_aug_v3_best.pth` | `enhanced_aug_v4_best.pth` |

## Architecture

Same architecture as V3 (AttentionFPN with EfficientNet-B2 encoder).

### Loss Function: CombinedLossV4

Enhanced loss function with stronger false positive penalty:

```python
class CombinedLossV4:
    focal_weight = 0.5    # Focal loss contribution
    dice_weight = 0.5     # Dice loss contribution
    fp_penalty = 4.0      # Doubled from V3's 2.0
```

**Loss calculation:**
```python
loss = 0.5 * focal_loss + 0.5 * dice_loss + 4.0 * fp_penalty_for_authentic
```

## Dataset Class: HardNegativeDatasetV4

Combines forged images with hard negative authentic images:

### Sample Weighting

| Sample Type | Weight | Purpose |
|-------------|--------|---------|
| Large forgeries (>5% area) | 1.0 | Standard weight |
| Medium forgeries (1-5%) | 2.0 | Prioritize medium |
| Small forgeries (<1%) | 3.0 | Prioritize small |
| Hard negative authentic | 5.0 | Highest priority |

### Augmentations

```python
A.Compose([
    A.RandomResizedCrop(512, scale=(0.7, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.OneOf([
        A.GaussNoise(std_range=(0.01, 0.05)),
        A.GaussianBlur(blur_limit=3),
        A.MedianBlur(blur_limit=3),
    ], p=0.3),
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
        A.HueSaturationValue(...),
        A.ColorJitter(...),
    ], p=0.5),
    A.Normalize(mean=ImageNet_mean, std=ImageNet_std),
])
```

## Configuration

```python
IMG_SIZE = 512       # Input resolution
BATCH_SIZE = 4       # Batch size
EPOCHS = 25          # Training epochs
LR = 1e-4            # Learning rate
```

## Training Process

1. **Load Hard Negatives:** Read image IDs from `/tmp/all_fp_images.txt`
2. **Load V3 Model:** Start from pre-trained V3 weights (or ImageNet if not found)
3. **Create Dataset:** Combine forged images with hard negatives
4. **Weighted Sampling:** Prioritize hard negatives with 5× weight
5. **Fine-tune:** Train for 25 epochs with cosine annealing
6. **Save Best:** Save model with lowest training loss

## Usage

### Train All V4 Models
```bash
cd bin
python script_1_train_v4.py
```

### Train Specific Models
```bash
python script_1_train_v4.py --models highres_no_ela hard_negative
python script_1_train_v4.py --models high_recall
```

## Prerequisites

Before running this script:

1. **V3 models must exist** in `models/` directory:
   - `highres_no_ela_v3_best.pth`
   - `hard_negative_v3_best.pth`
   - `high_recall_v3_best.pth`
   - `enhanced_aug_v3_best.pth`

2. **Hard negative file** must exist at `/tmp/all_fp_images.txt`
   - Format: one image ID per line
   - Can include comma-separated metadata

## Output

Models saved to `models/` directory:
- `models/highres_no_ela_v4_best.pth`
- `models/hard_negative_v4_best.pth`
- `models/high_recall_v4_best.pth`
- `models/enhanced_aug_v4_best.pth`

## Expected Training Output

```
============================================================
Hard Negative Mining V4
============================================================
Device: cuda
Models to train: ['highres_no_ela', 'hard_negative', 'high_recall', 'enhanced_aug']
Hard negative images: 677
Found 677 hard negative images
Forged images: 2751

============================================================
Training highres_no_ela v4
============================================================
  Loaded base model: highres_no_ela_v3_best.pth
Dataset: 2751 forged + 677 hard negatives
  Saved best model (epoch 1, loss=0.2345)
  ...
  Training complete. Best: epoch 18, loss=0.1234

Total training time: 3.5 hours
```

## Performance Impact

Fine-tuning with expanded hard negatives typically achieves:
- **20-30% reduction** in false positive rate
- **Minimal impact** on true positive rate
- Better handling of dark/challenging images

## Dependencies

- PyTorch
- segmentation-models-pytorch (smp)
- albumentations
- OpenCV (cv2)
- NumPy

## Next Steps

After training V4 models:
1. **script_2_train_binary_classifier.py** - Train binary classifier
2. **script_3_two_stage_submission.py** - Generate submissions with two-stage pipeline
