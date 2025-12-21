# Script 2: Train Binary Classifier

**File:** `bin/script_2_train_binary_classifier.py`

## Overview

This script trains a binary image classifier that distinguishes between forged and authentic scientific images. It serves as Stage 1 of the two-stage detection pipeline, acting as a fast filter to reject obvious non-forgeries before running the more expensive segmentation models.

## Purpose

Create a lightweight classifier that can:
- Quickly filter out authentic images (reduce processing time)
- Reduce false positives by rejecting authentic images before segmentation
- Improve overall pipeline efficiency and accuracy

## Pipeline Role

```
┌─────────────────────────────────────────────────────────────┐
│                    Two-Stage Pipeline                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input Image                                                 │
│       │                                                      │
│       ▼                                                      │
│  ┌─────────────────────┐                                    │
│  │  STAGE 1 (This)     │  ← Binary Classifier               │
│  │  Is it forged?      │     Fast, lightweight              │
│  └─────────┬───────────┘                                    │
│       │    │                                                 │
│  P<0.25    P≥0.25                                           │
│       │    │                                                 │
│       ▼    ▼                                                 │
│  "authentic"  ┌───────────────────┐                         │
│  (skip seg)   │  STAGE 2          │  ← Segmentation         │
│               │  Where is forgery?│     4-model ensemble    │
│               └───────────────────┘                         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Architecture: ForgeryClassifier

```python
class ForgeryClassifier(nn.Module):
    """
    Binary classifier: forged (1) vs authentic (0)
    """
```

### Network Structure

```
┌─────────────────────────────────────────┐
│        Input Image (384×384×3)          │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│    EfficientNet-B2 Backbone             │
│    (ImageNet pretrained, num_classes=0) │
│    Output: 1408 features                │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│           Classifier Head               │
│    Linear(1408, 256) + ReLU             │
│    Dropout(0.3)                         │
│    Linear(256, 1)                       │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│      Output: Forgery Probability        │
│      (after sigmoid)                    │
└─────────────────────────────────────────┘
```

## Configuration

```python
IMG_SIZE = 384       # Smaller than segmentation (512) for speed
BATCH_SIZE = 16      # Larger batch size possible due to smaller model
EPOCHS = 20          # Training epochs
LR = 1e-4            # Learning rate
```

## Training Data

Uses all training images split 90/10 for train/validation:

| Split | Forged | Authentic | Total |
|-------|--------|-----------|-------|
| Train | ~2,476 | ~2,139 | ~4,615 |
| Val | ~275 | ~238 | ~513 |

## Data Augmentation

### Training Augmentations
```python
A.Compose([
    A.Resize(384, 384),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
    ], p=0.3),
    A.Normalize(mean=ImageNet_mean, std=ImageNet_std),
])
```

### Validation Augmentations
```python
A.Compose([
    A.Resize(384, 384),
    A.Normalize(mean=ImageNet_mean, std=ImageNet_std),
])
```

## Loss Function

**Binary Cross-Entropy with Logits:**
```python
criterion = nn.BCEWithLogitsLoss()
```

## Training Metrics

The script tracks and reports:

| Metric | Description |
|--------|-------------|
| Train Loss | BCE loss on training set |
| Train Accuracy | Correct predictions / total |
| Val Accuracy | Validation accuracy |
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| TP/FP/TN/FN | Confusion matrix counts |

## Training Process

1. **Load Data:** Collect forged and authentic image paths
2. **Split Data:** 90% train, 10% validation (seed=42)
3. **Create Loaders:** Shuffled training, sequential validation
4. **Train Loop:** 20 epochs with cosine annealing LR
5. **Validation:** Evaluate after each epoch
6. **Save Best:** Save model with highest validation accuracy

## Usage

```bash
cd bin
python script_2_train_binary_classifier.py
```

No command-line arguments - uses default configuration.

## Output

Model saved to `models/binary_classifier_best.pth`

Checkpoint contains:
```python
{
    'epoch': int,               # Best epoch number
    'model_state_dict': dict,   # Model weights
    'val_acc': float,           # Best validation accuracy
    'precision': float,         # Precision at best epoch
    'recall': float,            # Recall at best epoch
}
```

## Expected Training Output

```
============================================================
TRAINING BINARY CLASSIFIER (Stage 1)
============================================================
Device: cuda

Forged images: 2751
Authentic images: 2377
Train: 2476 forged, 2139 authentic
Val: 275 forged, 238 authentic

Epoch 1/20 - Loss: 0.4523, Train Acc: 0.782, Val Acc: 0.856, Time: 45s
  Precision: 0.872, Recall: 0.845
  TP: 232, FP: 34, TN: 204, FN: 43
  ✓ Saved best model (Acc: 0.856)

Epoch 2/20 - Loss: 0.2891, Train Acc: 0.891, Val Acc: 0.912, Time: 44s
  Precision: 0.923, Recall: 0.905
  TP: 249, FP: 21, TN: 217, FN: 26
  ✓ Saved best model (Acc: 0.912)
...

Training complete! Best Acc: 0.945
Model saved to: models/binary_classifier_best.pth
```

## Performance Characteristics

### Speed
- **Inference time:** ~5-10ms per image (GPU)
- **Memory:** ~200MB GPU memory
- **Throughput:** ~100-200 images/second

### Accuracy (typical)
- **Validation accuracy:** 90-95%
- **Precision:** 90-95%
- **Recall:** 85-92%

## Optimal Threshold Selection

The default threshold (0.5) may not be optimal. For the two-stage pipeline:
- **Lower threshold (e.g., 0.25):** More images pass to segmentation, higher recall
- **Higher threshold (e.g., 0.75):** More filtering, lower false positives

The optimal threshold (0.25) was determined through validation sweeps.

## Dependencies

- PyTorch
- timm (for EfficientNet-B2)
- albumentations
- OpenCV (cv2)
- NumPy

## Hardware Requirements

- GPU with at least 4GB VRAM
- Training time: ~15-20 minutes on modern GPU

## Integration with Pipeline

This classifier is used by `script_3_two_stage_submission.py`:

```python
# Load classifier
classifier = ForgeryClassifier('efficientnet_b2')
classifier.load_state_dict(checkpoint['model_state_dict'])

# Classify image
prob = torch.sigmoid(classifier(image_tensor)).item()

if prob < 0.25:
    return "authentic"  # Skip segmentation
else:
    # Proceed to segmentation
    ...
```

## Next Steps

After training the binary classifier:
1. **script_3_two_stage_submission.py** - Use classifier in two-stage pipeline
2. Validate threshold selection on held-out data
