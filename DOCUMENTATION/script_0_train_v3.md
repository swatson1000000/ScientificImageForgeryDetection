# Script 0: Train V3 Models

**File:** `bin/script_0_train_v3.py`

## Overview

This script trains the base V3 segmentation models for scientific image forgery detection. It creates 4 specialized models using augmented forged data combined with hard negative mining to improve both detection accuracy and reduce false positives.

## Purpose

Train ensemble models that can detect manipulated regions in scientific images by learning from:
- Original forged images with ground truth masks
- Augmented forged images for better generalization
- Hard negative authentic images (images that commonly produce false positives)

## Training Data

| Dataset | Count | Description |
|---------|-------|-------------|
| Original Forged | ~2,751 | Forged images with manipulation masks |
| Augmented Forged | ~7,564 | Data-augmented versions of forged images |
| Authentic | ~12,200 | Non-manipulated images (balanced sampling) |
| Hard Negatives | 73 | Authentic images from test set that produce FPs |

**Total training samples:** ~22,500 images

## Models Trained

The script trains 4 models with identical architecture but potentially different training emphasis:

| Model Name | Output File | Purpose |
|------------|-------------|---------|
| `highres_no_ela` | `highres_no_ela_v3_best.pth` | High-resolution detection without ELA preprocessing |
| `hard_negative` | `hard_negative_v3_best.pth` | Emphasizes hard negative mining |
| `high_recall` | `high_recall_v3_best.pth` | Optimized for high forgery recall |
| `enhanced_aug` | `enhanced_aug_v3_best.pth` | Enhanced augmentation training |

## Architecture

### Base Model: AttentionFPN

```
┌─────────────────────────────────────────┐
│           Input Image (512×512×3)       │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│  FPN (Feature Pyramid Network)          │
│  Encoder: EfficientNet-B2 (ImageNet)    │
│  Decoder: FPN with multi-scale features │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│         Attention Gate                   │
│  - Conv 1×1 → hidden1                   │
│  - BatchNorm + ReLU                      │
│  - Conv 3×3 → hidden2                   │
│  - BatchNorm + ReLU                      │
│  - Conv 1×1 → 1 channel                 │
│  - Sigmoid activation                    │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│      Output Mask (512×512×1)            │
└─────────────────────────────────────────┘
```

### Loss Function: CombinedLoss

A custom loss function combining three components:

1. **Focal Loss** (weight: 0.5)
   - α = 0.25, γ = 2.0
   - Addresses class imbalance by down-weighting easy examples

2. **Dice Loss** (weight: 0.5)
   - Optimizes overlap between predicted and ground truth masks
   - Handles small forgery regions well

3. **False Positive Penalty** (weight: 2.0)
   - Extra penalty for predicting forgery on authentic images
   - Reduces false positive rate

```python
loss = 0.5 * focal_loss + 0.5 * dice_loss + 2.0 * fp_penalty
```

## Data Augmentation

Training augmentations applied to both images and masks:

| Augmentation | Parameters | Probability |
|--------------|------------|-------------|
| RandomResizedCrop | scale=(0.7, 1.0) | Always |
| HorizontalFlip | - | 50% |
| VerticalFlip | - | 50% |
| RandomRotate90 | - | 50% |
| ColorJitter | brightness=0.2, contrast=0.2 | 50% (OneOf) |
| GaussianBlur | blur_limit=(3, 5) | 20% |
| GaussNoise | std_range=(0.01, 0.03) | 20% |
| Normalize | ImageNet mean/std | Always |

## Weighted Sampling

Samples are weighted to prioritize difficult cases:

| Sample Type | Weight | Rationale |
|-------------|--------|-----------|
| Small forgeries (<1% area) | 3.0 | Hardest to detect |
| Medium forgeries (1-5% area) | 2.0 | Moderately difficult |
| Large forgeries (>5% area) | 1.0 | Standard weight |
| Authentic images | 1.0 | Standard weight |
| Hard negatives | 5.0 | Highest priority - reduce FPs |

## Configuration

```python
IMG_SIZE = 512       # Input resolution
BATCH_SIZE = 4       # Batch size per GPU
EPOCHS = 25          # Training epochs
LR = 1e-4            # Learning rate
```

## Training Process

1. **Data Collection:** Gathers forged, augmented, authentic, and hard negative images
2. **Dataset Creation:** Creates weighted datasets for each image type
3. **Weighted Sampling:** Uses WeightedRandomSampler to balance training
4. **Model Training:** 25 epochs with cosine annealing learning rate
5. **Model Saving:** Saves best model checkpoint based on training loss

## Usage

### Train All Models
```bash
cd bin
python script_0_train_v3.py
```

### Train Specific Model
```bash
python script_0_train_v3.py --model highres_no_ela
python script_0_train_v3.py --model hard_negative
python script_0_train_v3.py --model high_recall
python script_0_train_v3.py --model enhanced_aug
```

## Output

Models are saved to `models/` directory:
- `models/highres_no_ela_v3_best.pth`
- `models/hard_negative_v3_best.pth`
- `models/high_recall_v3_best.pth`
- `models/enhanced_aug_v3_best.pth`

Each checkpoint contains:
```python
{
    'epoch': int,               # Best epoch number
    'model_state_dict': dict,   # Model weights
    'optimizer_state_dict': dict,  # Optimizer state
    'loss': float,              # Best loss value
}
```

## Dependencies

- PyTorch
- segmentation-models-pytorch (smp)
- albumentations
- OpenCV (cv2)
- NumPy

## Hardware Requirements

- GPU with at least 8GB VRAM recommended
- Training time: ~2-3 hours per model on modern GPU

## Next Steps

After training V3 models, proceed to:
1. **script_1_train_v4.py** - Fine-tune V3 models with expanded hard negatives
2. **script_2_train_binary_classifier.py** - Train binary classifier for two-stage pipeline
