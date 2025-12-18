# script_1_train_v4.py - Detailed Reference

## Overview

This script trains four V4 segmentation model variants using **hard negative mining** to reduce false positives. It fine-tunes pre-trained V3 models with a dataset augmented with authentic images that commonly produce false positives.

## Purpose

The V4 training specifically addresses the false positive problem:
- V3 models had good recall but too many FPs on authentic images
- V4 adds 677 hard negative samples (authentic images that produce FPs)
- Training with strong FP penalty reduces false positive rate significantly

## Architecture

### AttentionFPN Model

```python
class AttentionFPN(nn.Module):
    """FPN with learned attention gating."""
    
    Components:
    1. base: smp.FPN(encoder="timm-efficientnet-b2", classes=1)
    2. attention: AttentionGate(in_channels=1)
    
    Forward:
    x → FPN → AttentionGate → output
```

### AttentionGate

```python
class AttentionGate(nn.Module):
    """Learns spatial attention to refine predictions."""
    
    Architecture:
    - Conv 1×1 (in→hidden1)
    - BatchNorm + ReLU
    - Conv 3×3 (hidden1→hidden2)
    - BatchNorm + ReLU  
    - Conv 1×1 (hidden2→1)
    - Sigmoid
    
    Output: x * attention_map (element-wise multiply)
```

### Loss Function

```python
class CombinedLossV4(nn.Module):
    """Combined loss with strong FP penalty."""
    
    Total Loss = 0.5 * FocalLoss + 0.5 * DiceLoss + 4.0 * FP_Penalty
    
    - FocalLoss: α=0.25, γ=2.0 (handles class imbalance)
    - DiceLoss: 1 - (2*intersection + 1) / (sum_pred + sum_target + 1)
    - FP_Penalty: mean(sigmoid(pred)) for authentic images × 4.0
```

## Configuration

### Hyperparameters

| Parameter | Value | Location | Description |
|-----------|-------|----------|-------------|
| `IMG_SIZE` | 512 | Line 35 | Input image resolution |
| `BATCH_SIZE` | 4 | Line 36 | Samples per batch |
| `EPOCHS` | 25 | Line 37 | Training epochs per model |
| `LR` | 1e-4 | Line 38 | Learning rate |
| `hard_neg_weight` | 5.0 | Line 162 | Weight for hard negative samples |
| `fp_penalty` | 4.0 | Line 133 | FP loss multiplier |

### Model Configurations

```python
MODELS_CONFIG = {
    'highres_no_ela': {
        'base_model': 'highres_no_ela_v3_best.pth',
        'output': 'highres_no_ela_v4_best.pth',
        'in_channels': 3,
    },
    'hard_negative': {...},
    'high_recall': {...},
    'enhanced_aug': {...},
}
```

## Data Pipeline

### HardNegativeDatasetV4

```python
class HardNegativeDatasetV4(Dataset):
    """Dataset combining forged images with hard negatives."""
    
    Samples:
    - Forged images with masks (weighted by forgery size)
    - Hard negative authentic images (weight=5.0)
    
    Weighting by forgery size:
    - < 1% of image: weight 3.0
    - 1-5% of image: weight 2.0
    - > 5% of image: weight 1.0
```

### Augmentations

```python
A.Compose([
    A.RandomResizedCrop(size=(512, 512), scale=(0.7, 1.0)),
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
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ], p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])
```

### Weighted Random Sampling

```python
sampler = WeightedRandomSampler(
    dataset.weights,      # Per-sample weights
    num_samples=len(dataset),
    replacement=True      # Allow re-sampling
)
```

## Training Process

### Per-Model Training Loop

```python
for model_name in args.models:
    # 1. Load pre-trained V3 model
    model = load_model(base_model_path)
    
    # 2. Create weighted dataset
    dataset = HardNegativeDatasetV4(forged, masks, hard_negatives)
    loader = DataLoader(dataset, sampler=WeightedRandomSampler(...))
    
    # 3. Setup optimizer with cosine annealing
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # 4. Train for 25 epochs
    for epoch in range(EPOCHS):
        for images, masks in loader:
            loss = criterion(model(images), masks)
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        # Save if best loss
        if avg_loss < best_loss:
            torch.save(model, output_path)
```

## Usage Examples

### Train All Models

```bash
python bin_4/script_1_train_v4.py
```

### Train Specific Models

```bash
# Train only two models
python bin_4/script_1_train_v4.py --models highres_no_ela hard_negative

# Train single model
python bin_4/script_1_train_v4.py --models high_recall
```

## Input Requirements

### Pre-trained Models (Required)

```
models/
├── highres_no_ela_v3_best.pth
├── hard_negative_v3_best.pth
├── high_recall_v3_best.pth
└── enhanced_aug_v3_best.pth
```

### Hard Negative List

File: `/tmp/all_fp_images.txt`

Format:
```
image_id1
image_id2
...
```

Each line is an image ID (without extension) from `train_images/authentic/` that produces false positives.

### Training Data

```
train_images/
├── forged/           # PNG images
│   ├── 10001.png
│   └── ...
└── authentic/        # PNG images
    ├── 20001.png
    └── ...

train_masks/          # NumPy arrays
├── 10001.npy
└── ...
```

## Output

### Model Checkpoints

```
models/
├── highres_no_ela_v4_best.pth
├── hard_negative_v4_best.pth
├── high_recall_v4_best.pth
└── enhanced_aug_v4_best.pth
```

### Checkpoint Contents

```python
{
    'epoch': int,                    # Best epoch number
    'model_state_dict': OrderedDict, # Model weights
    'optimizer_state_dict': dict,    # Optimizer state
    'loss': float,                   # Best loss value
}
```

## Performance Notes

### Training Time

- ~1-1.5 hours per model on GPU
- ~4-6 hours total for all 4 models

### Memory Usage

- ~6-8 GB GPU memory at batch_size=4
- Reduce to batch_size=2 if OOM

### Key Metrics

The training saves model with lowest average loss per epoch. Loss components:
- Focal loss: Balanced class prediction
- Dice loss: Mask overlap quality
- FP penalty: Authentic image false positive rate

## Troubleshooting

### "Base model not found"

Ensure V3 models exist in `models/` directory. You need to train V3 first, or obtain pre-trained V3 weights.

### "Hard negative file not found"

Create the file by running validation sweep to identify FP-producing authentic images:

```bash
python bin_4/validate_v4_sweep.py --save-fp
```

Or create empty file for initial training:
```bash
touch /tmp/all_fp_images.txt
```

### CUDA Out of Memory

Reduce batch size:
```python
BATCH_SIZE = 2  # Instead of 4
```

Or reduce image size:
```python
IMG_SIZE = 384  # Instead of 512
```
