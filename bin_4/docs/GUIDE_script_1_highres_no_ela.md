# Script 1: High Resolution Training (No ELA)

## Overview

`script_1_highres_no_ela.py` trains the **base model** of the ensemble using high-resolution RGB images. This model forms the foundation of the detection pipeline by learning to identify forged regions from pure visual patterns.

## Purpose

The primary goal is to establish a strong baseline detector that:
- Works with standard 3-channel RGB images
- Operates at high resolution (512×512) to capture fine details
- Uses focal loss to handle class imbalance (small forgeries)
- Applies attention mechanisms to focus on suspicious regions

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Input Image                          │
│                    512×512×3 (RGB)                      │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              EfficientNet-B2 Encoder                    │
│              (ImageNet pretrained)                      │
│              Features: 16×16 → 128×128                  │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              FPN (Feature Pyramid Network)              │
│              Multi-scale feature fusion                 │
│              256 decoder channels                       │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              Attention Gate                             │
│              Learned spatial attention                  │
│              Amplifies suspicious regions               │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              Output Mask                                │
│              512×512×1 (probability)                    │
└─────────────────────────────────────────────────────────┘
```

## Key Components

### 1. Attention Gate

```python
class AttentionGate(nn.Module):
    """Learns to focus on suspicious regions."""
    def __init__(self, in_channels):
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden1, kernel_size=1),  # Reduce channels
            nn.BatchNorm2d(hidden1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden1, hidden2, kernel_size=3, padding=1),  # Spatial processing
            nn.BatchNorm2d(hidden2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden2, 1, kernel_size=1),  # Single attention map
            nn.Sigmoid()  # 0-1 attention weights
        )
    
    def forward(self, x):
        attention = self.conv(x)
        return x * attention  # Element-wise multiplication
```

The attention gate learns to identify which spatial locations are most relevant for forgery detection. Forged regions often have distinctive patterns (compression artifacts, edge discontinuities) that the gate learns to highlight.

### 2. Focal Loss

```python
class FocalLoss(nn.Module):
    """Handles severe class imbalance in forgery detection."""
    def __init__(self, alpha=0.75, gamma=3.0):
        ...
    
    def forward(self, inputs, targets):
        p = torch.sigmoid(inputs)
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma  # Down-weight easy examples
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * focal_weight * bce
        return focal_loss.mean()
```

**Why Focal Loss?**
- Most pixels are non-forged (easy negatives)
- Standard BCE gives too much weight to easy examples
- γ=3.0 aggressively down-weights easy examples
- α=0.75 gives more weight to forged pixels

### 3. Small Forgery Oversampling

```python
# Configuration
SMALL_FORGERY_THRESHOLD = 3.0  # % of image
OVERSAMPLE_FACTOR = 3

# In Dataset
if forgery_pct < SMALL_FORGERY_THRESHOLD and forgery_pct > 0:
    self.weights.append(OVERSAMPLE_FACTOR)  # 3x sampling
else:
    self.weights.append(1.0)
```

Small forgeries (<3% of image area) are oversampled 3× to ensure the model sees them frequently during training.

### 4. Combined Loss

```python
class CombinedLoss(nn.Module):
    """Focal Loss (70%) + Dice Loss (30%)"""
    def forward(self, inputs, targets):
        return 0.7 * self.focal(inputs, targets) + \
               0.3 * self.dice(inputs, targets)
```

- **Focal Loss**: Pixel-wise classification, handles class imbalance
- **Dice Loss**: Region-based, encourages compact predictions

## Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Image Size | 512×512 | Preserve small forgery details |
| Batch Size | 4 | Memory constraint with 512px images |
| Epochs | 30 | Sufficient for convergence |
| Learning Rate | 0.0003 | Standard for Adam with FPN |
| Focal Gamma | 3.0 | Aggressive hard example mining |
| Focal Alpha | 0.75 | Weight forged pixels higher |
| Oversample Factor | 3× | For forgeries <3% area |

## Data Pipeline

```
┌─────────────────────────────────────────────────────────┐
│ 1. Load Image (PNG/JPG)                                 │
│    - BGR → RGB conversion                               │
│    - Resize to 512×512                                  │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ 2. Load Mask (.npy)                                     │
│    - Handle 3D masks → max projection                   │
│    - Resize to 512×512                                  │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ 3. Augmentation (training only)                         │
│    - Horizontal/Vertical flip                           │
│    - Random rotation (90°)                              │
│    - Color jitter (brightness, contrast, saturation)    │
│    - Gaussian blur                                      │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ 4. Normalization                                        │
│    - ImageNet mean: [0.485, 0.456, 0.406]              │
│    - ImageNet std: [0.229, 0.224, 0.225]               │
└─────────────────────────────────────────────────────────┘
```

## Training Loop

```python
for epoch in range(EPOCHS):
    model.train()
    for images, masks in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
    
    # Validation
    model.eval()
    val_dice, val_recall = evaluate(model, val_loader)
    
    # Save best model
    if val_dice > best_dice:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'best_dice': val_dice
        }, 'highres_no_ela_best.pth')
```

## Output

**Model File**: `models/highres_no_ela_best.pth`

Contains:
- `model_state_dict`: Trained weights
- `optimizer_state_dict`: Optimizer state (for resuming)
- `epoch`: Best epoch number
- `best_dice`: Best validation Dice score

## Expected Performance

| Metric | Value |
|--------|-------|
| Validation Dice | ~0.35 |
| Recall (all forgeries) | ~70% |
| Recall (small <2%) | ~45% |

## Role in Ensemble

This model has the **highest weight (1.5)** in the ensemble because:
1. It's trained on the full dataset without specialized sampling
2. RGB-only input makes it robust to compression variations
3. Strong baseline performance across all forgery sizes

## Usage

### Training
```bash
python script_1_highres_no_ela.py
```

### In Ensemble
The submission generator loads this model as:
```python
{'name': 'highres_no_ela', 
 'path': 'models/highres_no_ela_best.pth', 
 'in_channels': 3, 
 'weight': 1.5}
```

## Why "No ELA"?

Early experiments showed that adding ELA (Error Level Analysis) as a 4th channel caused overfitting in some cases. This model uses pure RGB to:
1. Avoid ELA artifacts from JPEG quality variations
2. Focus on visual inconsistencies rather than compression patterns
3. Complement script_5 which does use ELA

## Common Issues

### Out of Memory
Reduce batch size:
```python
BATCH_SIZE = 2  # Instead of 4
```

### Slow Training
- Ensure GPU is being used: `torch.cuda.is_available()` should return `True`
- Check DataLoader workers: `num_workers=4` by default

### Poor Small Forgery Detection
This model isn't optimized for tiny forgeries. For better small forgery detection, see:
- `script_3_hard_fn_mining.py` - Trains on missed detections
- `script_6_small_forgery_specialist.py` - Specialized for <2% forgeries
