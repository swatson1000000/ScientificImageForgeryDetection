# script_2_train_binary_classifier.py - Detailed Reference

## Overview

This script trains a **binary classifier** that distinguishes forged images from authentic images. It serves as Stage 1 of the two-stage pipeline, filtering out obviously authentic images before the slower segmentation stage.

## Purpose

The binary classifier provides:
1. **Speed**: Fast filtering (~10ms per image vs ~200ms for segmentation)
2. **FP Reduction**: Prevents segmentation from running on authentic images
3. **Efficiency**: ~30% of images filtered out, reducing total inference time

## Architecture

### ForgeryClassifier Model

```python
class ForgeryClassifier(nn.Module):
    """Binary classifier: forged (1) vs authentic (0)."""
    
    def __init__(self, backbone='efficientnet_b2'):
        # Backbone: EfficientNet-B2 (pretrained on ImageNet)
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)
        # → outputs 1408 features
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(1408, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        features = self.backbone(x)    # (B, 1408)
        logits = self.classifier(features)  # (B, 1)
        return logits.squeeze(-1)      # (B,)
```

### Model Summary

```
Input: (B, 3, 384, 384) RGB image tensor

EfficientNet-B2 Backbone
├── Stem: Conv 3×3, stride 2
├── Blocks: MBConv layers
└── Head: Global average pooling → 1408 features

Classification Head
├── Linear: 1408 → 256
├── ReLU
├── Dropout: p=0.3
└── Linear: 256 → 1 (logit)

Output: (B,) logit values → sigmoid for probability
```

## Configuration

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `IMG_SIZE` | 384 | Input image resolution |
| `BATCH_SIZE` | 16 | Samples per batch |
| `EPOCHS` | 20 | Training epochs |
| `LR` | 1e-4 | Initial learning rate |

### Training Split

- 90% training / 10% validation
- Stratified: maintains forged/authentic ratio in both splits

## Data Pipeline

### BinaryClassificationDataset

```python
class BinaryClassificationDataset(Dataset):
    """Dataset for forged vs authentic classification."""
    
    def __init__(self, forged_paths, authentic_paths, transform):
        # Combine samples with labels
        self.samples = [
            (path, 1) for path in forged_paths  # forged = 1
        ] + [
            (path, 0) for path in authentic_paths  # authentic = 0
        ]
        
        # Shuffle
        np.random.shuffle(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image=image)['image']
        
        return image, torch.tensor(label, dtype=torch.float32)
```

### Augmentations

**Training Augmentations:**
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
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])
```

**Validation Augmentations:**
```python
A.Compose([
    A.Resize(384, 384),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])
```

## Training Process

### Training Loop

```python
def main():
    # 1. Load data
    forged_images = list(TRAIN_IMAGES_PATH / 'forged').glob('*.png'))
    authentic_images = list((TRAIN_IMAGES_PATH / 'authentic').glob('*.png'))
    
    # 2. Split into train/val (90/10)
    train_forged, val_forged = split(forged_images, 0.9)
    train_authentic, val_authentic = split(authentic_images, 0.9)
    
    # 3. Create model
    model = ForgeryClassifier('efficientnet_b2').to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # 4. Training loop
    for epoch in range(1, EPOCHS + 1):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        
        # Validate
        val_acc, precision, recall, tp, fp, tn, fn = validate(model, val_loader)
        
        # Update learning rate
        scheduler.step()
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'precision': precision,
                'recall': recall,
            }, save_path)
```

### Loss Function

**BCEWithLogitsLoss**: Binary Cross-Entropy with built-in sigmoid
```python
loss = criterion(outputs, labels)
# Internally: sigmoid(outputs) → BCE with labels
```

### Learning Rate Schedule

**CosineAnnealingLR**: Cosine decay from initial LR to 0
```
LR(t) = LR_min + 0.5 * (LR_max - LR_min) * (1 + cos(π * t / T_max))
```

## Metrics

### Training Metrics

| Metric | Description |
|--------|-------------|
| `train_loss` | Average BCE loss per batch |
| `train_acc` | Training accuracy (threshold=0.5) |

### Validation Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| `val_acc` | (TP+TN) / (TP+TN+FP+FN) | Overall accuracy |
| `precision` | TP / (TP+FP) | Positive predictive value |
| `recall` | TP / (TP+FN) | True positive rate |
| `TP` | - | True positives (forged correctly identified) |
| `FP` | - | False positives (authentic labeled as forged) |
| `TN` | - | True negatives (authentic correctly identified) |
| `FN` | - | False negatives (forged labeled as authentic) |

### Target Metrics

For the two-stage pipeline:
- **High Recall**: Want to catch most forged images (minimize FN)
- **Moderate Precision**: Some FP okay (segmentation will filter them)

Typical results:
- Accuracy: 82-85%
- Recall: 85-88%
- Precision: 80-83%

## Usage

### Basic Training

```bash
python bin_4/script_2_train_binary_classifier.py
```

### Expected Output

```
============================================================
TRAINING BINARY CLASSIFIER (Stage 1)
============================================================
Device: cuda
Forged images: 2751
Authentic images: 2377

Train: 2475 forged, 2139 authentic
Val: 276 forged, 238 authentic

Epoch 1/20 - Loss: 0.4521, Train Acc: 0.782, Val Acc: 0.801, Time: 45s
  Precision: 0.789, Recall: 0.823
  TP: 227, FP: 61, TN: 177, FN: 49
  ✓ Saved best model (Acc: 0.801)

...

Epoch 20/20 - Loss: 0.1234, Train Acc: 0.951, Val Acc: 0.828, Time: 43s
  Precision: 0.812, Recall: 0.865
  TP: 239, FP: 55, TN: 183, FN: 37

Training complete! Best Acc: 0.828
Model saved to: models/binary_classifier_best.pth
```

## Input Requirements

### Training Data Structure

```
train_images/
├── forged/           # 2751 forged images
│   ├── 10001.png
│   ├── 10002.png
│   └── ...
└── authentic/        # 2377 authentic images
    ├── 20001.png
    ├── 20002.png
    └── ...
```

### Image Format

- Format: PNG, JPG, JPEG, TIFF
- Size: Any (resized to 384×384)
- Channels: RGB (BGR files converted automatically)

## Output

### Model Checkpoint

Location: `models/binary_classifier_best.pth`

Contents:
```python
{
    'epoch': int,                    # Best epoch number
    'model_state_dict': OrderedDict, # Model weights
    'val_acc': float,                # Best validation accuracy
    'precision': float,              # Precision at best epoch
    'recall': float,                 # Recall at best epoch
}
```

### Loading the Model

```python
from script_2_train_binary_classifier import ForgeryClassifier

# Create model
model = ForgeryClassifier('efficientnet_b2').to(device)

# Load weights
checkpoint = torch.load('models/binary_classifier_best.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Inference
with torch.no_grad():
    logit = model(img_tensor)
    prob = torch.sigmoid(logit).item()
    is_forged = prob > 0.25  # Use threshold 0.25 for production
```

## Performance Notes

### Training Time

- ~30-60 minutes on GPU
- ~2-3 hours on CPU

### Memory Usage

- ~3-4 GB GPU memory at batch_size=16
- Reduce to batch_size=8 if OOM

### Inference Speed

- ~10ms per image on GPU
- ~50ms per image on CPU

## Threshold Selection

The validation uses threshold=0.5, but production uses **threshold=0.25**:

| Threshold | Effect |
|-----------|--------|
| 0.5 (default) | Balanced precision/recall |
| 0.25 (production) | Higher recall, more images to segmentation |
| 0.15 | Very high recall, minimal filtering |
| 0.35 | Balanced filtering |

Lower threshold = fewer false negatives (missed forgeries) but more images pass to Stage 2.

## Troubleshooting

### "No forged images found"

Ensure images are in correct directory:
```bash
ls train_images/forged/*.png | head
ls train_images/authentic/*.png | head
```

### Low Accuracy

Possible causes:
1. Data imbalance - add class weighting
2. Learning rate too high - try 1e-5
3. Overfitting - add more dropout or augmentation

### CUDA Out of Memory

Reduce batch size:
```python
BATCH_SIZE = 8  # Instead of 16
```
