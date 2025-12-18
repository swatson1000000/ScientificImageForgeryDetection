# Script 5: Comprehensive Model (RGB + ELA)

## Overview

`script_5_comprehensive.py` trains the most feature-rich model in the ensemble, combining **RGB images with Error Level Analysis (ELA)** as a 4-channel input. ELA reveals compression inconsistencies that are invisible in standard RGB, making this model particularly effective at detecting copy-paste forgeries from different sources.

## Purpose

The primary goals are:
- Leverage ELA to detect compression-based forgery artifacts
- Combine visual (RGB) and forensic (ELA) features
- Detect forgeries that RGB-only models miss
- Provide complementary detection to other ensemble members

## What is ELA?

### Error Level Analysis Explained

```
┌─────────────────────────────────────────────────────────┐
│              ERROR LEVEL ANALYSIS (ELA)                 │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Original Image                                         │
│       │                                                 │
│       ▼                                                 │
│  Re-compress as JPEG (quality 90)                       │
│       │                                                 │
│       ▼                                                 │
│  Compute: |Original - Recompressed|                     │
│       │                                                 │
│       ▼                                                 │
│  ELA Result: Highlights compression inconsistencies     │
│                                                         │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐                 │
│  │░░░░░░░░░│  │▓▓▓░░░░░░│  │█████░░░░│                 │
│  │░░░░░░░░░│  │░░░░░░░░░│  │█████░░░░│                 │
│  │░░░░░░░░░│  │░░░░░░░░░│  │░░░░░░░░░│                 │
│  └─────────┘  └─────────┘  └─────────┘                 │
│  Authentic    Forgery      ELA reveals                  │
│  (uniform)    (pasted)     (bright = different)         │
└─────────────────────────────────────────────────────────┘
```

### Why ELA Works

When you re-save a JPEG image:
- **Authentic regions**: Similar error levels (consistent compression history)
- **Forged regions**: Different error levels (came from different source)

```python
def compute_ela(image_bgr, quality=90):
    """Compute Error Level Analysis."""
    # Re-compress the image
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded = cv2.imencode('.jpg', image_bgr, encode_param)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    
    # Compute absolute difference
    ela = cv2.absdiff(image_bgr, decoded)
    
    # Convert to grayscale and amplify
    ela_gray = cv2.cvtColor(ela, cv2.COLOR_BGR2GRAY)
    ela_amplified = np.clip(ela_gray.astype(np.float32) * 10, 0, 255)
    
    return ela_amplified.astype(np.uint8)
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Input Image                          │
│                    512×512×4 (RGB + ELA)                │
│                                                         │
│    ┌─────────┐    ┌─────────┐                          │
│    │   RGB   │    │   ELA   │                          │
│    │  (3ch)  │    │  (1ch)  │                          │
│    └────┬────┘    └────┬────┘                          │
│         └──────┬───────┘                               │
│                │                                        │
│                ▼                                        │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│              EfficientNet-B2 Encoder                    │
│              (ImageNet pretrained, adapted for 4ch)     │
│              First conv: 3→4 input channels             │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              FPN (Feature Pyramid Network)              │
│              Multi-scale feature fusion                 │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              Attention Gate                             │
│              Learned spatial attention                  │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              Output Mask                                │
│              512×512×1 (probability)                    │
└─────────────────────────────────────────────────────────┘
```

## Key Components

### 1. 4-Channel Input Processing

```python
def prepare_image_ela(image_bgr, size=512):
    """Prepare 4-channel RGB+ELA image."""
    # Compute ELA
    ela = compute_ela(image_bgr)
    
    # Convert to RGB
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # Resize both
    image = cv2.resize(image, (size, size))
    ela = cv2.resize(ela, (size, size))
    
    # Normalize RGB (ImageNet stats)
    image = image.astype(np.float32) / 255.0
    image[..., 0] = (image[..., 0] - 0.485) / 0.229
    image[..., 1] = (image[..., 1] - 0.456) / 0.224
    image[..., 2] = (image[..., 2] - 0.406) / 0.225
    
    # Normalize ELA (simple 0-1)
    ela = ela.astype(np.float32) / 255.0
    
    # Stack to 4 channels
    image_4ch = np.zeros((size, size, 4), dtype=np.float32)
    image_4ch[..., :3] = image
    image_4ch[..., 3] = ela
    
    return torch.from_numpy(image_4ch.transpose(2, 0, 1)).float()
```

### 2. Model Adaptation for 4 Channels

```python
# Create base model with 4 input channels
base_model = smp.FPN(
    encoder_name='timm-efficientnet-b2',
    encoder_weights='imagenet',  # Still use ImageNet weights
    in_channels=4,  # 4 instead of 3
    classes=1
)
```

The first convolutional layer is automatically adjusted:
- Original: `Conv2d(3, 32, kernel_size=3)`
- Adapted: `Conv2d(4, 32, kernel_size=3)`

The extra channel weights are initialized randomly but fine-tuned during training.

### 3. ELA-Specific Dataset

```python
class ComprehensiveDataset(Dataset):
    def __getitem__(self, idx):
        # Load image
        image = cv2.imread(self.image_paths[idx])
        
        # Compute ELA
        ela = compute_ela(image)
        
        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask = np.load(self.mask_paths[idx])
        
        # Apply augmentation to image+ELA together
        # (important: augment both consistently)
        if self.augment:
            # Stack for augmentation
            image_4ch = np.dstack([image, ela])
            transformed = self.transform(image=image_4ch, mask=mask)
            image_4ch = transformed['image']
            mask = transformed['mask']
        
        return image_4ch, mask
```

## ELA Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| JPEG Quality | 90 | Standard for forensic ELA |
| Amplification | 10× | Make differences visible |
| Output Range | 0-255 | Grayscale image |

### Why Quality 90?

```
Quality 100: Minimal compression → minimal differences
Quality 90:  Good compression → detectable differences ✓
Quality 70:  Heavy compression → too much noise
```

Quality 90 is the forensic standard for ELA.

## Training Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Image Size | 512×512 | Standard |
| Channels | 4 (RGB+ELA) | Additional ELA channel |
| Batch Size | 4 | Same as others |
| Epochs | 30 | Standard |
| Learning Rate | 0.0003 | Standard |
| Focal Gamma | 3.0 | Standard |

## Training Flow

```
┌─────────────────────────────────────────────────────────┐
│ 1. Load Training Images                                 │
│    └── For each image, compute ELA on-the-fly          │
│    └── No pre-computed ELA storage needed               │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ 2. Create 4-Channel Samples                             │
│    └── RGB channels: 0, 1, 2                            │
│    └── ELA channel: 3                                   │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ 3. Train with Combined Loss                             │
│    └── Focal Loss (70%) + Dice Loss (30%)              │
│    └── Small forgery oversampling (3×)                  │
│    └── 30 epochs                                        │
└─────────────────────────────────────────────────────────┘
```

## Output

**Model File**: `models/comprehensive_best.pth`

## Expected Performance

| Metric | Value |
|--------|-------|
| Validation Dice | ~0.30 |
| Recall | ~60% |
| Copy-paste Detection | **Best** |

This model may have lower overall metrics but excels at specific forgery types.

## Role in Ensemble

This model has **weight 0.5** (lowest) in the ensemble:

```python
{'name': 'comprehensive', 
 'path': 'models/comprehensive_best.pth', 
 'in_channels': 4,  # Note: 4 channels!
 'weight': 0.5}
```

Lower weight because:
1. ELA can be unreliable for non-JPEG sources
2. May produce false positives on legitimate edits
3. Best used as tie-breaker, not primary signal

## When ELA Helps

| Forgery Type | ELA Effectiveness |
|--------------|-------------------|
| Copy-paste from web | ✓✓✓ Excellent |
| JPEG splicing | ✓✓✓ Excellent |
| Different source compression | ✓✓ Good |
| Same-image clone | ✓ Moderate |
| In-painting | ✗ Poor |
| PNG manipulations | ✗ Poor |

## When ELA Struggles

```
┌─────────────────────────────────────────────────────────┐
│ ELA Limitations:                                        │
│                                                         │
│ • PNG images (no JPEG compression history)              │
│ • Same-quality sources (similar error levels)           │
│ • Multiple re-saves (uniform error levels)              │
│ • Professional edits (matched compression)              │
│ • Non-JPEG originals (TIFF, RAW)                        │
└─────────────────────────────────────────────────────────┘
```

This is why ELA is combined with RGB, not used alone.

## Usage

### Training
```bash
python script_5_comprehensive.py
```

### Prerequisites
- Training images and masks available
- Images should include JPEGs for ELA effectiveness

### Monitoring
```bash
tail -f logs/train_5_comprehensive.log
```

## Inference Note

When using this model for inference, ELA must be computed:

```python
def predict_with_ela(model, image_path):
    image_bgr = cv2.imread(image_path)
    
    # Prepare 4-channel input
    input_tensor = prepare_image_ela(image_bgr)
    input_tensor = input_tensor.unsqueeze(0).to(DEVICE)
    
    # Predict
    with torch.no_grad():
        pred = torch.sigmoid(model(input_tensor))
    
    return pred.cpu().numpy()[0, 0]
```

The submission generator handles this automatically:

```python
if cfg['in_channels'] == 3:
    input_tensor = prepare_image_rgb(image_bgr)
else:
    input_tensor = prepare_image_ela(image_bgr)  # 4-channel
```

## Technical Details

### ImageNet Weights with 4 Channels

When using `in_channels=4`, segmentation-models-pytorch:
1. Loads ImageNet weights for the original 3-channel network
2. Creates a new first conv layer with 4 input channels
3. Copies RGB weights to first 3 channels
4. Randomly initializes the 4th channel weights

```python
# Conceptually:
new_conv.weight[:, :3, :, :] = pretrained_conv.weight  # RGB
new_conv.weight[:, 3:4, :, :] = random_init()  # ELA (random)
```

### ELA Computation Speed

ELA computation adds overhead:
- JPEG encode/decode: ~5ms per image
- Difference calculation: ~1ms
- Total: ~6ms extra per image

For training, this is computed on-the-fly in the DataLoader.

### Memory Considerations

4-channel images use 33% more GPU memory than 3-channel:
```
3 channels: 512×512×3×4 = 3.15 MB per image
4 channels: 512×512×4×4 = 4.19 MB per image
```

Batch size of 4 is usually fine with 4GB+ VRAM.

## Common Issues

### ELA Shows Nothing
- Image might be PNG (no JPEG history)
- Already heavily compressed (uniform errors)
- Try lower JPEG quality: `quality=80`

### False Positives from ELA
- Text/annotations have different compression
- Natural high-contrast edges
- Noise in dark regions
- Solution: Lower ensemble weight (already 0.5)

### Model Learns Wrong Patterns
- ELA channel might dominate
- Verify augmentation applies to all 4 channels consistently
- Check ELA normalization (should be 0-1)

## Visualization

To visualize ELA:

```python
import matplotlib.pyplot as plt

image = cv2.imread('test_image.jpg')
ela = compute_ela(image)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
ax1.set_title('Original')
ax2.imshow(ela, cmap='hot')
ax2.set_title('ELA (bright = different compression)')
plt.savefig('ela_visualization.png')
```

This helps verify ELA is working correctly for your images.
