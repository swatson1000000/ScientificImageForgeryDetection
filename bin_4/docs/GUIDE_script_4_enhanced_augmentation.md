# Script 4: Enhanced Augmentation Training

## Overview

`script_4_enhanced_augmentation.py` trains a model using **aggressive data augmentation**, including a novel **copy-paste augmentation** technique. This approach creates synthetic forgeries during training, dramatically increasing the diversity of forgery patterns the model learns.

## Purpose

The primary goals are:
- Expand training data diversity through augmentation
- Simulate different forgery techniques via copy-paste
- Improve generalization to unseen forgery patterns
- Handle variations in image quality and compression

## The Problem It Solves

```
┌─────────────────────────────────────────────────────────┐
│ Limited Training Data Issues:                           │
│                                                         │
│ • Same forgery patterns repeated                        │
│ • Limited forgery sizes and positions                   │
│ • Consistent image quality                              │
│ • Fixed compression artifacts                           │
│ • Model memorizes specific patterns                     │
└─────────────────────────────────────────────────────────┘
```

Enhanced augmentation creates virtually unlimited variations.

## Key Innovation: Copy-Paste Augmentation

### Concept

```
┌─────────────────────────────────────────────────────────┐
│                  COPY-PASTE AUGMENTATION                │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Source Image          Target Image                     │
│  ┌─────────┐          ┌─────────┐                      │
│  │    █    │          │         │                      │
│  │   ███   │   -->    │   ███   │                      │
│  │    █    │          │         │                      │
│  └─────────┘          └─────────┘                      │
│  (forgery patch)      (new location)                   │
│                                                         │
│  This simulates copy-paste forgery creation!           │
└─────────────────────────────────────────────────────────┘
```

### Implementation

```python
class CopyPasteAugmentation:
    """Copy-paste augmentation for forgery detection."""
    
    def __init__(self, forgery_patches, prob=0.5):
        self.forgery_patches = forgery_patches  # (image_patch, mask_patch) tuples
        self.prob = prob
    
    def __call__(self, image, mask):
        if random.random() > self.prob:
            return image, mask
        
        # Select random patch
        patch_img, patch_mask = random.choice(self.forgery_patches)
        
        # Random scale (0.5× to 1.5×)
        scale = random.uniform(0.5, 1.5)
        new_h = int(patch_img.shape[0] * scale)
        new_w = int(patch_img.shape[1] * scale)
        
        patch_img = cv2.resize(patch_img, (new_w, new_h))
        patch_mask = cv2.resize(patch_mask, (new_w, new_h))
        
        # Random position
        y = random.randint(0, image.shape[0] - new_h)
        x = random.randint(0, image.shape[1] - new_w)
        
        # Blend patch where mask is positive
        blend_mask = (patch_mask > 0.5).astype(np.float32)
        image[y:y+new_h, x:x+new_w] = (
            image[y:y+new_h, x:x+new_w] * (1 - blend_mask) +
            patch_img * blend_mask
        )
        
        # Update mask
        mask[y:y+new_h, x:x+new_w] = np.maximum(
            mask[y:y+new_h, x:x+new_w],
            patch_mask
        )
        
        return image, mask
```

## How It Works

### Phase 1: Extract Forgery Patches

```python
def extract_forgery_patches(image_paths, mask_paths, min_size=32, max_patches=1000):
    """Extract forgery regions as reusable patches."""
    patches = []
    
    for img_path, mask_path in zip(image_paths, mask_paths):
        image = cv2.imread(str(img_path))
        mask = np.load(str(mask_path))
        
        # Find connected components (individual forgeries)
        contours, _ = cv2.findContours(
            (mask > 0.5).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w >= min_size and h >= min_size:
                patch_img = image[y:y+h, x:x+w]
                patch_mask = mask[y:y+h, x:x+w]
                patches.append((patch_img, patch_mask))
    
    return patches[:max_patches]
```

### Phase 2: Training with Augmentation

```
┌─────────────────────────────────────────────────────────┐
│              Training Data Generation                   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Original Image ──┐                                     │
│                   │                                     │
│  Standard Augs ───┼──▶ Augmented Image ───┐            │
│  (flip, rotate)   │                       │            │
│                   │                       ▼            │
│  Copy-Paste ──────┘           ┌───────────────────┐    │
│  (30% prob)                   │ Final Training    │    │
│                               │ Sample with       │    │
│                               │ Combined Mask     │    │
│                               └───────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

## Augmentation Pipeline

```python
self.transform = A.Compose([
    # Spatial augmentations
    A.RandomResizedCrop(size=(512, 512), scale=(0.7, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Affine(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
    
    # Color augmentations (STRONGER than other scripts)
    A.OneOf([
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
    ], p=0.7),  # 70% probability!
    
    # Noise and blur
    A.OneOf([
        A.GaussNoise(),
        A.GaussianBlur(blur_limit=(3, 7)),
        A.MotionBlur(blur_limit=7),
    ], p=0.3),
    
    # Compression artifacts
    A.ImageCompression(quality_range=(70, 100), p=0.3),
    
    # Cutout/GridMask
    A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(16, 32), 
                   hole_width_range=(16, 32), p=0.2),
    
    # Normalize
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])
```

## Augmentation Comparison

| Augmentation | Script 1 | Script 4 | Effect |
|--------------|----------|----------|--------|
| Color Jitter | 0.2 | **0.3** | Stronger color variation |
| Jitter Prob | 0.5 | **0.7** | More frequent |
| Noise Types | 1 | **3** | Gaussian, Blur, Motion |
| Compression | No | **Yes** | JPEG artifacts simulation |
| Cutout | No | **Yes** | Occlusion robustness |
| Copy-Paste | No | **Yes** | Synthetic forgeries |

## Key Components

### 1. Enhanced Dataset

```python
class EnhancedForgeryDataset(Dataset):
    def __init__(self, image_paths, mask_paths, augment=True, copy_paste_patches=None):
        self.copy_paste = None
        if copy_paste_patches:
            self.copy_paste = CopyPasteAugmentation(copy_paste_patches, prob=0.3)
    
    def __getitem__(self, idx):
        image = load_image(self.image_paths[idx])
        mask = load_mask(self.mask_paths[idx])
        
        # Resize first
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
        
        # Apply copy-paste BEFORE standard augmentation
        if self.copy_paste:
            image, mask = self.copy_paste(image, mask)
        
        # Then apply standard augmentation
        transformed = self.transform(image=image, mask=mask)
        return transformed['image'], transformed['mask']
```

### 2. Training Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Image Size | 512×512 | Standard |
| Batch Size | 4 | Memory constrained |
| Epochs | 30 | Standard |
| Learning Rate | 1e-4 | Standard |
| Copy-Paste Prob | 0.3 | 30% of samples |
| Min Patch Size | 32px | Avoid tiny noise patches |

## Training Flow

```
┌─────────────────────────────────────────────────────────┐
│ 1. Extract Forgery Patches                              │
│    └── Scan all training images                         │
│    └── Extract bounded forgery regions                  │
│    └── Store as (image_patch, mask_patch) pairs         │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ 2. Create Enhanced Dataset                              │
│    └── Original images + masks                          │
│    └── Copy-paste augmentation (30%)                    │
│    └── Strong standard augmentation                     │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ 3. Train From Scratch                                   │
│    └── Fresh EfficientNet-B2 FPN model                  │
│    └── ImageNet pretrained encoder                      │
│    └── 30 epochs                                        │
└─────────────────────────────────────────────────────────┘
```

**Note**: Unlike scripts 2-3, this trains from scratch (not fine-tuned).

## Output

**Model File**: `models/enhanced_aug_best.pth`

## Expected Performance

| Metric | Value |
|--------|-------|
| Validation Dice | ~0.32 |
| Recall | ~65% |
| Generalization | **Best** |

This model may have slightly lower peak performance but generalizes better to unseen data.

## Role in Ensemble

This model has **weight 0.8** in the ensemble:

```python
{'name': 'enhanced_aug', 
 'path': 'models/enhanced_aug_best.pth', 
 'in_channels': 3, 
 'weight': 0.8}
```

Lower weight because:
1. May be noisier on validation set
2. Complementary role (diversity, not raw performance)
3. Strength is in catching unusual patterns

## Why Copy-Paste Works

### The Intuition

Real forgeries are created by copying regions and pasting them elsewhere. By simulating this during training:
1. Model learns to detect paste boundaries
2. Sees forgeries in diverse contexts
3. Learns forgery characteristics, not just specific patterns

### Example Scenario

```
Training without copy-paste:
  Model sees: "Forgery A at position (100, 200)"
  Model learns: "Pattern at position (100, 200)"
  
Training with copy-paste:
  Model sees: "Forgery A at position (50, 300)"
  Model sees: "Forgery A at position (400, 150)"
  Model sees: "Forgery B at position (100, 200)"
  Model learns: "Forgery patterns regardless of position"
```

## Augmentation Rationale

### Color Augmentation (Strong)

Scientific images have varying:
- Staining intensities (microscopy)
- White balance (photography)
- Display calibration

Strong color augmentation makes the model robust to these variations.

### Compression Artifacts

```python
A.ImageCompression(quality_range=(70, 100), p=0.3)
```

- Test images may have different compression
- Forgeries often have different compression levels
- Model learns to look past compression artifacts

### Cutout/CoarseDropout

```python
A.CoarseDropout(num_holes_range=(1, 8), ...)
```

- Simulates occluded regions
- Forces model to use multiple evidence sources
- Improves robustness to partial visibility

## Usage

### Training
```bash
python script_4_enhanced_augmentation.py
```

### Prerequisites
- Training images and masks available
- No pretrained model required (trains from scratch)

### Monitoring
```bash
tail -f logs/train_4_enhanced.log
```

## Technical Details

### Why Train from Scratch?

Unlike scripts 2-3 which fine-tune:
1. **Different distribution**: Augmented data is substantially different
2. **Avoid bias**: Base model patterns may not transfer well
3. **Fresh perspective**: Learns augmented patterns from ground up
4. **Diversity**: Creates a more diverse ensemble member

### Copy-Paste Probability (30%)

Why 30% instead of higher?
- **Too low (10%)**: Not enough effect
- **Too high (50%+)**: Too many synthetic forgeries, unrealistic
- **30%**: Good balance between original and augmented

### Patch Extraction Details

```python
# Minimum patch size prevents noise
min_size=32  # 32×32 pixels minimum

# Maximum patches prevents memory issues
max_patches=1000

# Contour detection finds individual forgeries
cv2.findContours(..., cv2.RETR_EXTERNAL, ...)
```

## Common Issues

### Model Underperforms
- May need more epochs due to harder training task
- Check that patches are extracted correctly
- Verify augmentation is applying

### Memory Issues
- Reduce max_patches
- Use smaller patch sizes
- Lower batch size

### Augmentation Too Strong
- Reduce color jitter limits
- Lower augmentation probabilities
- Remove cutout

## Visualization

To visualize augmented samples:

```python
import matplotlib.pyplot as plt

dataset = EnhancedForgeryDataset(image_paths, mask_paths, 
                                  augment=True, 
                                  copy_paste_patches=patches)

for i in range(5):
    img, mask = dataset[i]
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(img.permute(1, 2, 0))
    ax2.imshow(mask)
    plt.savefig(f'augmented_sample_{i}.png')
```

This helps verify that augmentation is working correctly.
