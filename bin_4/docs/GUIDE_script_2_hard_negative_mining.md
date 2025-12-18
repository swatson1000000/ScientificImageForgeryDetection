# Script 2: Hard Negative Mining

## Overview

`script_2_hard_negative_mining.py` improves the ensemble's **precision** by training on images that cause false positives. It identifies authentic images that the base model incorrectly flags as forged, then trains a specialized model to reject these "hard negatives."

## Purpose

The primary goal is to reduce false positives by:
- Identifying authentic images that confuse the base model
- Training on these hard cases with higher sampling weight
- Learning to distinguish authentic artifacts from actual forgeries

## The Problem It Solves

```
┌─────────────────────────────────────────────────────────┐
│ Base Model (script_1) False Positive Sources:           │
│                                                         │
│ • Natural texture boundaries (cells, tissue edges)      │
│ • Strong gradients in microscopy images                 │
│ • JPEG compression artifacts in authentic images        │
│ • High-contrast annotations or labels                   │
│ • Dust/debris on slides                                 │
└─────────────────────────────────────────────────────────┘
```

These authentic features can look like manipulation artifacts to the base model.

## How It Works

### Phase 1: Find Hard Negatives

```python
def find_hard_negatives(model, authentic_dir, threshold=0.5, max_samples=500):
    """Scan authentic images for false positives."""
    hard_negatives = []
    
    for img_path in authentic_images:
        # Get prediction
        with torch.no_grad():
            pred = torch.sigmoid(model(input_tensor))
        
        # Count false positive pixels
        fp_pixels = (pred > threshold).sum()
        fp_ratio = fp_pixels / pred.size
        
        # If >0.1% false positive pixels, it's a hard negative
        if fp_ratio > 0.001:
            hard_negatives.append({
                'path': str(img_path),
                'fp_ratio': fp_ratio,
                'max_conf': pred.max()
            })
    
    # Sort by worst offenders first
    hard_negatives.sort(key=lambda x: x['fp_ratio'], reverse=True)
    return hard_negatives[:max_samples]
```

### Phase 2: Train on Mixed Dataset

```
┌────────────────────────────────────────────────┐
│          Training Dataset Composition          │
├────────────────────────────────────────────────┤
│ Forged Images (with masks)                     │
│   └── Normal training samples                  │
│                                                │
│ Hard Negative Authentics (mask = all zeros)    │
│   └── 2× sampling weight                       │
│   └── Model learns "this is NOT forgery"       │
└────────────────────────────────────────────────┘
```

## Key Components

### 1. Hard Negative Dataset

```python
class HardNegativeDataset(Dataset):
    def __init__(self, forged_paths, mask_paths, hard_negative_paths, ...):
        # Combine forged images with hard negatives
        self.all_paths = list(zip(forged_paths, mask_paths, [True]*len(forged_paths)))
        self.all_paths += [(hn, None, False) for hn in hard_negative_paths]
        
        # Weight assignment
        self.weights = []
        for img_path, mask_path, is_forged in self.all_paths:
            if not is_forged:
                self.weights.append(2.0)  # Hard negatives get 2× weight
            else:
                # Forged images weighted by forgery size
                forgery_ratio = compute_forgery_ratio(mask_path)
                if forgery_ratio < 1.0:
                    self.weights.append(3.0)  # Tiny forgeries
                elif forgery_ratio < 5.0:
                    self.weights.append(2.0)  # Small forgeries
                else:
                    self.weights.append(1.0)  # Large forgeries
```

### 2. Sample Weighting Strategy

| Sample Type | Weight | Rationale |
|-------------|--------|-----------|
| Hard negative (authentic) | 2.0× | Focus on reducing FP |
| Tiny forgery (<1%) | 3.0× | Rare, important to detect |
| Small forgery (1-5%) | 2.0× | Common failure mode |
| Large forgery (>5%) | 1.0× | Already detected well |

### 3. Pretrained Initialization

```python
# Load pretrained weights from script_1
model = load_model('models/highres_no_ela_best.pth', DEVICE)

# Fine-tune on hard negative dataset
# Uses lower learning rate for stability
optimizer = optim.Adam(model.parameters(), lr=1e-4)
```

## Training Flow

```
┌─────────────────────────────────────────────────────────┐
│ 1. Load Base Model (script_1)                           │
│    └── Pre-trained highres_no_ela_best.pth             │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ 2. Scan Authentic Images for False Positives            │
│    └── Input: train_images/authentic/*.png             │
│    └── Output: List of hard negatives + FP ratios      │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ 3. Create Combined Dataset                              │
│    └── All forged images with masks                     │
│    └── Hard negative authentics with zero masks         │
│    └── Weighted random sampling                         │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ 4. Fine-tune Model                                      │
│    └── 30 epochs, LR=1e-4                              │
│    └── Save best model based on combined metric         │
└─────────────────────────────────────────────────────────┘
```

## Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning Rate | 1e-4 | Lower than script_1 for fine-tuning |
| FP Threshold | 0.5 | Predictions above this are FP |
| Min FP Ratio | 0.001 | 0.1% of pixels must be FP |
| Hard Negative Weight | 2.0× | In weighted sampler |
| Max Hard Negatives | 500 | Limit for memory |

## Augmentation Pipeline

```python
if augment:
    self.transform = A.Compose([
        A.RandomResizedCrop(size=(512, 512), scale=(0.7, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.OneOf([
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
        ], p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
```

## Output

**Model File**: `models/hard_negative_best.pth`

**Expected Impact**:
- Reduced false positive rate on authentic images
- Slightly higher precision with minimal recall loss

## Expected Performance

| Metric | Before (script_1) | After (script_2) |
|--------|-------------------|------------------|
| Precision | ~93% | ~97% |
| Recall | ~70% | ~68% |
| FP on Authentic | 7% | 3% |

## Role in Ensemble

This model has **weight 1.2** in the ensemble:

```python
{'name': 'hard_negative', 
 'path': 'models/hard_negative_best.pth', 
 'in_channels': 3, 
 'weight': 1.2}
```

It contributes to the weighted average by:
1. Suppressing predictions on authentic-looking regions
2. Reducing overconfident false positives
3. Complementing high-recall models (script_3, script_6)

## Relationship to Other Scripts

```
script_1 (base model)
    │
    ├──▶ script_2 (hard NEGATIVE mining) ←── You are here
    │    └── Focuses on FALSE POSITIVES
    │
    └──▶ script_3 (hard FALSE NEGATIVE mining)
         └── Focuses on MISSED DETECTIONS
```

- **Script 2**: "Stop flagging these authentic images"
- **Script 3**: "Start detecting these missed forgeries"

Together, they provide complementary error correction.

## Typical Hard Negatives

Based on analysis, common hard negatives include:

| Image Type | Why It Confuses the Model |
|------------|---------------------------|
| High-contrast microscopy | Sharp edges look like cut/paste |
| Annotated images | Text/arrows have distinct compression |
| Low-quality scans | JPEG artifacts everywhere |
| Tissue boundaries | Natural gradients look suspicious |
| Dust particles | Point artifacts like manipulation |

## Usage

### Training
```bash
# Requires script_1 model to exist first
python script_2_hard_negative_mining.py
```

### Prerequisites
- `models/highres_no_ela_best.pth` must exist
- `train_images/authentic/` directory with authentic samples

### Monitoring
```bash
tail -f logs/train_2_hard_neg.log
```

## Technical Details

### Why Fine-tune Instead of Train from Scratch?

1. **Transfer learning**: Base model already knows forgery patterns
2. **Efficiency**: Converges faster (30 epochs vs 50+)
3. **Stability**: Less risk of catastrophic forgetting
4. **Focus**: Only need to correct specific error patterns

### Why Weight Hard Negatives at 2×?

- Lower weight (1×): Insufficient correction
- Higher weight (3-4×): Model becomes too conservative, misses actual forgeries
- 2× is experimentally optimal balance

## Common Issues

### No Hard Negatives Found
- Check that `train_images/authentic/` exists and contains PNG files
- Lower the threshold: `threshold=0.3`
- Ensure base model is loaded correctly

### Model Becomes Too Conservative
- Reduce hard negative weight: `weight=1.5`
- Increase forgery sample weight
- Use fewer hard negative samples

### Out of Memory
- Reduce batch size
- Limit max_samples in `find_hard_negatives()`
