# Script 3: Hard False Negative Mining

## Overview

`script_3_hard_fn_mining.py` improves the ensemble's **recall** by training on forgeries that the base model fails to detect. It identifies images where forgeries are present but missed, then trains a specialized model with heavy oversampling of these "hard false negatives."

## Purpose

The primary goal is to increase recall by:
- Identifying forged images where the model misses >50% of the forgery
- Oversampling these hard cases during training
- Learning subtle forgery patterns that evade basic detection

## The Problem It Solves

```
┌─────────────────────────────────────────────────────────┐
│ Base Model (script_1) Misses These Forgeries:           │
│                                                         │
│ • Very small forgeries (<1% of image)                   │
│ • Forgeries with smooth blending                        │
│ • Copy-paste from same image (similar statistics)       │
│ • Forgeries in noisy/textured regions                   │
│ • Heavily compressed manipulated regions                │
└─────────────────────────────────────────────────────────┘
```

## How It Works

### Phase 1: Find Hard False Negatives

```python
def find_hard_false_negatives(model, image_paths, mask_paths, threshold=0.5):
    """Find forgeries the model fails to detect."""
    hard_fn = []
    
    for img_path, mask_path in zip(image_paths, mask_paths):
        # Load and predict
        gt_mask = np.load(mask_path)
        pred = model_predict(img_path)
        
        # Calculate pixel-level recall
        gt_binary = (gt_mask > 0.5)
        pred_binary = (pred > threshold)
        
        gt_pixels = gt_binary.sum()
        detected_pixels = (pred_binary & gt_binary).sum()
        recall = detected_pixels / gt_pixels
        
        # If recall < 50%, this is a hard false negative
        if recall < 0.5:
            hard_fn.append({
                'image_path': img_path,
                'mask_path': mask_path,
                'recall': recall,
                'forgery_ratio': forgery_ratio
            })
    
    return hard_fn
```

### Phase 2: Train with Heavy Oversampling

```
┌────────────────────────────────────────────────┐
│          Sample Weighting Strategy             │
├────────────────────────────────────────────────┤
│ Hard False Negative (recall < 50%)    →  5.0× │
│ Tiny forgery (<1% area)               →  3.0× │
│ Small forgery (1-5% area)             →  2.0× │
│ Normal forgery (>5% area)             →  1.0× │
└────────────────────────────────────────────────┘
```

## Key Components

### 1. Hard FN Dataset

```python
class HardFNDataset(Dataset):
    def __init__(self, image_paths, mask_paths, hard_fn_indices, 
                 img_size=512, augment=True, hard_fn_weight=5.0):
        self.hard_fn_indices = set(hard_fn_indices)
        
        # Compute weights
        self.weights = []
        for i, mask_path in enumerate(mask_paths):
            if i in self.hard_fn_indices:
                self.weights.append(hard_fn_weight)  # 5× for hard FN
            else:
                # Weight by forgery size
                forgery_ratio = compute_forgery_ratio(mask_path)
                if forgery_ratio < 1.0:
                    self.weights.append(3.0)
                elif forgery_ratio < 5.0:
                    self.weights.append(2.0)
                else:
                    self.weights.append(1.0)
```

### 2. Analysis by Size

```python
# After finding hard FN, analyze by size
small = [x for x in hard_fn if x['forgery_ratio'] < 2]
medium = [x for x in hard_fn if 2 <= x['forgery_ratio'] < 10]
large = [x for x in hard_fn if x['forgery_ratio'] >= 10]

print(f"Small (<2%): {len(small)}")    # Usually most
print(f"Medium (2-10%): {len(medium)}")
print(f"Large (>10%): {len(large)}")   # Usually fewest
```

**Typical distribution**: 70% small, 25% medium, 5% large

### 3. Recall-Focused Training

The model optimizes for recall while maintaining acceptable precision:

```python
# During validation
val_recall = calculate_recall(predictions, targets)
val_dice = calculate_dice(predictions, targets)

# Combined metric favoring recall
combined_metric = 0.6 * val_recall + 0.4 * val_dice

if combined_metric > best_metric:
    save_model()
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
│ 2. Scan All Forged Images for Missed Detections         │
│    └── For each image, calculate recall                 │
│    └── Flag images with recall < 50%                    │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ 3. Analyze Failure Patterns                             │
│    └── Group by forgery size                            │
│    └── Identify common characteristics                  │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ 4. Train with Weighted Sampling                         │
│    └── Hard FN get 5× weight                            │
│    └── Small forgeries get 3× weight                    │
│    └── 30 epochs                                        │
└─────────────────────────────────────────────────────────┘
```

## Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Recall Threshold | 0.5 | <50% recall = hard FN |
| Hard FN Weight | 5.0× | Very aggressive oversampling |
| Tiny Forgery Weight | 3.0× | For <1% forgeries |
| Small Forgery Weight | 2.0× | For 1-5% forgeries |
| Learning Rate | 1e-4 | Fine-tuning rate |
| Epochs | 30 | Standard |

## Why Weight 5× for Hard FN?

The 5× weight is empirically chosen to:
1. Ensure model sees hard FN multiple times per epoch
2. Force gradient updates on these specific patterns
3. Overcome the natural sampling bias toward easy examples

```
Without 5× weight:
  Hard FN seen: ~1-2 times/epoch
  Model ignores rare patterns

With 5× weight:
  Hard FN seen: ~5-10 times/epoch
  Model forced to learn these patterns
```

## Output

**Model File**: `models/high_recall_best.pth`

Note: Named "high_recall" because the goal is maximizing recall.

## Expected Performance

| Metric | Before (script_1) | After (script_3) |
|--------|-------------------|------------------|
| Overall Recall | ~70% | ~78% |
| Recall (<2%) | ~45% | ~65% |
| Precision | ~93% | ~90% |

The precision drop is acceptable because:
1. Other ensemble members (script_2) maintain precision
2. Weighted ensemble averaging balances the trade-off

## Role in Ensemble

This model has **weight 1.0** in the ensemble:

```python
{'name': 'high_recall', 
 'path': 'models/high_recall_best.pth', 
 'in_channels': 3, 
 'weight': 1.0}
```

It contributes to the weighted average by:
1. Boosting predictions for subtle forgeries
2. Reducing false negative rate
3. Complementing high-precision models (script_2)

## Relationship to Other Scripts

```
script_1 (base model)
    │
    ├──▶ script_2 (hard NEGATIVE mining)
    │    └── Reduces FALSE POSITIVES (improves precision)
    │
    └──▶ script_3 (hard FALSE NEGATIVE mining) ←── You are here
         └── Reduces MISSED DETECTIONS (improves recall)
```

The two mining scripts work in opposite directions:
- **Script 2**: Make model more conservative on authentic images
- **Script 3**: Make model more aggressive on forgeries

## Typical Hard False Negatives

Analysis shows these forgeries are commonly missed:

| Forgery Type | Why It's Missed |
|--------------|-----------------|
| Tiny spot clones | Below detection threshold |
| Color-matched pastes | No statistical difference |
| Smooth blends | Gradual transitions hide edges |
| In-painting | Professional-level smoothing |
| Same-image clones | Identical source statistics |
| Noisy regions | Noise masks artifacts |

## Usage

### Training
```bash
# Requires script_1 model to exist first
python script_3_hard_fn_mining.py
```

### Prerequisites
- `models/highres_no_ela_best.pth` must exist
- Training images and masks available

### Monitoring
```bash
tail -f logs/train_3_hard_fn.log
```

## Difference from Script 6

| Aspect | Script 3 (Hard FN Mining) | Script 6 (Small Specialist) |
|--------|---------------------------|----------------------------|
| Training Data | All forgeries (weighted) | Only small forgeries (<2%) |
| Focus | Missed detections of any size | Small forgery detection |
| Weight | 5× for hard FN | N/A (filtered dataset) |
| Ensemble Role | Weighted average | Union operation |
| Purpose | General recall improvement | Small forgery specialist |

Script 3 improves recall across all sizes, while Script 6 specifically targets tiny forgeries with a dedicated model.

## Technical Details

### Why Fine-tune from Script 1?

1. **Knowledge transfer**: Base model knows general forgery patterns
2. **Targeted correction**: Only need to fix specific failure modes
3. **Efficiency**: Faster convergence than training from scratch
4. **Stability**: Maintains performance on easy examples

### The 50% Recall Threshold

```python
if recall < 0.5:  # Hard false negative
```

Why 50%?
- **Too low (e.g., 30%)**: Misses moderately failed examples
- **Too high (e.g., 70%)**: Includes too many "normal" failures
- **50%**: Captures truly problematic cases without noise

### Weighted Random Sampler

```python
sampler = WeightedRandomSampler(
    weights=dataset.weights,
    num_samples=len(dataset),
    replacement=True  # Allows repeated sampling
)
```

With `replacement=True`, hard FN can be sampled multiple times per epoch.

## Common Issues

### No Hard FN Found
- Base model may already be very good
- Lower threshold: `recall < 0.7`
- Check mask loading (ensure not all zeros)

### Recall Doesn't Improve
- Increase hard FN weight: `weight=7.0`
- More epochs: `EPOCHS=40`
- Check augmentation isn't destroying forgery evidence

### Precision Drops Too Much
- Reduce hard FN weight: `weight=3.0`
- Use more diverse training data
- Ensemble with script_2 will compensate
