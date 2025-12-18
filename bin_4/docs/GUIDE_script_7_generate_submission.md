# Script 7: Generate Submission

## Overview

`script_7_generate_submission.py` is the **inference script** that combines all trained models into a 6-model ensemble, processes test images, and generates the final submission file in RLE (Run-Length Encoding) format.

## Purpose

The primary goals are:
- Load and combine all 6 trained models
- Process test images with weighted ensemble predictions
- Generate binary masks using configurable thresholds
- Encode masks as RLE for submission
- Support multiple precision/recall trade-off modes

## How It Works

### Ensemble Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                       6-MODEL ENSEMBLE                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │              WEIGHTED AVERAGE (5 models)                     │  │
│  │                                                              │  │
│  │  highres_no_ela ──┐                                         │  │
│  │    (weight 1.5)   │                                         │  │
│  │                   │                                         │  │
│  │  hard_negative ───┼──▶ Σ(pred × weight) / Σ(weight) ──┐    │  │
│  │    (weight 1.2)   │                                    │    │  │
│  │                   │                                    │    │  │
│  │  high_recall ─────┤                                    │    │  │
│  │    (weight 1.0)   │                                    │    │  │
│  │                   │                                    │    │  │
│  │  enhanced_aug ────┤                                    │    │  │
│  │    (weight 0.8)   │                                    │    │  │
│  │                   │                                    │    │  │
│  │  comprehensive ───┘                                    │    │  │
│  │    (weight 0.5)                                        │    │  │
│  └────────────────────────────────────────────────────────│────┘  │
│                                                           │        │
│                                                           ▼        │
│  ┌────────────────────────────────────────────────┐  ┌────────┐   │
│  │        small_forgery_specialist                │  │  MAX   │   │
│  │        (union strategy × 0.8)                  │──│        │   │
│  └────────────────────────────────────────────────┘  └───┬────┘   │
│                                                           │        │
│                                                           ▼        │
│                                                    Final Prediction │
└─────────────────────────────────────────────────────────────────────┘
```

### Prediction Pipeline

```python
def predict(self, image_bgr):
    """Get ensemble prediction for an image."""
    predictions = []
    
    # 1. Get predictions from 5 main models
    for model, cfg in self.models:
        if cfg['in_channels'] == 3:
            input_tensor = prepare_image_rgb(image_bgr)
        else:
            input_tensor = prepare_image_ela(image_bgr)
        
        with torch.no_grad():
            pred = torch.sigmoid(model(input_tensor))
        
        predictions.append((pred, cfg['weight']))
    
    # 2. Weighted average of main ensemble
    result = np.zeros_like(predictions[0][0])
    for pred, weight in predictions:
        result += pred * weight
    result /= self.total_weight  # Normalize
    
    # 3. UNION with small forgery specialist
    if self.small_specialist is not None:
        small_pred = self.small_specialist(input_tensor)
        result = np.maximum(result, small_pred * 0.8)
    
    return result
```

## Model Loading

```python
class SubmissionGenerator:
    def __init__(self, model_dir=None):
        # 5 main models with weights
        self.model_configs = [
            {'name': 'highres_no_ela', 'path': 'highres_no_ela_best.pth', 
             'in_channels': 3, 'weight': 1.5},
            {'name': 'hard_negative', 'path': 'hard_negative_best.pth', 
             'in_channels': 3, 'weight': 1.2},
            {'name': 'high_recall', 'path': 'high_recall_best.pth', 
             'in_channels': 3, 'weight': 1.0},
            {'name': 'enhanced_aug', 'path': 'enhanced_aug_best.pth', 
             'in_channels': 3, 'weight': 0.8},
            {'name': 'comprehensive', 'path': 'comprehensive_best.pth', 
             'in_channels': 4, 'weight': 0.5},  # Note: 4 channels!
        ]
        
        # Small specialist uses union strategy (not weighted average)
        self.small_specialist = load_model('small_forgery_specialist_best.pth', 
                                            in_channels=3)
```

## Submission Modes

| Mode | Threshold | Min Area | Recall | Precision | Use Case |
|------|-----------|----------|--------|-----------|----------|
| `high-precision` | 0.80 | 100px | ~50% | ~100% | Minimize false positives |
| `balanced` | 0.70 | 50px | ~64% | ~97% | Balanced performance |
| `default` | 0.65 | 50px | ~74% | ~95% | Good all-around |
| `high-recall` | 0.60 | 30px | ~82% | ~93% | Catch more forgeries |
| `max-recall` | 0.55 | 20px | ~88% | ~90% | Maximum detection |

```python
MODES = {
    'high-precision': {'threshold': 0.80, 'min_area': 100},
    'balanced': {'threshold': 0.70, 'min_area': 50},
    'default': {'threshold': 0.65, 'min_area': 50},
    'high-recall': {'threshold': 0.60, 'min_area': 30},
    'max-recall': {'threshold': 0.55, 'min_area': 20},
}
```

## RLE Encoding

### What is RLE?

Run-Length Encoding compresses binary masks by storing run lengths:

```
Binary mask (flattened):
[0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0]

RLE encoding:
"3 0 4 1 4 0 2 1 1 0"
(3 zeros, 4 ones, 4 zeros, 2 ones, 1 zero)
```

### Implementation

```python
def mask_to_rle(mask):
    """Convert binary mask to RLE string."""
    flat = mask.flatten()  # Row-major order
    
    runs = []
    current_val = flat[0]
    count = 1
    
    for i in range(1, len(flat)):
        if flat[i] == current_val:
            count += 1
        else:
            runs.append(f"{count} {int(current_val)}")
            current_val = flat[i]
            count = 1
    
    runs.append(f"{count} {int(current_val)}")
    
    return " ".join(runs)
```

### Verification

```python
def rle_to_mask(rle_string, shape):
    """Convert RLE string back to binary mask."""
    if rle_string == "authentic":
        return np.zeros(shape, dtype=np.uint8)
    
    parts = rle_string.split()
    flat = []
    
    for i in range(0, len(parts), 2):
        count = int(parts[i])
        val = int(parts[i + 1])
        flat.extend([val] * count)
    
    return np.array(flat, dtype=np.uint8).reshape(shape)
```

## Mask Generation

```python
def generate_mask(self, image_path, threshold=0.65, min_area=50):
    """Generate binary mask for an image."""
    image = cv2.imread(str(image_path))
    orig_h, orig_w = image.shape[:2]
    
    # Get prediction (512×512)
    pred = self.predict(image)
    
    # Threshold
    binary = (pred > threshold).astype(np.uint8)
    
    # Remove small regions
    if min_area > 0:
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        binary = np.zeros_like(binary)
        for contour in contours:
            if cv2.contourArea(contour) >= min_area:
                cv2.drawContours(binary, [contour], -1, 1, -1)
    
    # Resize to original resolution
    mask = cv2.resize(binary, (orig_w, orig_h), 
                      interpolation=cv2.INTER_NEAREST)
    
    is_forged = mask.max() > 0
    
    return mask, is_forged
```

## Output Format

### CSV Structure

```csv
case_id,annotation
12345,authentic
67890,240654 0 3 1 1597 0 3 1 850 0 1 1 ...
```

- **case_id**: Image filename without extension
- **annotation**: Either `authentic` or RLE-encoded mask

### File Locations

```
output/
└── submission.csv    # Generated submission

models/
├── highres_no_ela_best.pth
├── hard_negative_best.pth
├── high_recall_best.pth
├── enhanced_aug_best.pth
├── comprehensive_best.pth
└── small_forgery_specialist_best.pth
```

## Usage

### Basic Usage

```bash
python script_7_generate_submission.py \
    --input /path/to/test/images \
    --output submission.csv
```

### With Mode

```bash
python script_7_generate_submission.py \
    --input test_images/ \
    --output submission.csv \
    --mode max-recall
```

### Custom Threshold

```bash
python script_7_generate_submission.py \
    --input test_images/ \
    --output submission.csv \
    --threshold 0.55 \
    --min-area 25
```

### All Options

```
Options:
  --input, -i      Input directory containing test images (required)
  --output, -o     Output CSV file path (required)
  --mode, -m       Preset mode: high-precision, balanced, default, 
                   high-recall, max-recall
  --threshold, -t  Detection threshold (default: 0.65)
  --min-area       Minimum forgery area in pixels (default: 50)
  --model-dir      Directory containing model files
```

## Processing Flow

```
┌─────────────────────────────────────────────────────────┐
│ 1. Load All 6 Models                                    │
│    └── Check for each model file                        │
│    └── Load into GPU                                    │
│    └── Print loading status                             │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ 2. For Each Test Image:                                 │
│    └── Read image (PNG/JPG/TIFF)                       │
│    └── Prepare inputs (RGB + ELA)                      │
│    └── Get predictions from all models                 │
│    └── Combine with weighted average + union           │
│    └── Threshold and filter                            │
│    └── Resize to original resolution                   │
│    └── Encode as RLE                                   │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ 3. Generate CSV                                         │
│    └── Collect all case_id + annotation pairs           │
│    └── Write to output file                             │
│    └── Print summary statistics                         │
└─────────────────────────────────────────────────────────┘
```

## Performance Considerations

### GPU Memory

- 6 models × ~38MB each = ~230MB GPU memory
- Plus input/output tensors: ~50MB
- Total: ~300MB GPU memory

### Speed

| Operation | Time per Image |
|-----------|----------------|
| Image loading | ~10ms |
| ELA computation | ~6ms |
| Model inference (×6) | ~50ms |
| Post-processing | ~5ms |
| **Total** | **~70ms** |

For 1000 images: ~70 seconds

### Batch Processing

Currently processes one image at a time. For faster processing:

```python
# Future optimization: batch processing
for batch in batch_iterator(images, batch_size=4):
    predictions = model(batch)  # Process 4 at once
```

## Model Weights Explained

### Why These Weights?

| Model | Weight | Rationale |
|-------|--------|-----------|
| highres_no_ela | **1.5** | Best overall, most reliable |
| hard_negative | **1.2** | High precision, reduces FP |
| high_recall | **1.0** | Balanced recall/precision |
| enhanced_aug | **0.8** | Good generalization |
| comprehensive | **0.5** | ELA can be unreliable |
| small_specialist | **Union** | Catch small forgeries |

### Weight Normalization

```python
# Total weight: 1.5 + 1.2 + 1.0 + 0.8 + 0.5 = 5.0
total_weight = sum(cfg['weight'] for _, cfg in models)

# Weighted average
result = sum(pred * weight for pred, weight in predictions) / total_weight
```

## Threshold Selection Guide

### Trade-off Visualization

```
                    Recall
                      │
                100%  │     ╭──────────────────╮
                      │    ╱                    ╲
                 80%  │   ╱   max-recall (0.55) ─╲─
                      │  ╱    high-recall (0.60)  ╲
                 60%  │ ╱     default (0.65)       ╲
                      │╱      balanced (0.70)       ╲
                 40%  │       high-precision (0.80)  ╲
                      │                               ╲
                 20%  │                                ╲
                      │                                 ╲
                  0%  └────────────────────────────────────
                      0%         50%        100%
                                Precision
```

### Selection Criteria

| If you want... | Use mode |
|----------------|----------|
| Zero false positives | `high-precision` |
| Balanced performance | `balanced` |
| Good default | `default` |
| Catch more forgeries | `high-recall` |
| Maximum detection | `max-recall` |

## Minimum Area Filtering

```python
# Remove small noise/artifacts
if cv2.contourArea(contour) >= min_area:
    cv2.drawContours(binary, [contour], -1, 1, -1)
```

| min_area | Effect |
|----------|--------|
| 100 | Very aggressive filtering, only large detections |
| 50 | Standard, removes most noise |
| 30 | Preserves smaller detections |
| 20 | Minimal filtering |
| 0 | No filtering (may include single-pixel noise) |

## Common Issues

### Missing Models
```
✗ highres_no_ela not found: models/highres_no_ela_best.pth
```
- Run the corresponding training script first
- Check model directory path

### Out of Memory
- Process images sequentially (default)
- Reduce batch size if modified
- Use CPU: `DEVICE = torch.device('cpu')`

### Slow Processing
- Ensure GPU is being used
- Check that models are on GPU
- Consider batch processing

### RLE Mismatch
- Ensure masks are binary (0/1)
- Check row-major flattening
- Verify resize uses INTER_NEAREST

## Verification

### Check RLE Correctness

```python
# After generating submission
mask, is_forged = generator.generate_mask(image_path)
rle = mask_to_rle(mask)
reconstructed = rle_to_mask(rle, mask.shape)

assert np.array_equal(mask, reconstructed), "RLE mismatch!"
```

### Visual Verification

```python
import matplotlib.pyplot as plt

image = cv2.imread('test_image.png')
mask, _ = generator.generate_mask('test_image.png')

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original')
plt.subplot(1, 2, 2)
plt.imshow(mask, cmap='hot')
plt.title('Predicted Mask')
plt.savefig('verification.png')
```

## Integration with Pipeline

The pipeline script (`run_pipeline.sh`) calls this script automatically:

```bash
python script_7_generate_submission.py \
    --input "$TEST_DIR" \
    --output "$OUTPUT_FILE" \
    --mode "$MODE"
```

All models must be trained before running inference.
