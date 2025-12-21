# System Overview

## Two-Stage Detection Pipeline

The system uses a **two-stage approach** to detect forged regions in scientific images:

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Image     │────▶│  Stage 1:        │────▶│  Stage 2:       │
│   Input     │     │  Binary Classifier│     │  Segmentation   │
└─────────────┘     └──────────────────┘     └─────────────────┘
                           │                        │
                           ▼                        ▼
                    prob < threshold?         mask area < min?
                           │                        │
                           ▼                        ▼
                      "authentic"              "authentic"
                                                    │
                                                    ▼
                                              "forged" + RLE mask
```

### Stage 1: Binary Classification

A fast classifier (EfficientNet-B2) predicts whether an image is likely forged.
- **Input**: 384×384 RGB image
- **Output**: Probability of forgery [0, 1]
- **Threshold**: If prob < 0.25, image is classified as authentic (no segmentation needed)
- **Purpose**: Filter out obvious authentic images to reduce false positives

### Stage 2: Segmentation Ensemble

For images passing Stage 1, a 4-model ensemble predicts the forged region mask.
- **Input**: 512×512 RGB image
- **Output**: Binary mask indicating forged pixels
- **Models**: 4 AttentionFPN models with different training emphases
- **Aggregation**: Mean of all model predictions
- **Threshold**: Pixels with prob > 0.35 marked as forged
- **Min Area**: Masks with <300 pixels are rejected as authentic

## Model Architecture

### Segmentation Models (V3/V4)

```
┌────────────────────────────────────────────────────┐
│                 AttentionFPN                        │
├────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────┐  │
│  │  FPN (Feature Pyramid Network)               │  │
│  │  - Encoder: timm-efficientnet-b2             │  │
│  │  - Decoder: FPN with skip connections        │  │
│  │  - Output: 1-channel segmentation mask       │  │
│  └──────────────────────────────────────────────┘  │
│                        │                            │
│                        ▼                            │
│  ┌──────────────────────────────────────────────┐  │
│  │  Attention Gate                              │  │
│  │  - Conv 1×1 → BN → ReLU                      │  │
│  │  - Conv 3×3 → BN → ReLU                      │  │
│  │  - Conv 1×1 → Sigmoid                        │  │
│  │  - Element-wise multiply with input          │  │
│  └──────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────┘
```

### 4-Model Ensemble

| Model | Focus | Training Emphasis |
|-------|-------|-------------------|
| `highres_no_ela` | High resolution details | Standard training, no ELA features |
| `hard_negative` | Reduce false positives | Heavy weighting on FP-producing authentic images |
| `high_recall` | Maximize detection | Lower thresholds, balanced loss |
| `enhanced_aug` | Robustness | Heavy augmentation during training |

### Binary Classifier

```
┌────────────────────────────────────────────────────┐
│              ForgeryClassifier                      │
├────────────────────────────────────────────────────┤
│  EfficientNet-B2 (pretrained ImageNet)             │
│           │                                         │
│           ▼                                         │
│  Global Average Pooling                             │
│           │                                         │
│           ▼                                         │
│  Linear(1408 → 256) → ReLU → Dropout(0.3)          │
│           │                                         │
│           ▼                                         │
│  Linear(256 → 1) → Sigmoid                         │
└────────────────────────────────────────────────────┘
```

## Training Pipeline

### Phase 1: Data Augmentation

1. **Forged Augmentation**: Generate 3× augmented versions of each forged image
   - Geometric: Flip, rotate, scale
   - Color: Brightness, contrast, hue
   - Noise: Gaussian blur, compression artifacts
   - Masks transformed identically to images

2. **Authentic Augmentation**: Generate 2× augmented versions of authentic images
   - Similar transforms (no masks needed)
   - Increases negative sample diversity

### Phase 2: V3 Training (Base Models)

Train 4 segmentation models from ImageNet-pretrained weights:
- 25 epochs per model
- Mixed loss: Focal + Dice + FP penalty
- Hard negative mining: Weight FP-producing authentic images 5×
- Outputs: `*_v3_best.pth` model files

### Phase 3: V4 Training (Fine-tuning)

Fine-tune V3 models on hard negative samples:
- Load V3 weights as starting point
- 25 additional epochs
- Increased FP penalty (4.0)
- Focus on reducing false positives
- Outputs: `*_v4_best.pth` model files

### Phase 4: Classifier Training

Train binary classifier:
- 20 epochs
- Binary cross-entropy loss
- Early stopping on validation accuracy
- Output: `binary_classifier_best.pth`

## Loss Functions

### Combined Segmentation Loss

```python
Loss = α × Focal_Loss + β × Dice_Loss + γ × FP_Penalty

Where:
- Focal_Loss: Focuses on hard examples, reduces class imbalance
- Dice_Loss: Overlap-based, good for segmentation
- FP_Penalty: Extra penalty for false positive predictions on authentic images
- α = 0.5, β = 0.5, γ = 2.0-4.0
```

### Hard Negative Mining

Images that produce false positives are identified and weighted heavily during training:
1. Run inference on all authentic images
2. Identify images with mask area > threshold
3. Weight these images 5× in training sampler
4. This forces the model to learn authentic image characteristics

## Inference Pipeline

1. Load binary classifier and 4 segmentation models
2. For each test image:
   - Run classifier → if prob < threshold, output "authentic"
   - Run 4 segmentation models with TTA (Test-Time Augmentation)
   - Average predictions across models
   - Apply threshold and minimum area filter
   - Convert mask to RLE encoding
3. Save results to submission.csv

### Test-Time Augmentation (TTA)

For each image, run inference on 4 variants:
- Original
- Horizontal flip
- Vertical flip
- Both flips

Average all predictions for more robust results.

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `classifier_threshold` | 0.25 | Min probability to pass to segmentation |
| `seg_threshold` | 0.35 | Min probability to mark pixel as forged |
| `min_area` | 300 | Min pixels to classify as forged |
| `IMG_SIZE` | 512 | Segmentation input size |
| `CLASSIFIER_SIZE` | 384 | Classifier input size |
| `BATCH_SIZE` | 4 | Training batch size |
| `EPOCHS` | 25 | Training epochs per model |
| `LR` | 1e-4 | Learning rate |
