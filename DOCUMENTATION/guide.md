# Scientific Image Forgery Detection - Complete Guide

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Running the Pipeline](#running-the-pipeline)
4. [General Concepts](#general-concepts)
5. [Script Reference](#script-reference)
6. [Configuration](#configuration)
7. [Troubleshooting](#troubleshooting)

---

## Overview

This project implements a **two-stage pipeline** for detecting manipulated regions in scientific images:

1. **Stage 1 - Binary Classifier**: Fast filtering to identify potentially forged images
2. **Stage 2 - Segmentation Ensemble**: Detailed mask generation for filtered images

### Architecture Summary

| Component | Model | Input Size | Purpose |
|-----------|-------|------------|---------|
| Classifier | EfficientNet-B2 + MLP | 384×384 | Binary: forged (1) vs authentic (0) |
| Segmentation | AttentionFPN × 4 ensemble | 512×512 | Pixel-wise forgery mask |

### Performance (Validated)

| Metric | Value |
|--------|-------|
| Net Score (TP - FP) | 2029 |
| True Positives | 2162 |
| False Positives | 133 |
| Recall | 78.6% |
| FP Rate | 5.6% |

---

## Quick Start

```bash
# Navigate to project
cd /path/to/ScientificImageForgeryDetection

# Full pipeline (train + inference)
./bin_4/run_pipeline.sh

# Inference only (skip training)
./bin_4/run_pipeline.sh --skip-training
```

---

## Running the Pipeline

### Prerequisites

1. **Conda Environment**: `phi4` with PyTorch, timm, segmentation-models-pytorch
2. **GPU**: CUDA-capable GPU recommended (tested on NVIDIA GPUs)
3. **Data**:
   - Training images in `train_images/forged/` and `train_images/authentic/`
   - Training masks in `train_masks/`
   - Test images in `validation_images/`

### Pipeline Script: `run_pipeline.sh`

The main pipeline script orchestrates the entire workflow.

#### Basic Usage

```bash
# Full pipeline (trains models + generates submission)
./bin_4/run_pipeline.sh

# Skip training (use existing models)
./bin_4/run_pipeline.sh --skip-training

# Custom thresholds
./bin_4/run_pipeline.sh --classifier-threshold 0.30 --seg-threshold 0.40

# Custom input/output
./bin_4/run_pipeline.sh --test-dir /path/to/images --output /path/to/submission.csv
```

#### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--skip-training` | false | Skip model training, run inference only |
| `--classifier-threshold T` | 0.25 | Binary classifier threshold (lower = more images to segmentation) |
| `--seg-threshold T` | 0.35 | Segmentation threshold for mask generation |
| `--test-dir DIR` | `validation_images/` | Directory containing test images |
| `--output FILE` | `output/submission.csv` | Output submission file |
| `-h, --help` | - | Show help message |

#### Pipeline Steps

The script executes three steps:

**Step 1: Train V4 Segmentation Models** (if not skipped)
- Trains 4 model variants with hard negative mining
- Output: 4 `.pth` files in `models/`
- Duration: ~4-6 hours on GPU

**Step 2: Train Binary Classifier** (if not skipped)
- Trains EfficientNet-B2 based classifier
- Output: `binary_classifier_best.pth`
- Duration: ~30-60 minutes

**Step 3: Generate Submission**
- Runs two-stage inference on test images
- Output: `submission.csv`
- Duration: ~3-5 minutes for 934 images

#### Output Files

```
output/
└── submission.csv          # Final submission file

models/
├── binary_classifier_best.pth
├── highres_no_ela_v4_best.pth
├── hard_negative_v4_best.pth
├── high_recall_v4_best.pth
└── enhanced_aug_v4_best.pth

bin_4/logs/
├── train_v4.log
├── train_classifier.log
└── submission.log
```

#### Running in Background

```bash
# Run in background with nohup
nohup bash bin_4/run_pipeline.sh --skip-training > log/run_pipeline.log 2>&1 &

# Check progress
tail -f log/run_pipeline.log

# Check if complete
cat log/run_pipeline.log | tail -50
```

---

## General Concepts

### Two-Stage Architecture

The pipeline uses a cascade approach to optimize the speed/accuracy trade-off:

```
Input Image
    │
    ▼
┌─────────────────────────────────┐
│  Stage 1: Binary Classifier     │
│  (EfficientNet-B2, 384×384)     │
│  Fast: ~10ms per image          │
└─────────────────────────────────┘
    │
    ├── prob < 0.25 ──────────────────► "authentic" (skip segmentation)
    │
    ▼ prob ≥ 0.25
┌─────────────────────────────────┐
│  Stage 2: Segmentation Ensemble │
│  (4× AttentionFPN, 512×512)     │
│  Slower: ~200ms per image       │
└─────────────────────────────────┘
    │
    ├── mask empty ───────────────────► "authentic"
    │
    ▼ mask has content
    RLE-encoded mask
```

### Model Architecture

#### Binary Classifier (`ForgeryClassifier`)

```
Input (3, 384, 384)
    │
    ▼
EfficientNet-B2 Backbone (pretrained)
    │ → 1408 features
    ▼
Linear(1408, 256) + ReLU + Dropout(0.3)
    │
    ▼
Linear(256, 1) → sigmoid → probability
```

#### Segmentation Model (`AttentionFPN`)

```
Input (3, 512, 512)
    │
    ▼
FPN with EfficientNet-B2 encoder
    │ → (1, 512, 512) raw logits
    ▼
AttentionGate
    │ → refined predictions
    ▼
Sigmoid → probability mask
```

The AttentionGate refines predictions by learning spatial attention:
```
Conv 1×1 → BN → ReLU → Conv 3×3 → BN → ReLU → Conv 1×1 → Sigmoid
```

### Ensemble Strategy

Four model variants are trained with different strategies:

| Model | Training Focus |
|-------|----------------|
| `highres_no_ela` | High-resolution RGB, no ELA features |
| `hard_negative` | Heavy focus on false-positive reduction |
| `high_recall` | Optimized for detecting subtle forgeries |
| `enhanced_aug` | Strong augmentation for generalization |

**Aggregation**: Mean of all 4 models after TTA

### Test-Time Augmentation (TTA)

4× TTA with flip augmentations:
1. Original
2. Horizontal flip
3. Vertical flip
4. Both flips

Predictions are averaged (mean aggregation) for stability.

### Adaptive Thresholding

The segmentation threshold adapts to image brightness:

| Brightness | Threshold Adjustment |
|------------|---------------------|
| < 50 (dark) | -0.10 (lower threshold) |
| 50-80 | -0.05 |
| 80-200 | No change |
| > 200 (bright) | +0.05 (higher threshold) |

### Loss Functions

#### Segmentation Loss (`CombinedLossV4`)

```
Loss = 0.5 × FocalLoss + 0.5 × DiceLoss + 4.0 × FP_Penalty
```

- **Focal Loss**: Handles class imbalance (α=0.25, γ=2.0)
- **Dice Loss**: Ensures spatial overlap
- **FP Penalty**: Extra penalty for false predictions on authentic images

#### Classification Loss

Standard Binary Cross-Entropy with Logits (BCEWithLogitsLoss)

### Data Handling

#### Training Data Structure

```
train_images/
├── forged/          # 2751 forged images
│   ├── 10001.png
│   └── ...
└── authentic/       # 2377 authentic images
    ├── 20001.png
    └── ...

train_masks/
├── 10001.npy        # Binary mask for each forged image
└── ...
```

#### Hard Negative Mining

V4 models use hard negative mining:
- 677 authentic images that produce false positives at t=0.75
- These images are weighted 5× in training
- Significantly reduces false positive rate

### RLE Encoding

Submission uses Run-Length Encoding for masks:

```
"count1 value1 count2 value2 ..."

Example: "100 0 50 1 200 0 30 1"
= 100 zeros, 50 ones, 200 zeros, 30 ones
```

---

## Script Reference

### script_1_train_v4.py

**Purpose**: Train 4 V4 segmentation model variants with hard negative mining.

#### Usage

```bash
python bin_4/script_1_train_v4.py

# Train specific models only
python bin_4/script_1_train_v4.py --models highres_no_ela hard_negative
```

#### Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `IMG_SIZE` | 512 | Input image size |
| `BATCH_SIZE` | 4 | Batch size |
| `EPOCHS` | 25 | Training epochs |
| `LR` | 1e-4 | Learning rate |

#### Input Requirements

- Pre-trained V3 models in `models/`:
  - `highres_no_ela_v3_best.pth`
  - `hard_negative_v3_best.pth`
  - `high_recall_v3_best.pth`
  - `enhanced_aug_v3_best.pth`
- Hard negative file at `/tmp/all_fp_images.txt`
- Training images in `train_images/`
- Training masks in `train_masks/`

#### Output

- 4 model files in `models/`:
  - `highres_no_ela_v4_best.pth`
  - `hard_negative_v4_best.pth`
  - `high_recall_v4_best.pth`
  - `enhanced_aug_v4_best.pth`

#### Training Strategy

1. Loads pre-trained V3 model
2. Creates dataset with forged images + hard negatives (5× weight)
3. Uses weighted random sampling
4. Trains with CombinedLossV4 (strong FP penalty)
5. Saves best model by validation loss

---

### script_2_train_binary_classifier.py

**Purpose**: Train binary classifier for Stage 1 filtering.

#### Usage

```bash
python bin_4/script_2_train_binary_classifier.py
```

#### Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `IMG_SIZE` | 384 | Input image size |
| `BATCH_SIZE` | 16 | Batch size |
| `EPOCHS` | 20 | Training epochs |
| `LR` | 1e-4 | Learning rate |

#### Input Requirements

- Training images in `train_images/forged/` and `train_images/authentic/`

#### Output

- `models/binary_classifier_best.pth`

#### Training Strategy

1. Loads all forged and authentic images
2. 90/10 train/val split (stratified)
3. Standard augmentations (flips, color jitter)
4. BCEWithLogitsLoss
5. Saves best model by validation accuracy

#### Metrics Tracked

- Accuracy
- Precision
- Recall
- TP/FP/TN/FN counts

---

### script_3_two_stage_submission.py

**Purpose**: Generate submission CSV using two-stage pipeline.

#### Usage

```bash
# Basic usage
python bin_4/script_3_two_stage_submission.py --input test_images/ --output submission.csv

# With custom thresholds
python bin_4/script_3_two_stage_submission.py \
    --input test_images/ \
    --output submission.csv \
    --classifier-threshold 0.25 \
    --seg-threshold 0.35 \
    --min-area 300

# Disable TTA for faster inference
python bin_4/script_3_two_stage_submission.py \
    --input test_images/ \
    --output submission.csv \
    --no-tta
```

#### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--input, -i` | (required) | Input directory with test images |
| `--output, -o` | `submission.csv` | Output CSV file |
| `--classifier-threshold, -c` | 0.25 | Classifier filtering threshold |
| `--seg-threshold, -s` | 0.35 | Segmentation threshold |
| `--min-area` | 300 | Minimum forgery area (pixels) |
| `--no-tta` | false | Disable test-time augmentation |
| `--no-adaptive` | false | Disable adaptive thresholding |
| `--model-dir` | `models/` | Directory containing model files |

#### Input Requirements

- Test images in input directory (PNG, JPG, TIFF supported)
- Trained models in `models/`:
  - `binary_classifier_best.pth`
  - 4× V4 ensemble models

#### Output

CSV file with columns:
- `case_id`: Image filename (without extension)
- `annotation`: Either `"authentic"` or RLE-encoded mask

#### Processing Flow

For each image:
1. **Classify**: Run through binary classifier
2. **Filter**: If prob < threshold → "authentic" (skip segmentation)
3. **Segment**: Run through 4-model ensemble with TTA
4. **Threshold**: Apply adaptive threshold
5. **Filter regions**: Remove small connected components
6. **Encode**: Convert mask to RLE

---

## Configuration

### Optimal Parameters (Validated)

| Parameter | Optimal Value | Notes |
|-----------|---------------|-------|
| Classifier Threshold | 0.25 | Lower = more images to segmentation |
| Seg Threshold | 0.35 | Base threshold before adaptive adjustment |
| Min Area | 300 | Pixels; removes noise |
| TTA | Enabled | 4× flip augmentations |
| Adaptive Threshold | Enabled | Adjusts for image brightness |

### Threshold Tuning Guidelines

**Classifier Threshold** (0.0 - 1.0):
- Lower values → more images pass to segmentation → higher recall, more FP
- Higher values → fewer images pass → lower recall, fewer FP
- Sweet spot: 0.20-0.30

**Segmentation Threshold** (0.0 - 1.0):
- Lower values → more pixels classified as forged → larger masks
- Higher values → fewer pixels → smaller, more precise masks
- Sweet spot: 0.30-0.40

**Min Area** (pixels):
- Higher values → more aggressive noise filtering
- Lower values → keep smaller detections
- Sweet spot: 200-500

---

## Troubleshooting

### Common Issues

#### CUDA Out of Memory

```bash
# Reduce batch size in scripts
# script_1: BATCH_SIZE = 2
# script_2: BATCH_SIZE = 8
```

#### Missing Models

```
Error: Model not found: models/highres_no_ela_v4_best.pth
```

Solution: Run training first or ensure models are in `models/` directory.

#### Hard Negative File Not Found

```
Error: Hard negative file not found: /tmp/all_fp_images.txt
```

Solution: Generate hard negatives by running validation sweep, or create empty file for initial training.

#### Permission Denied

```
./run_pipeline.sh: Permission denied
```

Solution:
```bash
chmod +x bin_4/run_pipeline.sh
# or
bash bin_4/run_pipeline.sh
```

### Logging

All scripts log to `bin_4/logs/`:
- `train_v4.log` - Segmentation training
- `train_classifier.log` - Classifier training
- `submission.log` - Inference

### Verifying Installation

```bash
# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate phi4

# Check dependencies
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import timm; print(f'timm: {timm.__version__}')"
python -c "import segmentation_models_pytorch as smp; print(f'smp: {smp.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
```

---

## File Structure

```
ScientificImageForgeryDetection/
├── bin_4/                          # Main scripts
│   ├── run_pipeline.sh             # Pipeline orchestrator
│   ├── script_1_train_v4.py        # Train segmentation models
│   ├── script_2_train_binary_classifier.py  # Train classifier
│   ├── script_3_two_stage_submission.py     # Generate submission
│   ├── validate_two_stage.py       # Validation utilities
│   └── logs/                       # Training/inference logs
├── models/                         # Trained model weights
├── train_images/                   # Training data
│   ├── forged/
│   └── authentic/
├── train_masks/                    # Ground truth masks
├── validation_images/              # Test images
├── output/                         # Submission files
├── DOCUMENTATION/                  # This documentation
└── kaggle_notebook_submission.ipynb  # Kaggle notebook for submission
```
