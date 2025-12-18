# Scientific Image Forgery Detection Pipeline

A 6-model ensemble for detecting and segmenting forged regions in scientific images.

## Overview

This pipeline trains 6 specialized models and combines them into an ensemble that achieves:
- **94% detection recall** on forged images
- **0.36 mean IoU** for segmentation
- **89% recall on tiny forgeries** (<0.5% of image area)

## Directory Structure

```
bin_4/
├── run_pipeline.sh                    # Main pipeline script
├── script_1_highres_no_ela.py        # Base model (RGB, 512x512)
├── script_2_hard_negative_mining.py  # Reduces false positives
├── script_3_hard_fn_mining.py        # Improves missed detections
├── script_4_enhanced_augmentation.py # Copy-paste augmentation
├── script_5_comprehensive.py         # RGB+ELA (4 channels)
├── script_6_small_forgery_specialist.py # Small forgery expert
└── script_7_generate_submission.py   # Inference & submission generation
```

## Quick Start

### Full Pipeline (Training + Inference)

```bash
./bin_4/run_pipeline.sh
```

This will:
1. Train all 6 models in parallel (~45-60 minutes each)
2. Generate submission using the trained models

### Inference Only (Skip Training)

If models are already trained:

```bash
./bin_4/run_pipeline.sh --skip-training --test-dir /path/to/test/images
```

## Pipeline Options

```bash
./bin_4/run_pipeline.sh [OPTIONS]

Options:
  --mode MODE         Submission mode (see below)
  --skip-training     Skip training, only run inference
  --test-dir DIR      Directory containing test images
  --output FILE       Output submission file
  -h, --help          Show help
```

## Submission Modes

| Mode | Threshold | Recall | Precision | Use Case |
|------|-----------|--------|-----------|----------|
| `high-precision` | 0.80 | ~50% | ~100% | Minimize false positives |
| `balanced` | 0.70 | ~64% | ~97% | Balanced performance |
| `default` | 0.65 | ~74% | ~95% | Good all-around |
| `high-recall` | 0.60 | ~82% | ~93% | Catch more forgeries |
| `max-recall` | 0.55 | ~88% | ~90% | Maximum detection |

Example:
```bash
./bin_4/run_pipeline.sh --skip-training --mode max-recall
```

## Training Individual Models

Each script can be run independently:

```bash
# Activate environment
conda activate phi4

# Train individual models
python bin_4/script_1_highres_no_ela.py
python bin_4/script_2_hard_negative_mining.py
python bin_4/script_3_hard_fn_mining.py
python bin_4/script_4_enhanced_augmentation.py
python bin_4/script_5_comprehensive.py
python bin_4/script_6_small_forgery_specialist.py
```

### Model Descriptions

| Script | Model | Purpose | Weight |
|--------|-------|---------|--------|
| script_1 | highres_no_ela | Base RGB model at 512x512 | 1.5 |
| script_2 | hard_negative | Trained on false positives | 1.2 |
| script_3 | high_recall | Trained on false negatives | 1.0 |
| script_4 | enhanced_aug | Copy-paste augmentation | 0.8 |
| script_5 | comprehensive | RGB+ELA (4 channels) | 0.5 |
| script_6 | small_specialist | Only small forgeries (<2%) | Union |

## Inference Only

```bash
python bin_4/script_7_generate_submission.py \
    --input /path/to/test/images \
    --output submission.csv \
    --mode default
```

## Output Format

The submission is a CSV file with RLE-encoded masks:

```csv
case_id,annotation
12345,authentic
67890,240654 0 3 1 1597 0 3 1 ...
```

- `authentic` = no forgery detected
- RLE format = run-length encoded binary mask

## Model Architecture

All models use the same architecture:
- **Backbone**: EfficientNet-B2 (ImageNet pretrained)
- **Decoder**: FPN (Feature Pyramid Network)
- **Attention Gate**: Learned attention on decoder output
- **Parameters**: 9.5M
- **Input**: 512×512 (3-ch RGB or 4-ch RGB+ELA)
- **Output**: 512×512 probability map

## Performance by Forgery Size

| Size | % of Dataset | Detection Rate |
|------|--------------|----------------|
| Tiny (<0.5%) | 24% | 87% |
| Small (0.5-2%) | 25% | 95% |
| Medium (2-5%) | 16% | 98% |
| Large (5-10%) | 20% | 98% |
| XLarge (>10%) | 15% | 95% |

## Monitoring Training

```bash
# Watch main pipeline log
tail -f logs/pipeline.log

# Watch individual training logs
tail -f logs/train_1_highres.log
tail -f logs/train_2_hard_neg.log
# etc.
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA GPU with 4GB+ VRAM (per model)
- Conda environment: `phi4`

### Python Packages

```
torch
torchvision
segmentation-models-pytorch
opencv-python
numpy
pandas
albumentations
```

## Troubleshooting

### Kill All Running Scripts

```bash
pkill -f "script_[0-9]_" && pkill -f "run_pipeline.sh"
```

### Check Running Processes

```bash
ps aux | grep "script_" | grep -v grep
```

### Clean Logs

```bash
rm -f logs/pipeline.log logs/train_*.log
```

## Files Generated

After training:
```
models/
├── highres_no_ela_best.pth
├── hard_negative_best.pth
├── high_recall_best.pth
├── enhanced_aug_best.pth
├── comprehensive_best.pth
└── small_forgery_specialist_best.pth
```

After inference:
```
output/
└── submission.csv
```
