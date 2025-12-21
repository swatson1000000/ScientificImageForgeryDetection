# Quick Start Guide

## Running the Pipeline

### Prerequisites

1. Activate the conda environment:
   ```bash
   source ~/miniconda3/etc/profile.d/conda.sh
   conda activate phi4
   ```

2. Ensure training data exists:
   - `train_images/forged/` - Forged images with manipulations
   - `train_images/authentic/` - Authentic unmodified images
   - `train_masks/` - Ground truth masks (numpy .npy files)

### Full Training Pipeline

Run the complete pipeline (augmentation → training → inference):

```bash
cd bin/
./run_pipeline.sh
```

This will:
1. Generate augmented data (if not exists)
2. Train V3 segmentation models (~4-6 hours)
3. Fine-tune V4 segmentation models (~4-6 hours)
4. Train binary classifier (~30 minutes)
5. Generate submission file

### Pipeline Options

```bash
# Skip training, only run inference (if models exist)
./run_pipeline.sh --skip-training

# Custom thresholds
./run_pipeline.sh --classifier-threshold 0.25 --seg-threshold 0.35

# View all options
./run_pipeline.sh --help
```

### Running in Background

For long training runs:

```bash
nohup ./run_pipeline.sh > ../log/run_pipeline.log 2>&1 &

# Monitor progress
tail -f ../log/run_pipeline.log
```

### Cleanup

Remove generated data and logs:

```bash
./clean.sh
```

This removes:
- All files in `log/` and `bin/logs/`
- All files in `train_images/forged_augmented/`
- All files in `train_images/authentic_augmented/`

Note: Directories are preserved, only contents are deleted.

## Validation

Test model performance on a sample:

```bash
python validate_test.py --classifier-threshold 0.25 --seg-threshold 0.35 --min-area 300
```

This runs inference on images in `test_forged_100/` and `test_authentic_100/` and reports TP, FP, TN, FN, and Net Score.

## Output Files

| File | Description |
|------|-------------|
| `submission.csv` | Kaggle submission file |
| `log/run_pipeline.log` | Main pipeline log |
| `log/train_v3.log` | V3 training log |
| `log/train_v4.log` | V4 training log |
| `log/train_classifier.log` | Classifier training log |
| `models/*_v3_best.pth` | V3 model checkpoints |
| `models/*_v4_best.pth` | V4 model checkpoints |
| `models/binary_classifier_best.pth` | Classifier checkpoint |

## Typical Training Times (RTX 4090)

| Step | Duration |
|------|----------|
| Data Augmentation | 10-20 min |
| V3 Training (4 models) | 4-6 hours |
| V4 Training (4 models) | 4-6 hours |
| Classifier Training | 20-30 min |
| Inference (1000 images) | 5-10 min |
| **Total** | **9-13 hours** |
