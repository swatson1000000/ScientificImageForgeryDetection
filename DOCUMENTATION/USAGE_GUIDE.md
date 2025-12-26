# Usage Guide - How to Run Scripts

## Quick Start

All scripts are located in the `bin/` directory and use absolute paths, so they can be run from any directory.

### Recommended Workflow

```bash
# 1. Start in project root
cd /home/swatson/work/MachineLearning/kaggle/ScientificImageForgeryDetection

# 2. Train base models (v3) - runs all 4 models in parallel
nohup python bin/script_1_train_v3.py > log/script_1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 3. Train binary classifier (while v3 trains)
nohup python bin/script_2_train_binary_classifier.py > log/script_2_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 4. After v3 completes: Fine-tune with hard negatives (v4)
nohup python bin/script_3_train_v4.py > log/script_3_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 5. Run inference on test images
python bin/script_4_two_pass_pipeline.py --input test_images/ --output results/
```

## Script Execution Guide

### General Rules

⚠️ **IMPORTANT**:
- Always use `nohup` and background execution for training (use `&` at end)
- Use `> logfile.log 2>&1` to capture output to file
- Check logs with: `tail -f log/logfile.log`
- Run scripts from project root or any directory (paths are absolute)
- Do NOT run from `bin/` directory

### Example: Proper Execution

```bash
# ✅ CORRECT: Background with logging
nohup python bin/script_1_train_v3.py > log/training_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# ❌ WRONG: Blocking foreground
python bin/script_1_train_v3.py

# ❌ WRONG: No logging
nohup python bin/script_1_train_v3.py &
```

### Checking Progress

```bash
# Monitor log file (last 50 lines, updates in real-time)
tail -f log/script_1_20250101_120000.log

# Count lines (to see training progress)
wc -l log/script_1_20250101_120000.log

# Search for specific patterns
grep "Epoch\|best_loss\|Saved" log/script_1_20250101_120000.log

# Stop monitoring (press Ctrl+C)
```

---

## Script 1: Base Model Training (v3)

**File**: `bin/script_1_train_v3.py`  
**Time**: ~4-6 hours  
**GPUs**: 1 NVIDIA GPU (8GB+ VRAM)

### Purpose
Trains 4 base segmentation models using:
- 2,751 original forged images + 7,564 augmented = ~10,315 forged
- ~12,200 authentic images (balanced sampling)
- 73 hard negative authentic images

### Command

```bash
# Basic: Train all 4 models
nohup python bin/script_1_train_v3.py > log/v3_training.log 2>&1 &

# Train specific model
nohup python bin/script_1_train_v3.py --model highres_no_ela > log/v3_highres.log 2>&1 &

# Skip hard negatives (first phase)
nohup python bin/script_1_train_v3.py --no-hard-negatives > log/v3_no_hn.log 2>&1 &
```

### Arguments

```
--model {all, highres_no_ela, hard_negative, high_recall, enhanced_aug}
    Which model to train. Default: all

--no-hard-negatives
    Skip hard negatives during training (useful for initial experiments)
```

### Output

```
models/
├── highres_no_ela_v3_best.pth       # Best model for this variant
├── hard_negative_v3_best.pth
├── high_recall_v3_best.pth
└── enhanced_aug_v3_best.pth
```

### What It Does

1. **Data Collection**:
   - Loads forged images from `train_images/forged` + augmented versions
   - Loads corresponding segmentation masks
   - Loads authentic images from `train_images/authentic` + augmented versions
   - Loads hard negatives (if `hard_negative_ids.txt` exists)

2. **Dataset Creation**:
   - Creates 3 separate datasets: forged, authentic, hard negative
   - Applies augmentations: random resize/crop, flips, rotations, color jitter, blur, noise
   - Combines into single dataset with weighted sampling

3. **Model Architecture**:
   - Base: FPN with EfficientNet B2 encoder (ImageNet pretrained)
   - Head: Attention Gate focusing on forgery regions
   - Output: Single-channel mask (512×512 resolution)

4. **Loss Function**:
   - Focal Loss: Focuses on hard examples, prevents easy negatives from dominating
   - Dice Loss: Overlap-based loss, good for segmentation
   - FP Penalty: Extra penalty for false positives on authentic images
   - Combined weight: 0.5 Focal + 0.5 Dice + 2.0 FP penalty

5. **Training**:
   - 25 epochs with batch size 4
   - Learning rate: 1e-4 with cosine annealing scheduler
   - Weighted sampling to oversample: hard negatives (5x), small forgeries (3x)
   - Saves best model based on validation loss

6. **Hardware**:
   - Device: CUDA GPU if available, else CPU
   - Memory: ~6GB VRAM per batch
   - Time per epoch: ~5-10 minutes (GPU dependent)

### Model Variants Explained

| Model | Focus | Augmentation | Use Case |
|-------|-------|--------------|----------|
| `highres_no_ela` | High-resolution details | Standard | General purpose |
| `hard_negative` | Rejecting tricky authentic | High overlap sampling | Reduce false positives |
| `high_recall` | Maximize sensitivity | Conservative | Find all forgeries |
| `enhanced_aug` | Robustness to variations | Heavy augmentation | Diverse image types |

### Troubleshooting

**Issue**: Out of memory
```bash
# Reduce batch size by modifying script
BATCH_SIZE = 2  # From 4
```

**Issue**: Slow training
```bash
# Check GPU usage
nvidia-smi
# If GPU util < 50%, may be I/O bound
# Try reducing num_workers: num_workers=0 in DataLoader
```

**Issue**: Loss not decreasing
```bash
# Check learning rate
LR = 1e-5  # Try smaller initial LR

# Check data loading
print(f"Loaded {len(forged_images)} forged images")  # Should be ~10k
```

---

## Script 2: Binary Classifier Training

**File**: `bin/script_2_train_binary_classifier.py`  
**Time**: ~2-3 hours  
**GPUs**: 1 NVIDIA GPU (8GB+ VRAM)

### Purpose
Trains binary classifier (forged vs. authentic) for Pass 1 filtering:
- Reduces computation by skipping segmentation for clear authentic images
- Improves precision by filtering likely true negatives early

### Command

```bash
# Basic: Train with default hard negative weighting
nohup python bin/script_2_train_binary_classifier.py > log/classifier.log 2>&1 &

# Disable hard negatives
nohup python bin/script_2_train_binary_classifier.py --no-hard-negatives > log/classifier_no_hn.log 2>&1 &

# Adjust hard negative weight (default 3.0)
nohup python bin/script_2_train_binary_classifier.py --hard-negative-weight 5.0 > log/classifier_hn5.log 2>&1 &

# Change number of epochs
nohup python bin/script_2_train_binary_classifier.py --epochs 30 > log/classifier_e30.log 2>&1 &
```

### Arguments

```
--no-hard-negatives
    Don't load hard negatives, train on regular forged/authentic only

--hard-negative-weight FLOAT
    Weight multiplier for hard negatives in sampling. Default: 3.0
    Higher = more emphasis on hard cases

--epochs INT
    Number of training epochs. Default: 20
```

### Output

```
models/
└── binary_classifier_best.pth      # Best classifier checkpoint
```

### What It Does

1. **Data Loading**:
   - Forged: All from `train_images/forged` + augmented versions
   - Authentic: All from `train_images/authentic` + augmented versions
   - Hard negatives: Loaded from `test_authentic_100/` (if `hard_negative_ids.txt` exists)

2. **Dataset Creation**:
   - Creates BinaryClassificationDataset with:
     - Regular samples: (image, label) pairs
     - Hard negatives: Marked as authentic (0) but flagged as hard
   - Applies transforms: 384×384 resize, flips, rotations, color augmentation
   - ImageNet normalization

3. **Model Architecture**:
   - Backbone: EfficientNet B2 (ImageNet pretrained)
   - Head: Global average pool → FC(256) → ReLU → Dropout(0.3) → FC(1)
   - Output: Binary logit (sigmoid for probability)

4. **Training**:
   - 20 epochs with batch size 16
   - Learning rate: 1e-4 with cosine annealing
   - Loss: BCE with logits
   - Validation split: 90% train, 10% val
   - Weighted sampling:
     - Regular samples: weight 1.0
     - Hard negatives: weight 3.0 (default, oversampled)
   - Tracks: Accuracy, Precision, Recall, TP/FP/TN/FN

5. **Metrics Reported**:
   - Training accuracy and loss per epoch
   - Validation metrics: Accuracy, Precision, Recall
   - Confusion matrix: TP, FP, TN, FN

### Understanding the Classifier

**Threshold Interpretation**:
- Probability ≥ 0.25: Likely forged → Pass to segmentation
- Probability < 0.25: Likely authentic → Return "authentic" (skip Pass 2)

**Why Two-Stage?**:
- Classifier is fast (10ms) but less precise
- Segmentation is slow (600ms) but highly precise
- Combination: Fast rejection of obvious authentic + precise detection for ambiguous

### Troubleshooting

**Issue**: Class imbalance problems
```bash
# Check label distribution in dataset
print(f"Forged: {len(train_forged)}")
print(f"Authentic: {len(train_authentic)}")
print(f"Hard negatives: {len(train_hard_negatives)}")

# If very imbalanced, increase hard_negative_weight
--hard-negative-weight 5.0
```

**Issue**: Overfitting (val acc plateaus)
```bash
# Increase dropout
# Decrease LR
LR = 1e-5

# Increase augmentation
# Reduce training epochs
--epochs 15
```

---

## Script 3: Hard Negative Fine-Tuning (v4)

**File**: `bin/script_3_train_v4.py`  
**Time**: ~2-4 hours per model  
**GPUs**: 1 NVIDIA GPU (8GB+ VRAM)  
**Prerequisites**: Script 1 must complete first (v3 models required)

### Purpose
Fine-tunes v3 models with hard negative images to significantly reduce false positives:
- Takes v3 models as starting points
- Trains on ALL authentic images that produce false positives
- Increases FP penalty (2.0 → 4.0)
- Creates v4 models with improved precision

### Key Insight

Hard negatives are authentic images where the v3 model makes false positives (detects forgery when there is none). By including these with high sample weight, the model learns to properly reject them.

### Command

```bash
# Train all 4 models (v3 → v4)
nohup python bin/script_3_train_v4.py > log/v4_training.log 2>&1 &

# Train specific models
nohup python bin/script_3_train_v4.py --models highres_no_ela hard_negative > log/v4_specific.log 2>&1 &
```

### Arguments

```
--models MODEL1 MODEL2 ...
    Models to train. Options: highres_no_ela, hard_negative, high_recall, enhanced_aug
    Default: all models
```

### Hard Negative Data

The script expects hard negative images in: `train_images/authentic/`

Hard negatives should be listed in a file (e.g., `/tmp/all_fp_images.txt`) with one image ID per line:

```
img_001
img_234
img_567
...
```

### Output

```
models/
├── highres_no_ela_v4_best.pth       # Fine-tuned model
├── hard_negative_v4_best.pth
├── high_recall_v4_best.pth
└── enhanced_aug_v4_best.pth
```

### What It Does

1. **Model Loading**:
   - Loads pre-trained v3 models (requires `*_v3_best.pth` files)
   - If v3 not found, initializes fresh model with ImageNet weights

2. **Hard Negative Loading**:
   - Reads hard negative IDs from file
   - Finds corresponding images in `train_images/authentic/`
   - Supports `.png`, `.jpg`, `.jpeg`, `.tif` extensions

3. **Dataset Creation**:
   - HardNegativeDatasetV4 combines:
     - Forged images: weighted by forgery size (large: 1.0, small: 3.0)
     - Hard negative authentic images: weight 5.0 (high priority)
   - Applies augmentations: random resize/crop, flips, rotations, blur, color jitter
   - Uses WeightedRandomSampler to oversample hard cases

4. **Loss Function (V4)**:
   - Enhanced FocalLoss with gamma=2.0
   - Dice Loss for segmentation accuracy
   - **FP Penalty**: Increased from 2.0 to 4.0
     - Penalizes any positive prediction on authentic images
     - Much stronger signal to reject non-forgeries
   - Combined: 0.5 Focal + 0.5 Dice + 4.0 FP Penalty

5. **Training**:
   - 25 epochs with batch size 4
   - Learning rate: 1e-4 (lower than v3)
   - Starts from v3 weights (transfer learning)
   - Saves best model based on loss

6. **Expected Improvements**:
   - False positive rate: Reduced significantly
   - Recall: Maintained or slightly improved
   - Overall precision: Substantially better

### When to Use v4

- **Use v4 models** in production (recommended default)
- **Use v3 models** if:
  - Hard negative data not available
  - You want maximum recall over precision
  - Baseline comparison needed

### Troubleshooting

**Issue**: Base model file not found
```bash
# Error: "Base model not found"
# Solution: Ensure script_1_train_v3.py completed successfully
# Check for *_v3_best.pth files in models/ directory
ls -lah models/*_v3_*.pth
```

**Issue**: Hard negative file not found
```bash
# Error: "Hard negative file not found"
# Create a hard_negative_ids.txt file:
echo "img_001" > hard_negative_ids.txt
echo "img_234" >> hard_negative_ids.txt
# Or run v4 without hard negatives (it will work with just forged/authentic)
```

**Issue**: Memory issues
```bash
# Reduce batch size or hard negative weight
# Edit script directly:
BATCH_SIZE = 2  # From 4
hard_neg_weight = 3.0  # From 5.0
```

---

## Script 4: Inference Pipeline (Two-Pass)

**File**: `bin/script_4_two_pass_pipeline.py`  
**Time**: ~300-600ms per image (depends on classifier threshold)  
**GPUs**: 1 NVIDIA GPU (4GB+ VRAM, optional but recommended)

### Purpose
Runs complete inference on test images:
- Pass 1: Fast binary classification
- Pass 2: Ensemble segmentation with TTA on suspected forgeries
- Outputs: Predictions, masks, overlays, CSV/JSON results

### Command

```bash
# Process single image
python bin/script_4_two_pass_pipeline.py --input image.png --output results/

# Process directory
python bin/script_4_two_pass_pipeline.py --input test_images/ --output results/

# Save CSV submission format
python bin/script_4_two_pass_pipeline.py --input test_images/ --output results/ --csv submission.csv

# Save JSON results
python bin/script_4_two_pass_pipeline.py --input test_images/ --output results/ --json results.json

# Adjust thresholds for higher precision
python bin/script_4_two_pass_pipeline.py --input test_images/ \
    --classifier-threshold 0.5 --seg-threshold 0.45 --min-area 500

# Disable test-time augmentation (faster but less accurate)
python bin/script_4_two_pass_pipeline.py --input test_images/ --no-tta

# Minimal output
python bin/script_4_two_pass_pipeline.py --input test_images/ --quiet
```

### Arguments

```
REQUIRED:
--input, -i PATH
    Input image file or directory containing images

OUTPUT:
--output, -o PATH
    Output directory for results (optional)
--csv PATH
    Save results to CSV file in submission format
--json PATH
    Save results to JSON file

THRESHOLDS:
--classifier-threshold FLOAT
    Binary classifier threshold (default: 0.25)
    Higher = fewer images pass to segmentation

--seg-threshold FLOAT
    Segmentation confidence threshold (default: 0.35)
    Higher = fewer pixels marked as forgery

--min-area INT
    Minimum connected component size in pixels (default: 300)
    Higher = removes smaller false positives

PROCESSING:
--no-tta
    Disable test-time augmentation (4x flips)
    Faster (~2x) but less accurate

--no-masks
    Don't save binary masks

--no-overlays
    Don't save visualization overlays

--quiet, -q
    Minimal output/logging
```

### Input/Output

**Input Formats**:
- Single file: `.png`, `.jpg`, `.jpeg`, `.tif`, `.bmp`
- Directory: All supported image formats in specified folder

**Output Structure**:
```
results/
├── masks/
│   ├── image_1_mask.png         # Binary mask (0=authentic, 255=forged)
│   ├── image_2_mask.png
│   └── ...
├── overlays/
│   ├── image_1_overlay.png      # Original + red overlays of forgery regions
│   ├── image_2_overlay.png
│   └── ...
├── results.csv                   # Submission format (filename, annotation)
└── results.json                  # Detailed results with scores
```

### CSV Output Format

For Kaggle submissions:
```
filename,annotation
image_001.png,authentic
image_002.png,<RLE-encoded mask>
image_003.png,authentic
```

Where `<RLE-encoded mask>` is run-length encoding of the binary forgery mask.

### JSON Output Format

```json
{
  "timestamp": "2025-01-01T12:00:00",
  "config": {
    "classifier_threshold": 0.25,
    "seg_threshold": 0.35,
    "min_area": 300,
    "tta": true
  },
  "results": [
    {
      "filename": "image_001.png",
      "is_forged": false,
      "classifier_prob": 0.15,
      "passed_to_segmentation": false,
      "seg_max_prob": 0.0,
      "forgery_area": 0
    },
    {
      "filename": "image_002.png",
      "is_forged": true,
      "classifier_prob": 0.85,
      "passed_to_segmentation": true,
      "seg_max_prob": 0.92,
      "forgery_area": 2450
    }
  ]
}
```

### Understanding the Output

**Classifier Probability**:
- Likelihood image contains forgery (Pass 1)
- 0.0 = definitely authentic
- 1.0 = definitely forged
- Decision threshold: 0.25 (default)

**Segmentation Max Probability**:
- Maximum pixel-level confidence in forgery regions
- Only available if passed to Pass 2
- 0.0-1.0 range

**Forgery Area**:
- Number of pixels marked as forged
- After applying threshold and min_area filter
- Useful for ranking confidence

**Mask Format**:
- Values: 0 (authentic) or 1 (forged)
- Size: Same as original image after resize
- Use for visualization or further analysis

### Performance Tuning

**For Higher Precision** (fewer false positives):
```bash
python bin/script_4_two_pass_pipeline.py \
    --input test_images/ \
    --classifier-threshold 0.4 \
    --seg-threshold 0.45 \
    --min-area 500
```

**For Higher Recall** (fewer missed forgeries):
```bash
python bin/script_4_two_pass_pipeline.py \
    --input test_images/ \
    --classifier-threshold 0.15 \
    --seg-threshold 0.30 \
    --min-area 100
```

**For Speed** (sacrifice accuracy for faster inference):
```bash
python bin/script_4_two_pass_pipeline.py \
    --input test_images/ \
    --no-tta                    # Skip augmentations
    --classifier-threshold 0.3  # More aggressive filtering
    --min-area 400              # More aggressive noise removal
```

### Example Workflow

```bash
# 1. Process all test images
python bin/script_4_two_pass_pipeline.py \
    --input test_images/ \
    --output results/ \
    --csv submission.csv \
    --json results.json

# 2. Review results
cat results.json | head -50

# 3. Check specific image
python bin/script_4_two_pass_pipeline.py \
    --input test_images/suspicious_image.png \
    --output results/detailed/ \
    --json detailed_result.json

# 4. If too many false positives, adjust:
python bin/script_4_two_pass_pipeline.py \
    --input test_images/ \
    --output results_v2/ \
    --classifier-threshold 0.35 \
    --seg-threshold 0.40 \
    --min-area 400

# 5. Compare results
diff <(jq '.results | length' results/results.json) \
     <(jq '.results | length' results_v2/results.json)
```

### Troubleshooting

**Issue**: "No images found"
```bash
# Check file format
ls test_images/ | head
# Ensure files have correct extensions: .png, .jpg, .jpeg, .tif, .bmp
```

**Issue**: "No segmentation models found"
```bash
# Ensure models are trained
ls models/*_v4_best.pth
# If missing, run script_3_train_v4.py first
```

**Issue**: GPU out of memory
```bash
# Run on CPU (slower but works)
# Script detects GPU automatically, runs on CPU if needed
# Or reduce batch processing if applicable
```

**Issue**: Inference too slow
```bash
# Disable TTA
--no-tta
# Disable saving overlays
--no-overlays
# Process fewer images at once
```

### Example: Batch Processing with Different Thresholds

```bash
#!/bin/bash
# Create multiple submissions with different thresholds

for hn_weight in 0.1 0.2 0.3 0.35 0.4 0.45 0.5; do
    echo "Running with seg_threshold=$hn_weight"
    python bin/script_4_two_pass_pipeline.py \
        --input test_images/ \
        --output results_seg${hn_weight}/ \
        --csv submission_seg${hn_weight}.csv \
        --seg-threshold $hn_weight
done

# Compare results
for f in submission_seg*.csv; do
    echo "$f: $(wc -l < $f) images"
done
```

---

## Advanced Topics

### Running Multiple Scripts in Parallel

```bash
# Start all training scripts at once
cd /home/swatson/work/MachineLearning/kaggle/ScientificImageForgeryDetection

nohup python bin/script_1_train_v3.py > log/s1.log 2>&1 &
nohup python bin/script_2_train_binary_classifier.py > log/s2.log 2>&1 &

# Wait for script 1 to finish, then run script 3
# (In practice, monitor logs)
```

### Monitoring GPU Usage

```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Or periodic snapshots
nvidia-smi -l 5

# Get GPU memory info
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
```

### Collecting Hard Negatives

Hard negatives are authentic images that produce false positives. To create the `hard_negative_ids.txt` file:

```bash
# 1. Run inference on authentic test images
python bin/script_4_two_pass_pipeline.py \
    --input test_authentic_100/ \
    --output inference_results/ \
    --json results.json

# 2. Extract images marked as forged (false positives)
python << 'EOF'
import json
with open('inference_results/results.json') as f:
    data = json.load(f)
    for result in data['results']:
        if result['is_forged']:
            # This is a false positive (authentic marked as forged)
            filename = result['filename'].rsplit('.', 1)[0]
            print(filename)
EOF

# 3. Save to hard_negative_ids.txt
# (redirect above output to file)
```

---

## Summary

| Script | Purpose | Time | Inputs | Outputs |
|--------|---------|------|--------|---------|
| 1 | Train v3 base models | 4-6h | train_images + train_masks | v3_*.pth |
| 2 | Train classifier | 2-3h | train_images | binary_classifier_best.pth |
| 3 | Fine-tune v4 (hard negatives) | 2-4h | v3_*.pth + hard negatives | v4_*.pth |
| 4 | Inference/predictions | 300-600ms/img | Test images | masks, overlays, CSV/JSON |

---

**Last Updated**: December 2025  
**Status**: Production-ready
