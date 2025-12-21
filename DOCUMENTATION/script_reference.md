# Script Reference

Complete documentation for all scripts in the `bin/` directory.

## Table of Contents
- [Pipeline Scripts](#pipeline-scripts)
  - [run_pipeline.sh](#run_pipelinesh)
  - [clean.sh](#cleansh)
- [Training Scripts](#training-scripts)
  - [script_0_train_v3.py](#script_0_train_v3py)
  - [script_1_train_v4.py](#script_1_train_v4py)
  - [script_2_train_binary_classifier.py](#script_2_train_binary_classifierpy)
- [Inference Scripts](#inference-scripts)
  - [script_3_two_stage_submission.py](#script_3_two_stage_submissionpy)
- [Data Generation Scripts](#data-generation-scripts)
  - [generate_forged_augmented.py](#generate_forged_augmentedpy)
  - [generate_authentic_augmented.py](#generate_authentic_augmentedpy)
- [Utility Scripts](#utility-scripts)
  - [validate_test.py](#validate_testpy)

---

## Pipeline Scripts

### run_pipeline.sh

**Purpose**: Main orchestrator that runs the complete training and inference pipeline.

**Location**: `bin/run_pipeline.sh`

**Usage**:
```bash
cd bin
./run_pipeline.sh
```

**Pipeline Steps**:

| Step | Description | Script Called |
|------|-------------|---------------|
| 1 | Generate forged augmented data | `generate_forged_augmented.py` |
| 2 | Generate authentic augmented data | `generate_authentic_augmented.py` |
| 1a | Train V3 base models | `script_0_train_v3.py` |
| 1b | Fine-tune V4 models | `script_1_train_v4.py` |
| 2 | Train binary classifier | `script_2_train_binary_classifier.py` |
| 3 | Generate submission | `script_3_two_stage_submission.py` |

**Outputs**:
- `log/run_pipeline.log` - Combined output log
- `models/*.pth` - Trained model files
- `submission.csv` - Final submission file

**Expected Runtime**: ~9-13 hours total
- V3 training: ~6-8 hours (4 models × 25 epochs)
- V4 training: ~2-3 hours (4 models × 25 epochs)
- Classifier: ~1 hour

---

### clean.sh

**Purpose**: Clean logs and augmented data while preserving directory structure.

**Location**: `bin/clean.sh`

**Usage**:
```bash
cd bin
./clean.sh
```

**Cleans**:
- `bin/logs/*` - Script logs
- `log/*` - Pipeline logs
- `train_images/forged_augmented/*` - Augmented forged images
- `train_images/authentic_augmented/*` - Augmented authentic images
- `train_masks_augmented/*` - Augmented masks

**Does NOT clean**:
- `models/` - Trained models
- Original training data

---

## Training Scripts

### script_0_train_v3.py

**Purpose**: Train V3 base segmentation models from ImageNet-pretrained weights.

**Location**: `bin/script_0_train_v3.py`

**Usage**:
```bash
python script_0_train_v3.py [--model MODEL_NAME]
```

**Arguments**:
| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | all | Train specific model or "all" for all 4 |

**Models Trained**:
- `highres_no_ela_v3_best.pth` - Standard training, no ELA
- `hard_negative_v3_best.pth` - Focus on FP reduction
- `high_recall_v3_best.pth` - Maximize detection
- `enhanced_aug_v3_best.pth` - Heavy augmentation

**Training Data**:
- Original forged images: ~2,751
- Augmented forged images: ~7,564
- Hard negative authentic images: 73 (known FP producers)
- Regular authentic images for balance

**Key Features**:
- Combined loss: Focal + Dice + FP penalty
- Weighted random sampler for class balance
- Hard negative mining from test set
- 25 epochs per model

**Outputs**:
- `models/*_v3_best.pth` - Best checkpoint for each model
- Log output to stdout

---

### script_1_train_v4.py

**Purpose**: Fine-tune V4 models from V3 base, focusing on hard negative samples.

**Location**: `bin/script_1_train_v4.py`

**Usage**:
```bash
python script_1_train_v4.py [--model MODEL_NAME]
```

**Arguments**:
| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | all | Train specific model or "all" for all 4 |

**Dependencies**:
- V3 models must exist (`*_v3_best.pth`)
- If V3 models don't exist, trains from scratch with ImageNet weights

**Training Focus**:
- Load V3 weights as starting point
- Increased FP penalty (4.0 vs 2.0)
- All 677 FP-producing authentic images as hard negatives
- 25 additional epochs

**Hard Negative Mining**:
1. Scans all authentic images
2. Identifies images producing false positives at threshold=0.75
3. Saves list to `/tmp/all_fp_images.txt`
4. Weights these images 5× in training sampler

**Outputs**:
- `models/*_v4_best.pth` - Fine-tuned models

---

### script_2_train_binary_classifier.py

**Purpose**: Train Stage 1 binary classifier (forged vs authentic).

**Location**: `bin/script_2_train_binary_classifier.py`

**Usage**:
```bash
python script_2_train_binary_classifier.py
```

**Architecture**:
```
EfficientNet-B2 (pretrained)
    → Global Average Pooling
    → Linear(1408, 256) + ReLU + Dropout(0.3)
    → Linear(256, 1) + Sigmoid
```

**Training Data**:
- All forged images (original + augmented)
- All authentic images (original + augmented)
- 80/20 train/val split

**Training Details**:
- Input size: 384×384
- Batch size: 16
- 20 epochs
- Binary cross-entropy loss
- AdamW optimizer, LR=1e-4
- Early stopping on validation accuracy

**Outputs**:
- `models/binary_classifier_best.pth`

---

## Inference Scripts

### script_3_two_stage_submission.py

**Purpose**: Generate submission.csv using two-stage detection pipeline.

**Location**: `bin/script_3_two_stage_submission.py`

**Usage**:
```bash
python script_3_two_stage_submission.py [options]
```

**Arguments**:
| Argument | Default | Description |
|----------|---------|-------------|
| `--test_dir` | `../test_images/` | Directory containing test images |
| `--models_dir` | `../models/` | Directory containing trained models |
| `--output` | `../submission.csv` | Output CSV path |
| `--classifier_threshold` | 0.25 | Stage 1 threshold |
| `--seg_threshold` | 0.35 | Segmentation threshold |
| `--min_area` | 300 | Minimum mask area |

**Two-Stage Pipeline**:
1. **Stage 1**: Run binary classifier
   - If probability < classifier_threshold → predict "authentic"
2. **Stage 2**: Run segmentation ensemble
   - 4 V4 models with TTA (8 predictions per image)
   - Average all predictions
   - Apply seg_threshold to get binary mask
   - If mask area < min_area → predict "authentic"
   - Otherwise → encode mask as RLE

**Test-Time Augmentation**:
- Original image
- Horizontal flip
- Vertical flip
- Both flips

**Output Format**:
```csv
id,rle
12345,
12346,123 45 200 30 ...
```
- Empty RLE = authentic
- Non-empty RLE = forged region mask

---

## Data Generation Scripts

### generate_forged_augmented.py

**Purpose**: Generate augmented versions of forged images and masks.

**Location**: `bin/generate_forged_augmented.py`

**Usage**:
```bash
python generate_forged_augmented.py
```

**Augmentations Applied**:
- Horizontal/Vertical flip (50% each)
- Random rotation (0-360°)
- Random scale (80-120%)
- Brightness/Contrast adjustment
- Hue/Saturation shift
- Gaussian blur
- JPEG compression artifacts

**Key Feature**:
- Masks are transformed identically to images
- Uses `ReplayCompose` to sync transforms

**Input**:
- `train_images/forged/` - Original forged images
- `train_masks/` - Original masks

**Output**:
- `train_images/forged_augmented/` - 3× augmented images
- `train_masks_augmented/` - Corresponding masks

**Expected Output**: ~7,500 augmented images (3× original 2,751)

---

### generate_authentic_augmented.py

**Purpose**: Generate augmented versions of authentic images.

**Location**: `bin/generate_authentic_augmented.py`

**Usage**:
```bash
python generate_authentic_augmented.py
```

**Augmentations**: Same as forged augmentation (no masks needed)

**Input**: `train_images/authentic/`

**Output**: `train_images/authentic_augmented/` - 2× augmented images

---

## Utility Scripts

### validate_test.py

**Purpose**: Validate model performance on held-out test sets.

**Location**: `bin/validate_test.py`

**Usage**:
```bash
python validate_test.py [options]
```

**Arguments**:
| Argument | Default | Description |
|----------|---------|-------------|
| `--cls_threshold` | 0.25 | Classifier threshold |
| `--seg_threshold` | 0.35 | Segmentation threshold |
| `--min_area` | 300 | Minimum mask area |

**Test Sets**:
- `test_authentic_100/` - 100 authentic images
- `test_forged_100/` - 100 forged images with ground truth masks

**Metrics Reported**:
- True Positives (TP): Forged images correctly detected
- False Positives (FP): Authentic images incorrectly marked forged
- Recall: TP / total forged images
- FP Rate: FP / total authentic images
- **Net Score**: TP - FP (competition metric)

**Threshold Sweep**:
Can be used to find optimal thresholds by testing multiple combinations.

---

## Script Dependencies

```
┌────────────────────────────────────────────────────────────┐
│                      run_pipeline.sh                        │
└────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐   ┌───────────────────┐   ┌─────────────────┐
│ generate_     │   │ generate_         │   │ script_0_       │
│ forged_       │   │ authentic_        │   │ train_v3.py     │
│ augmented.py  │   │ augmented.py      │   └────────┬────────┘
└───────────────┘   └───────────────────┘            │
                                                     ▼
                                          ┌──────────────────┐
                                          │ script_1_        │
                                          │ train_v4.py      │
                                          └────────┬─────────┘
                                                   │
                    ┌──────────────────────────────┼──────────┐
                    ▼                              ▼          │
          ┌─────────────────────┐   ┌─────────────────────┐  │
          │ script_2_train_     │   │ script_3_two_stage_ │  │
          │ binary_classifier.py│   │ submission.py       │◀─┘
          └─────────────────────┘   └─────────────────────┘
```

---

## Common Issues

### Model Loading Errors

**Problem**: `KeyError: 'base_model.encoder...'`

**Cause**: Model weights were saved with `base.` prefix but code uses `base_model.`

**Solution**: Ensure `AttentionFPN` uses `self.base` not `self.base_model`

### CenterCrop Errors

**Problem**: `CenterCrop: crop size exceeds image dimensions`

**Cause**: Some images are smaller than 480×480

**Solution**: Use `RandomScale` instead of `CenterCrop` in augmentation

### V3 Models Not Found

**Problem**: V4 training fails because V3 models don't exist

**Cause**: Pipeline skipped V3 training

**Solution**: Run `script_0_train_v3.py` first, or V4 will train from scratch

### Low Recall

**Problem**: Recall drops from ~80% to ~50%

**Cause**: Models trained from scratch instead of fine-tuned

**Solution**: Ensure V3 → V4 training flow is followed

---

## Model Files Reference

| File | Size | Description |
|------|------|-------------|
| `highres_no_ela_v3_best.pth` | ~30MB | V3 base model |
| `highres_no_ela_v4_best.pth` | ~30MB | V4 fine-tuned |
| `hard_negative_v3_best.pth` | ~30MB | V3 FP-focused |
| `hard_negative_v4_best.pth` | ~30MB | V4 fine-tuned |
| `high_recall_v3_best.pth` | ~30MB | V3 recall-focused |
| `high_recall_v4_best.pth` | ~30MB | V4 fine-tuned |
| `enhanced_aug_v3_best.pth` | ~30MB | V3 augmentation-robust |
| `enhanced_aug_v4_best.pth` | ~30MB | V4 fine-tuned |
| `binary_classifier_best.pth` | ~15MB | Stage 1 classifier |
