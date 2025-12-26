# Scientific Image Forgery Detection - Documentation Index

## Quick Navigation

This documentation set provides comprehensive coverage of the Scientific Image Forgery Detection project.

### Start Here

1. **[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)** - Start here!
   - Project summary and key features
   - Architecture overview (two-stage pipeline)
   - Training strategy and performance metrics
   - Data organization
   - Requirements and hyperparameter tuning

2. **[USAGE_GUIDE.md](USAGE_GUIDE.md)** - How to run everything
   - Quick start workflow
   - Execution guidelines (best practices)
   - **Script 1**: Base model training (v3)
   - **Script 2**: Binary classifier training
   - **Script 3**: Hard negative fine-tuning (v4)
   - **Script 4**: Inference and predictions
   - Example workflows and troubleshooting

3. **[DETAILED_EXPLANATIONS.md](DETAILED_EXPLANATIONS.md)** - Deep technical dives
   - Complete code walkthroughs for all scripts
   - Architecture details (FPN, Attention, Loss functions)
   - **Supplementary Scripts**:
     - Generate augmented forged images
     - Generate augmented authentic images
     - Hard negative mining
     - Test validation with parameter sweep
     - Full pipeline orchestration
   - Mathematical foundations
   - Performance analysis

---

## Documentation by Topic

### Getting Started
- Read: [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) → Overview section
- Read: [USAGE_GUIDE.md](USAGE_GUIDE.md) → Quick Start

### Running Training Scripts
- Script 1 (v3): [USAGE_GUIDE.md#script-1](USAGE_GUIDE.md#script-1-base-model-training-v3) + [DETAILED_EXPLANATIONS.md#script-1](DETAILED_EXPLANATIONS.md#script-1-detailed-breakdown)
- Script 2 (Classifier): [USAGE_GUIDE.md#script-2](USAGE_GUIDE.md#script-2-binary-classifier-training) + [DETAILED_EXPLANATIONS.md#script-2](DETAILED_EXPLANATIONS.md#script-2-detailed-breakdown)
- Script 3 (v4): [USAGE_GUIDE.md#script-3](USAGE_GUIDE.md#script-3-hard-negative-fine-tuning-v4) + [DETAILED_EXPLANATIONS.md#script-3](DETAILED_EXPLANATIONS.md#script-3-detailed-breakdown)

### Running Inference
- Script 4 Pipeline: [USAGE_GUIDE.md#script-4](USAGE_GUIDE.md#script-4-inference-pipeline-two-pass) + [DETAILED_EXPLANATIONS.md#script-4](DETAILED_EXPLANATIONS.md#script-4-detailed-breakdown)

### Data Augmentation
- Forged images: [DETAILED_EXPLANATIONS.md#generate-forged-augmented](DETAILED_EXPLANATIONS.md#generate-forged-augmented-images)
- Authentic images: [DETAILED_EXPLANATIONS.md#generate-authentic-augmented](DETAILED_EXPLANATIONS.md#generate-authentic-augmented-images)

### Hard Negative Mining
- Overview: [PROJECT_OVERVIEW.md#hard-negative-mining-strategy](PROJECT_OVERVIEW.md#hard-negative-mining-strategy)
- Find hard negatives: [DETAILED_EXPLANATIONS.md#find-hard-negatives](DETAILED_EXPLANATIONS.md#find-hard-negatives)
- Full pipeline: [DETAILED_EXPLANATIONS.md#full-pipeline-orchestration](DETAILED_EXPLANATIONS.md#full-pipeline-orchestration)

### Architecture & Algorithms
- Two-pass pipeline: [PROJECT_OVERVIEW.md#architecture-overview](PROJECT_OVERVIEW.md#architecture-overview)
- Loss functions: [DETAILED_EXPLANATIONS.md#loss-functions](DETAILED_EXPLANATIONS.md#loss-functions)
- Model architectures: [DETAILED_EXPLANATIONS.md#model-architectures](DETAILED_EXPLANATIONS.md#model-architectures)
- FPN: [DETAILED_EXPLANATIONS.md#fpn-feature-pyramid-network](DETAILED_EXPLANATIONS.md#fpn-feature-pyramid-network)

### Troubleshooting & Optimization
- Training issues: [USAGE_GUIDE.md#troubleshooting](USAGE_GUIDE.md#troubleshooting)
- Hyperparameter tuning: [PROJECT_OVERVIEW.md#hyperparameter-tuning](PROJECT_OVERVIEW.md#hyperparameter-tuning)
- Performance tuning: [USAGE_GUIDE.md#performance-tuning](USAGE_GUIDE.md#performance-tuning)

---

## Script Reference

| Script | File | Purpose | Time | Read |
|--------|------|---------|------|------|
| 1 | `bin/script_1_train_v3.py` | Train base segmentation models | 4-6h | [USAGE](USAGE_GUIDE.md#script-1-base-model-training-v3) / [DETAIL](DETAILED_EXPLANATIONS.md#script-1-detailed-breakdown) |
| 2 | `bin/script_2_train_binary_classifier.py` | Train classifier for fast filtering | 2-3h | [USAGE](USAGE_GUIDE.md#script-2-binary-classifier-training) / [DETAIL](DETAILED_EXPLANATIONS.md#script-2-detailed-breakdown) |
| 3 | `bin/script_3_train_v4.py` | Fine-tune with hard negatives | 2-4h | [USAGE](USAGE_GUIDE.md#script-3-hard-negative-fine-tuning-v4) / [DETAIL](DETAILED_EXPLANATIONS.md#script-3-detailed-breakdown) |
| 4 | `bin/script_4_two_pass_pipeline.py` | Inference and predictions | 300-600ms/img | [USAGE](USAGE_GUIDE.md#script-4-inference-pipeline-two-pass) / [DETAIL](DETAILED_EXPLANATIONS.md#script-4-detailed-breakdown) |
| - | `bin/generate_forged_augmented.py` | Create augmented forged training data | ~5min | [DETAIL](DETAILED_EXPLANATIONS.md#generate-forged-augmented-images) |
| - | `bin/generate_authentic_augmented.py` | Create augmented authentic training data | ~10min | [DETAIL](DETAILED_EXPLANATIONS.md#generate-authentic-augmented-images) |
| - | `bin/find_hard_negatives.py` | Identify false positives for mining | ~10min | [DETAIL](DETAILED_EXPLANATIONS.md#find-hard-negatives) |
| - | `bin/validate_test.py` | Validate on test sets | ~5min | [DETAIL](DETAILED_EXPLANATIONS.md#validate-test) |
| - | `bin/full_pipeline.py` | Orchestrate complete workflow | 10-15h | [DETAIL](DETAILED_EXPLANATIONS.md#full-pipeline-orchestration) |

---

## Key Concepts

### Two-Pass Pipeline
**What**: Fast classification → detailed segmentation

```
Pass 1: Binary Classifier (10ms)
├─ Input: 384×384 image
├─ Output: Probability (0.0-1.0)
└─ Decision: If prob < 0.25 → AUTHENTIC (stop)
            If prob ≥ 0.25 → Continue to Pass 2

Pass 2: Ensemble Segmentation (600ms)
├─ 4 models + 4× TTA
├─ Adaptive thresholding
├─ Component filtering
└─ Output: Binary mask of forged regions
```

**Why**: 50% of images stop at Pass 1 (fast), 50% do full segmentation (accurate)

### Hard Negative Mining
**What**: Find authentic images the model incorrectly detects as forged

```
Authentic Image → Model → Predicts "FORGED" (False Positive)
                            ↓
                     This is a "Hard Negative"
                            ↓
                  Retrain model on it (5x weight)
                            ↓
                  Model learns to reject it
```

**Impact**: Reduces FP rate from 8% → 3.4%

### Weighted Sampling
**What**: Oversample difficult cases during training

```
Dataset: 10k forged + 12k authentic + 73 hard negatives

Weights:
├─ Small forgeries (< 1%): 3.0×
├─ Medium forgeries (1-5%): 2.0×
├─ Large forgeries (> 5%): 1.0×
└─ Hard negatives: 5.0×

Effect: Important cases appear more often in each epoch
```

### Test-Time Augmentation (TTA)
**What**: Run inference on 4 different orientations, take MAX

```
Original → Predict
H-flip   → Predict → MAX
V-flip   → Predict
Both     → Predict
```

**Result**: More robust, catches forgeries from any orientation

---

## Performance Metrics

### Best Configuration
- **Classifier Threshold**: 0.25
- **Segmentation Threshold**: 0.35 (adaptive)
- **Min Area**: 300 pixels
- **Recall**: 79%
- **FP Rate**: 3.4%

### Timing (on typical NVIDIA GPU)
- Inference: 300-600ms per image (varies by classifier decision)
- Training (v3): 4-6 hours for 4 models
- Training (v4): 2-4 hours for 4 models with fine-tuning
- Hard negative mining: 5-15 minutes on 2,377 images

---

## Common Workflows

### Complete Training from Scratch
```bash
cd /home/swatson/work/MachineLearning/kaggle/ScientificImageForgeryDetection
python bin/full_pipeline.py
```

### Just Run Inference
```bash
python bin/script_4_two_pass_pipeline.py --input test_images/ --output results/
```

### Evaluate on Test Set
```bash
python bin/validate_test.py --sweep
```

### Find Problematic Images
```bash
python bin/find_hard_negatives.py
cat hard_negative_ids.txt  # See which images fool the model
```

---

## Additional Resources

- **CLAUDE.md**: Critical inference preprocessing requirements
- **Project README**: (if exists) Project-level overview
- **Code Comments**: Inline documentation in each script
- **Git History**: Review commits for design decisions

---

**Last Updated**: December 2025  
**Status**: Complete documentation  
**Total Pages**: 4 markdown files  
**Total Lines**: ~5,000+ lines of detailed explanations
