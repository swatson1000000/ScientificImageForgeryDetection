# Instructions for Claude AI Assistant

## ⚠️ CRITICAL: Inference Preprocessing Requirements

When creating ANY new inference or validation scripts, **ALWAYS** include these three critical components:

### 1. ImageNet Normalization (REQUIRED)
```python
def preprocess_for_segmentation(image_bgr, device):
    img_resized = cv2.resize(image_bgr, (512, 512))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    # CRITICAL: ImageNet normalization - models trained with this!
    img_norm = (img_norm - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    return tensor
```

### 2. Adaptive Threshold (REQUIRED)
```python
def get_adaptive_threshold(img_bgr, base_threshold=0.35):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)  # 0-255 scale
    if brightness < 50:
        return max(0.20, base_threshold - 0.10)
    elif brightness < 80:
        return max(0.25, base_threshold - 0.05)
    elif brightness > 200:
        return min(0.50, base_threshold + 0.05)
    return base_threshold
```

### 3. Connected Components Filtering (REQUIRED)
```python
# After thresholding, remove small noise regions
if min_area > 0:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            binary_mask[labels == i] = 0
is_forged = binary_mask.max() > 0
```

### Best Configuration (79% recall, 3.4% FP rate)
| Parameter | Value |
|-----------|-------|
| classifier_threshold | 0.25 |
| seg_threshold | 0.35 |
| min_area | 300 |
| TTA | 4x flips with MAX aggregation |

---

## Important: Running Scripts

### Script Location
**All training and inference scripts are in the `bin/` directory:**
- Location: `/home/swatson/work/MachineLearning/kaggle/ScientificImageForgeryDetection/bin/`
- Scripts: `script_1_training_focal_loss.py` through `script_8_deep_ensemble.py` and `script_9_ensemble_inference.py`
- Pipeline: `run_full_pipeline.sh` (orchestrates all training scripts in parallel)

### Always use nohup and background execution
When running training or inference scripts, **always** use `nohup` and run in the background:

```bash
nohup python bin/script_name.py > log/script_name_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

**Do NOT run scripts in foreground** unless explicitly requested. This allows:
- User to continue working
- Training to continue if connection drops
- Easy monitoring via log files
- Prevention of terminal blocking

### Important: Absolute Paths
All scripts use absolute home_dir path:
```python
home_dir = '/home/swatson/work/MachineLearning/kaggle/ScientificImageForgeryDetection'
```

Scripts can be run from ANY directory and will work correctly with absolute paths.

### Examples

```bash
# ✅ CORRECT: Run in background with nohup from project root
nohup python bin/script_1_training_focal_loss.py > log/training_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# ✅ CORRECT: Run full pipeline (orchestrates all 8 scripts in parallel)
nohup bash bin/run_full_pipeline.sh > log/pipeline_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# ❌ WRONG: Blocking foreground execution
python bin/script_1_training_focal_loss.py

# ❌ WRONG: Without logging
nohup python bin/script_1_training_focal_loss.py &

# ❌ WRONG: Running from bin/ directory (breaks relative paths)
cd bin && nohup python script_1_training_focal_loss.py &
```

### Monitoring Running Scripts

```bash
# Check if script is running
ps aux | grep script_name | grep -v grep

# Monitor log in real-time
tail -f log/script_name_*.log

# Check all background jobs
jobs -l
```

---

## Documentation Organization

### All documentation goes in DOCUMENTATION directory
- Location: `/DOCUMENTATION/`
- Keep all markdown files (.md) in this directory
- Do NOT create .md files in the root directory

### Current Documentation
- `LABEL_CORRECTION_PLAN.md` - Label swap and retraining strategy
- `DIRECTORY_STRUCTURE.md` - Project directory layout
- `BIN_STRUCTURE.md` - Details on bin/ directory scripts
- `CLAUDE.md` - This file (instructions for AI assistant)

### When Creating New Documentation
1. Create file in `DOCUMENTATION/` directory
2. Use descriptive filenames (e.g., `RETRAINING_RESULTS.md`, `PERFORMANCE_ANALYSIS.md`)
3. Reference from this file if important
4. Do NOT clutter root directory

---

## Project Status (As of Dec 10, 2025)

### Current Architecture: 7-Model Ensemble (bin_4/)
Located in `bin_4/` directory with improved pipeline:
- **5 Weighted Models**: highres_no_ela (1.5), hard_negative (1.2), high_recall (1.0), enhanced_aug (0.8), comprehensive (0.5)
- **1 Small Forgery Specialist**: OR-combined to catch small forgeries
- **1 FP Reduction Model**: VETO model to reduce false positives (currently problematic)

### Test Results (Dec 10, 2025)

| Configuration | Forged Detection | Authentic FP Rate |
|---------------|------------------|-------------------|
| 6 Models (no veto) | **85%** | 62% |
| 7 Models (with veto) | 39% | **2%** |

**Problem Identified**: FP reduction model trained with FP_WEIGHT=3.0 is too aggressive:
- Specificity: 99.82% (great at identifying authentic)
- Recall: 7.25% (terrible at detecting forgeries)
- The model predicts almost everything as "authentic"

### Completed Tasks
- ✅ 6-model ensemble trained and working (85% recall on forged)
- ✅ FP reduction model trained (20 epochs) - but too aggressive
- ✅ Added `--no-veto` flag to script_8 for comparison testing
- ✅ Generated 7,131 authentic augmented images
- ✅ Fixed deprecated albumentations parameters

### In Progress
- 🔄 Retraining FP reduction model with lower FP_WEIGHT (1.5x instead of 3.0x)

### Next Steps
1. Retrain FP reduction model with FP_WEIGHT=1.5
2. Test new veto model on both test sets
3. Find optimal veto factor formula
4. Generate final submission

---

## Key Directories

```
/home/swatson/work/MachineLearning/kaggle/ScientificImageForgeryDetection/
├── bin/                    # All executable scripts
├── models/                 # Trained model checkpoints
├── log/                    # Training and inference logs
├── output/                 # CSV submission files
├── DOCUMENTATION/          # All documentation files
├── train_images/           # Training data (corrected labels)
├── train_masks/            # Training masks
├── validation_images/      # Validation dataset
└── train_images_backup_original/  # Original (mislabeled) backup
```

---

## Quick Reference: Useful Commands

```bash
# Check if training running
ps aux | grep script | grep python | grep -v grep

# Monitor training log
tail -f log/script_*.log

# List all logs
ls -lh log/

# List all outputs (submissions)
ls -lh output/

# Check model files
ls -lh models/ | grep best

# View project structure
tree -L 2 -I '__pycache__'
```

---

## Notes

- All scripts now use relative paths: `os.path.dirname(os.path.dirname(os.path.abspath(__file__)))`
- Scripts can be run from any directory (paths will resolve correctly)
- Log files auto-save to `log/` with timestamp
- CSV outputs auto-save to `output/` with timestamp
- Original training data backed up in `train_images_backup_original/`

---

**Last Updated**: Nov 23, 2025  
**Status**: Label correction complete, retraining in progress
