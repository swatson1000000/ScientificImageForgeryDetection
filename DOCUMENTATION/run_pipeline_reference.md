# run_pipeline.sh - Detailed Reference

## Overview

`run_pipeline.sh` is the main orchestration script that runs the complete training and inference pipeline. It coordinates execution of all three Python scripts in sequence.

## Purpose

This shell script provides:
1. **Single entry point** for the entire workflow
2. **Conda environment activation** (phi4)
3. **Logging** of all steps
4. **Timing** of each phase
5. **Command-line configuration** of thresholds
6. **Skip-training mode** for inference-only runs

## Location

```
bin_4/run_pipeline.sh
```

## Quick Reference

```bash
# Full pipeline (train + inference)
./bin_4/run_pipeline.sh

# Inference only
./bin_4/run_pipeline.sh --skip-training

# Custom thresholds
./bin_4/run_pipeline.sh --classifier-threshold 0.30 --seg-threshold 0.40

# Custom input/output
./bin_4/run_pipeline.sh --test-dir /path/to/images --output results.csv

# Show help
./bin_4/run_pipeline.sh --help
```

## Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--skip-training` | false | Skip Steps 1 & 2, run inference only |
| `--classifier-threshold T` | 0.25 | Binary classifier threshold |
| `--seg-threshold T` | 0.35 | Segmentation threshold |
| `--test-dir DIR` | `validation_images/` | Test images directory |
| `--output FILE` | `output/submission.csv` | Output submission file |
| `-h, --help` | - | Show help message |

## Pipeline Steps

### Step 1: Train V4 Segmentation Models

**Condition**: Only runs if `--skip-training` is NOT set

**Script**: `script_1_train_v4.py`

**Duration**: ~4-6 hours on GPU

**Output**:
- `models/highres_no_ela_v4_best.pth`
- `models/hard_negative_v4_best.pth`
- `models/high_recall_v4_best.pth`
- `models/enhanced_aug_v4_best.pth`

**Log**: `bin_4/logs/train_v4.log`

### Step 2: Train Binary Classifier

**Condition**: Only runs if `--skip-training` is NOT set

**Script**: `script_2_train_binary_classifier.py`

**Duration**: ~30-60 minutes on GPU

**Output**: `models/binary_classifier_best.pth`

**Log**: `bin_4/logs/train_classifier.log`

### Step 3: Generate Submission

**Condition**: Always runs

**Script**: `script_3_two_stage_submission.py`

**Duration**: ~3-5 minutes for 934 images

**Output**: `output/submission.csv` (or `--output` path)

**Log**: `bin_4/logs/submission.log`

## Directory Structure

```
ScientificImageForgeryDetection/
├── bin_4/
│   ├── run_pipeline.sh              # This script
│   ├── script_1_train_v4.py         # Step 1
│   ├── script_2_train_binary_classifier.py  # Step 2
│   ├── script_3_two_stage_submission.py     # Step 3
│   └── logs/                         # Log files
│       ├── train_v4.log
│       ├── train_classifier.log
│       └── submission.log
├── models/                           # Trained models
├── validation_images/                # Default test images
└── output/                           # Submission output
```

## Environment Setup

The script automatically activates the conda environment:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate phi4
```

**Required packages in phi4 environment**:
- Python 3.10+
- PyTorch 2.0+
- timm
- segmentation-models-pytorch
- albumentations
- opencv-python
- pandas
- tqdm

## Script Walkthrough

### 1. Configuration Section

```bash
#!/bin/bash
set -e  # Exit on error

# Path setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$SCRIPT_DIR/logs"
OUTPUT_DIR="$PROJECT_DIR/output"

# Default options
SKIP_TRAINING=false
TEST_DIR="$PROJECT_DIR/validation_images"
OUTPUT_FILE="$OUTPUT_DIR/submission.csv"
CLASSIFIER_THRESHOLD=0.25
SEG_THRESHOLD=0.35
```

### 2. Argument Parsing

```bash
while [[ $# -gt 0 ]]; do
    case $1 in
        --classifier-threshold)
            CLASSIFIER_THRESHOLD="$2"
            shift 2
            ;;
        --seg-threshold)
            SEG_THRESHOLD="$2"
            shift 2
            ;;
        --skip-training)
            SKIP_TRAINING=true
            shift
            ;;
        --test-dir)
            TEST_DIR="$2"
            shift 2
            ;;
        --output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -h|--help)
            # Print help and exit
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done
```

### 3. Environment Activation

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate phi4
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Python: $(which python)"
```

### 4. Training Phase (Conditional)

```bash
if [ "$SKIP_TRAINING" = false ]; then
    # Step 1: V4 Training
    python "$SCRIPT_DIR/script_1_train_v4.py" 2>&1 | tee "$LOG_DIR/train_v4.log"
    
    # Step 2: Classifier Training  
    python "$SCRIPT_DIR/script_2_train_binary_classifier.py" 2>&1 | tee "$LOG_DIR/train_classifier.log"
fi
```

### 5. Inference Phase

```bash
# Step 3: Two-stage submission
CMD="python $SCRIPT_DIR/script_3_two_stage_submission.py"
CMD="$CMD --input $TEST_DIR --output $OUTPUT_FILE"
CMD="$CMD --classifier-threshold $CLASSIFIER_THRESHOLD"
CMD="$CMD --seg-threshold $SEG_THRESHOLD"

$CMD 2>&1 | tee "$LOG_DIR/submission.log"
```

### 6. Summary

```bash
echo "========================================================================"
echo "PIPELINE COMPLETE"
echo "========================================================================"
echo "Submission file: $OUTPUT_FILE"
head -10 "$OUTPUT_FILE"
```

## Usage Examples

### Example 1: Full Training + Inference

```bash
./bin_4/run_pipeline.sh
```

Output:
```
========================================================================
SCIENTIFIC IMAGE FORGERY DETECTION - TWO-STAGE PIPELINE
========================================================================
Project dir: /path/to/ScientificImageForgeryDetection
Skip training: false

========================================================================
STEP 1: TRAINING V4 SEGMENTATION MODELS (4 models)
========================================================================
Training started at: 2025-12-17 10:00:00
...
V4 training completed in 4h 32m

========================================================================
STEP 2: TRAINING BINARY CLASSIFIER
========================================================================
...
Classifier training completed in 45m

========================================================================
STEP 3: GENERATING SUBMISSION (Two-Stage Pipeline)
========================================================================
...
Submission generated in 205s

========================================================================
PIPELINE COMPLETE
========================================================================
Submission file: output/submission.csv
```

### Example 2: Inference Only

```bash
./bin_4/run_pipeline.sh --skip-training
```

Output:
```
========================================================================
SCIENTIFIC IMAGE FORGERY DETECTION - TWO-STAGE PIPELINE
========================================================================
Skip training: true

========================================================================
STEP 3: GENERATING SUBMISSION (Two-Stage Pipeline)
========================================================================
...
```

### Example 3: Custom Thresholds

```bash
./bin_4/run_pipeline.sh \
    --skip-training \
    --classifier-threshold 0.30 \
    --seg-threshold 0.40
```

### Example 4: Custom Input/Output

```bash
./bin_4/run_pipeline.sh \
    --skip-training \
    --test-dir /path/to/test_images \
    --output /path/to/my_submission.csv
```

### Example 5: Running in Background

```bash
# Run with nohup
nohup bash bin_4/run_pipeline.sh --skip-training > log/run_pipeline.log 2>&1 &

# Get the PID
echo $!

# Check status
ps aux | grep run_pipeline

# Monitor progress
tail -f log/run_pipeline.log

# Check completion
cat log/run_pipeline.log | tail -50
```

## Error Handling

The script uses `set -e` which causes immediate exit on any error.

### Common Errors

#### Permission Denied

```bash
./run_pipeline.sh: Permission denied
```

**Solution**:
```bash
chmod +x bin_4/run_pipeline.sh
# or
bash bin_4/run_pipeline.sh
```

#### Conda Environment Not Found

```bash
conda: command not found
```

**Solution**: Ensure miniconda is installed at `~/miniconda3/`

#### Test Directory Not Found

```bash
Warning: Test directory not found: validation_images/
```

**Solution**: Provide valid path with `--test-dir`

#### Python Script Error

The script will exit with the Python script's error code. Check the corresponding log file:
- `bin_4/logs/train_v4.log`
- `bin_4/logs/train_classifier.log`
- `bin_4/logs/submission.log`

## Timing Information

The script tracks duration for each phase:

```
V4 training completed in 4h 32m
Classifier training completed in 45m
Submission generated in 205s
```

## Log Files

All output is captured with `tee` to both console and log files:

```bash
python script.py 2>&1 | tee "$LOG_DIR/logfile.log"
```

Log files location: `bin_4/logs/`

## Customization

### Modifying Default Values

Edit the script to change defaults:

```bash
# Default options
SKIP_TRAINING=false                    # Change to true
TEST_DIR="$PROJECT_DIR/validation_images"  # Change path
CLASSIFIER_THRESHOLD=0.25              # Change threshold
SEG_THRESHOLD=0.35                     # Change threshold
```

### Adding New Options

Add to the argument parsing section:

```bash
--my-option)
    MY_OPTION="$2"
    shift 2
    ;;
```

### Modifying Python Script Calls

Edit the CMD construction:

```bash
CMD="python $SCRIPT_DIR/script_3_two_stage_submission.py"
CMD="$CMD --input $TEST_DIR"
CMD="$CMD --output $OUTPUT_FILE"
CMD="$CMD --my-new-option $MY_VALUE"
```

## Integration

### Calling from Another Script

```bash
#!/bin/bash
cd /path/to/ScientificImageForgeryDetection
./bin_4/run_pipeline.sh --skip-training
```

### Using with Cron

```bash
# crontab -e
0 2 * * * cd /path/to/project && ./bin_4/run_pipeline.sh --skip-training >> /var/log/pipeline.log 2>&1
```

### Docker Integration

```dockerfile
COPY . /app
WORKDIR /app
RUN chmod +x bin_4/run_pipeline.sh
CMD ["./bin_4/run_pipeline.sh", "--skip-training"]
```
