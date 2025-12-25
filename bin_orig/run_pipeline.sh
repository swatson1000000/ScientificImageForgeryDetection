#!/bin/bash
#
# Two-Stage Training + Inference Pipeline
# ========================================
# 1. Train V4 segmentation ensemble (4 models)
# 2. Train binary classifier
# 3. Generate submission with two-stage pipeline
#
# Usage:
#   ./run_pipeline.sh                    # Full pipeline (train + inference)
#   ./run_pipeline.sh --skip-training    # Skip training, just generate submission
#

set -e

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

# Parse arguments
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
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --classifier-threshold T  Classifier threshold (default: 0.25)"
            echo "  --seg-threshold T         Segmentation threshold (default: 0.35)"
            echo "  --skip-training           Skip training, only run inference"
            echo "  --test-dir DIR            Directory containing test images"
            echo "  --output FILE             Output submission file (default: output/submission.csv)"
            echo "  -h, --help                Show this help"
            echo ""
            echo "Two-Stage Pipeline:"
            echo "  Stage 1: Binary classifier filters images (threshold=0.25)"
            echo "  Stage 2: 4-model V4 ensemble with TTA generates masks"
            echo ""
            echo "Validated Performance:"
            echo "  Net: 2033 (TP: 2173, FP: 140)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"

# Record pipeline start time
PIPELINE_START=$(date +%s)
PIPELINE_START_TIME=$(date '+%Y-%m-%d %H:%M:%S')

echo "========================================================================"
echo "SCIENTIFIC IMAGE FORGERY DETECTION - TWO-STAGE PIPELINE"
echo "========================================================================"
echo "Started at: $PIPELINE_START_TIME"
echo "Project dir: $PROJECT_DIR"
echo "Log dir: $LOG_DIR"
echo "Classifier threshold: $CLASSIFIER_THRESHOLD"
echo "Seg threshold: $SEG_THRESHOLD"
echo "Skip training: $SKIP_TRAINING"
echo ""

# Activate phi4 conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate phi4
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Python: $(which python)"
echo ""

# ============================================================================
# DATA AUGMENTATION PHASE
# ============================================================================

if [ "$SKIP_TRAINING" = false ]; then
    echo "========================================================================"
    echo "STEP 0a: GENERATING AUGMENTED FORGED IMAGES"
    echo "========================================================================"
    echo ""
    
    if [ -d "$PROJECT_DIR/train_images/forged_augmented" ] && [ "$(ls -A $PROJECT_DIR/train_images/forged_augmented 2>/dev/null)" ]; then
        echo "Augmented forged images already exist. Skipping..."
    else
        echo "Log file: $LOG_DIR/augment_forged.log"
        python "$SCRIPT_DIR/generate_forged_augmented.py" 2>&1 | tee "$LOG_DIR/augment_forged.log"
        echo "Forged augmentation complete."
    fi
    echo ""

    echo "========================================================================"
    echo "STEP 0b: GENERATING AUGMENTED AUTHENTIC IMAGES"
    echo "========================================================================"
    echo ""
    
    if [ -d "$PROJECT_DIR/train_images/authentic_augmented" ] && [ "$(ls -A $PROJECT_DIR/train_images/authentic_augmented 2>/dev/null)" ]; then
        echo "Augmented authentic images already exist. Skipping..."
    else
        echo "Log file: $LOG_DIR/augment_authentic.log"
        python "$SCRIPT_DIR/generate_authentic_augmented.py" \
            --input "$PROJECT_DIR/train_images/authentic" \
            --output "$PROJECT_DIR/train_images/authentic_augmented" \
            --multiplier 2 2>&1 | tee "$LOG_DIR/augment_authentic.log"
        echo "Authentic augmentation complete."
    fi
    echo ""
fi

# ============================================================================
# TRAINING PHASE
# ============================================================================

if [ "$SKIP_TRAINING" = false ]; then
    echo "========================================================================"
    echo "STEP 1a: TRAINING V3 SEGMENTATION MODELS (4 models - base training)"
    echo "========================================================================"
    echo ""

    V3_START=$(date +%s)
    V3_START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
    echo "V3 training started at: $V3_START_TIME"
    echo "Log file: $LOG_DIR/train_v3.log"
    echo ""
    
    python "$SCRIPT_DIR/script_1_train_v3.py" 2>&1 | tee "$LOG_DIR/train_v3.log"
    
    V3_END=$(date +%s)
    V3_DURATION=$((V3_END - V3_START))
    echo ""
    echo "V3 training completed in $((V3_DURATION / 3600))h $((V3_DURATION % 3600 / 60))m"
    echo ""

    echo "========================================================================"
    echo "STEP 1b: TRAINING V4 SEGMENTATION MODELS (4 models - fine-tuning)"
    echo "========================================================================"
    echo ""

    TRAIN_START=$(date +%s)
    TRAIN_START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
    echo "V4 training started at: $TRAIN_START_TIME"
    echo "Log file: $LOG_DIR/train_v4.log"
    echo ""
    
    python "$SCRIPT_DIR/script_3_train_v4.py" 2>&1 | tee "$LOG_DIR/train_v4.log"
    
    TRAIN_END=$(date +%s)
    TRAIN_DURATION=$((TRAIN_END - TRAIN_START))
    echo ""
    echo "V4 training completed in $((TRAIN_DURATION / 3600))h $((TRAIN_DURATION % 3600 / 60))m"
    echo ""

    echo "========================================================================"
    echo "STEP 2: TRAINING BINARY CLASSIFIER"
    echo "========================================================================"
    echo ""

    CLASSIFIER_START=$(date +%s)
    echo "Log file: $LOG_DIR/train_classifier.log"
    echo ""
    
    python "$SCRIPT_DIR/script_2_train_binary_classifier.py" 2>&1 | tee "$LOG_DIR/train_classifier.log"
    
    CLASSIFIER_END=$(date +%s)
    CLASSIFIER_DURATION=$((CLASSIFIER_END - CLASSIFIER_START))
    echo ""
    echo "Classifier training completed in $((CLASSIFIER_DURATION / 60))m"
    echo ""
fi

# ============================================================================
# INFERENCE PHASE
# ============================================================================

echo "========================================================================"
echo "STEP 3: GENERATING SUBMISSION (Two-Stage Pipeline)"
echo "========================================================================"
echo ""

if [ ! -d "$TEST_DIR" ]; then
    echo "Warning: Test directory not found: $TEST_DIR"
    echo "Skipping submission generation."
    exit 0
fi

INFER_START=$(date +%s)
echo "Input: $TEST_DIR"
echo "Output: $OUTPUT_FILE"
echo "Classifier threshold: $CLASSIFIER_THRESHOLD"
echo "Seg threshold: $SEG_THRESHOLD"
echo "Log file: $LOG_DIR/submission.log"
echo ""


# Two-stage submission command (now using script_4_two_pass_pipeline.py)
CMD="python $SCRIPT_DIR/script_4_two_pass_pipeline.py --input $TEST_DIR --output $OUTPUT_FILE"
CMD="$CMD --classifier-threshold $CLASSIFIER_THRESHOLD --seg-threshold $SEG_THRESHOLD"

$CMD 2>&1 | tee "$LOG_DIR/submission.log"

INFER_END=$(date +%s)
INFER_DURATION=$((INFER_END - INFER_START))
echo ""
echo "Submission generated in ${INFER_DURATION}s"
echo ""

# ============================================================================
# SUMMARY
# ============================================================================

# Record pipeline end time
PIPELINE_END=$(date +%s)
PIPELINE_END_TIME=$(date '+%Y-%m-%d %H:%M:%S')
PIPELINE_DURATION=$((PIPELINE_END - PIPELINE_START))

echo "========================================================================"
echo "PIPELINE COMPLETE"
echo "========================================================================"
echo ""
echo "Started at:  $PIPELINE_START_TIME"
echo "Finished at: $PIPELINE_END_TIME"
echo "Total time:  $((PIPELINE_DURATION / 3600))h $((PIPELINE_DURATION % 3600 / 60))m $((PIPELINE_DURATION % 60))s"
echo ""
echo "Submission file: $OUTPUT_FILE"
echo ""
if [ -f "$OUTPUT_FILE" ]; then
    echo "Preview (first 10 lines):"
    head -10 "$OUTPUT_FILE"
fi
