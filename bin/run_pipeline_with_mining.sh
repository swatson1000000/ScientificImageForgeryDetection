#!/bin/bash
#===============================================================================
# COMPLETE TRAINING PIPELINE WITH HARD NEGATIVE MINING
#===============================================================================
#
# This script runs the full training pipeline:
#   1. Initial training of segmentation models + binary classifier
#   2. Hard negative mining on authentic test images
#   3. Retraining with hard negatives included
#   4. Final inference/submission generation
#
# Usage:
#   ./run_pipeline_with_mining.sh                    # Full pipeline
#   ./run_pipeline_with_mining.sh --skip-initial     # Skip initial training
#   ./run_pipeline_with_mining.sh --skip-mining      # Skip mining step
#   ./run_pipeline_with_mining.sh --inference-only   # Only run inference
#
#===============================================================================

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/log"

mkdir -p "$LOG_DIR"

# Track overall script start time
SCRIPT_START_TIME=$(date +%s)
SCRIPT_START_DISPLAY=$(date '+%Y-%m-%d %H:%M:%S')

# Argument parsing moved up, so augmentation phase uses correct values

# Parse arguments
SKIP_INITIAL=false
SKIP_MINING=false
INFERENCE_ONLY=false

for arg in "$@"; do
    case $arg in
        --skip-initial)
            SKIP_INITIAL=true
            shift
            ;;
        --skip-mining)
            SKIP_MINING=true
            shift
            ;;
        --inference-only)
            INFERENCE_ONLY=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-initial    Skip initial model training (Phase 1)"
            echo "  --skip-mining     Skip hard negative mining (Phase 2)"
            echo "  --inference-only  Only run inference (Phase 4)"
            echo "  --help            Show this help message"
            exit 0
            ;;
    esac
done

# ============================================================================
# DATA AUGMENTATION PHASE
# ============================================================================

if [ "$INFERENCE_ONLY" = false ] && [ "$SKIP_INITIAL" = false ]; then
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

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="$LOG_DIR/pipeline_${TIMESTAMP}.log"

# Timing variables
PHASE_START_TIME=""
PHASE_START_DISPLAY=""

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$MAIN_LOG"
}

log_section() {
    echo "" | tee -a "$MAIN_LOG"
    echo "========================================================================" | tee -a "$MAIN_LOG"
    echo " $1" | tee -a "$MAIN_LOG"
    echo "========================================================================" | tee -a "$MAIN_LOG"
}

# Function to track phase timing
phase_start() {
    PHASE_START_TIME=$(date +%s)
    PHASE_START_DISPLAY=$(date '+%Y-%m-%d %H:%M:%S')
}

# Function to display phase elapsed time
phase_end() {
    local phase_name="$1"
    local phase_end_time=$(date +%s)
    local phase_end_display=$(date '+%Y-%m-%d %H:%M:%S')
    local elapsed=$((phase_end_time - PHASE_START_TIME))
    local hours=$((elapsed / 3600))
    local minutes=$(((elapsed % 3600) / 60))
    local seconds=$((elapsed % 60))
    
    log "✓ $phase_name completed"
    log "  Start: $PHASE_START_DISPLAY | End: $phase_end_display | Duration: ${hours}h ${minutes}m ${seconds}s"
}

# Display script start time
log_section "PIPELINE START"
log "Script started at: $SCRIPT_START_DISPLAY"
log ""

#===============================================================================
# PHASE 1: INITIAL MODEL TRAINING
#===============================================================================

if [ "$INFERENCE_ONLY" = false ] && [ "$SKIP_INITIAL" = false ]; then
    log_section "PHASE 1: INITIAL MODEL TRAINING"
    phase_start
    
    # Step 1a: Train segmentation models FIRST (needed for hard negative mining)
    log "Training initial segmentation models..."
    python "$SCRIPT_DIR/script_1_train_v3.py" --no-hard-negatives 2>&1 | tee -a "$LOG_DIR/segmentation_initial_${TIMESTAMP}.log"
    log "Initial segmentation training complete."
    
    # Step 1b: Train binary classifier SECOND (without hard negatives initially)
    log "Training binary classifier..."
    python "$SCRIPT_DIR/script_2_train_binary_classifier.py" --no-hard-negatives 2>&1 | tee -a "$LOG_DIR/classifier_initial_${TIMESTAMP}.log"
    
    if [ ! -f "$PROJECT_DIR/models/binary_classifier_best.pth" ]; then
        log "ERROR: Binary classifier training failed!"
        exit 1
    fi
    log "Binary classifier training complete."
    
    phase_end "PHASE 1: INITIAL MODEL TRAINING"
else
    log "Skipping Phase 1 (initial training)"
fi

#===============================================================================
# PHASE 2: HARD NEGATIVE MINING
#===============================================================================

if [ "$INFERENCE_ONLY" = false ] && [ "$SKIP_MINING" = false ]; then
    log_section "PHASE 2: HARD NEGATIVE MINING"
    phase_start
    
    log "Running hard negative mining on authentic test images..."
    
    # Run hard negative finder
    python "$SCRIPT_DIR/find_hard_negatives.py" \
        --authentic-dir "$PROJECT_DIR/test_authentic_100" \
        --output "hard_negative_ids.txt" \
        --classifier-threshold 0.25 \
        --seg-threshold 0.35 \
        --min-area 300 \
        2>&1 | tee -a "$LOG_DIR/mining_${TIMESTAMP}.log"
    
    if [ ! -f "$PROJECT_DIR/hard_negative_ids.txt" ]; then
        log "WARNING: No hard negatives file created (possibly no false positives found)"
        touch "$PROJECT_DIR/hard_negative_ids.txt"
    fi
    
    # Count hard negatives found
    HN_COUNT=$(wc -l < "$PROJECT_DIR/hard_negative_ids.txt" 2>/dev/null || echo "0")
    log "Found $HN_COUNT hard negatives"
    
    log "Hard negative mining complete."
    phase_end "PHASE 2: HARD NEGATIVE MINING"
else
    log "Skipping Phase 2 (hard negative mining)"
fi

#===============================================================================
# PHASE 3: RETRAIN WITH HARD NEGATIVES
#===============================================================================

if [ "$INFERENCE_ONLY" = false ]; then
    log_section "PHASE 3: RETRAINING WITH HARD NEGATIVES"
    phase_start
    
    # Check if we have hard negatives to use
    if [ -f "$PROJECT_DIR/hard_negative_ids.txt" ]; then
        HN_COUNT=$(wc -l < "$PROJECT_DIR/hard_negative_ids.txt" 2>/dev/null || echo "0")
        log "Retraining with $HN_COUNT hard negatives..."
    else
        log "No hard_negative_ids.txt found, training with defaults..."
    fi
    
    # Step 3a: Retrain segmentation models (WITH hard negatives)
    log "Retraining segmentation models..."
    python "$SCRIPT_DIR/script_1_train_v3.py" 2>&1 | tee -a "$LOG_DIR/segmentation_retrain_${TIMESTAMP}.log"
    log "Segmentation retraining complete."
    
    # Step 3b: Retrain classifier (WITH hard negatives)
    log "Retraining binary classifier with hard negatives..."
    python "$SCRIPT_DIR/script_2_train_binary_classifier.py" 2>&1 | tee -a "$LOG_DIR/classifier_retrain_${TIMESTAMP}.log"
    log "Classifier retraining complete."
    
    # Optionally train v4 models as well
    if [ -f "$SCRIPT_DIR/script_3_train_v4.py" ]; then
        log "Training v4 segmentation models..."
        python "$SCRIPT_DIR/script_3_train_v4.py" 2>&1 | tee -a "$LOG_DIR/segmentation_v4_${TIMESTAMP}.log"
    fi
    
    log "All retraining complete."
    phase_end "PHASE 3: RETRAINING WITH HARD NEGATIVES"
fi

#===============================================================================
# PHASE 4: VALIDATION & INFERENCE
#===============================================================================

log_section "PHASE 4: VALIDATION & INFERENCE"
phase_start

# Run validation on test set
if [ -f "$SCRIPT_DIR/validate_test.py" ]; then
    log "Running validation..."
    python "$SCRIPT_DIR/validate_test.py" 2>&1 | tee -a "$LOG_DIR/validation_${TIMESTAMP}.log"
fi

# Generate submission using two-stage pipeline
log "Generating submission with two-pass pipeline..."
python "$SCRIPT_DIR/script_4_two_pass_pipeline.py" \
    --input "$PROJECT_DIR/test" \
    --output "$PROJECT_DIR/predictions" \
    --csv "$PROJECT_DIR/submission.csv" \
    --json "$PROJECT_DIR/predictions/results.json" \
    2>&1 | tee -a "$LOG_DIR/inference_${TIMESTAMP}.log"

phase_end "PHASE 4: VALIDATION & INFERENCE"

#===============================================================================
# SUMMARY
#===============================================================================

log_section "PIPELINE COMPLETE"

# Calculate and display overall script timing
SCRIPT_END_TIME=$(date +%s)
SCRIPT_END_DISPLAY=$(date '+%Y-%m-%d %H:%M:%S')
TOTAL_ELAPSED=$((SCRIPT_END_TIME - SCRIPT_START_TIME))
TOTAL_HOURS=$((TOTAL_ELAPSED / 3600))
TOTAL_MINUTES=$(((TOTAL_ELAPSED % 3600) / 60))
TOTAL_SECONDS=$((TOTAL_ELAPSED % 60))

log ""
log "╔════════════════════════════════════════════════════════════════════════╗"
log "║ COMPLETE PIPELINE EXECUTION TIME                                       ║"
log "╠════════════════════════════════════════════════════════════════════════╣"
log "║ Start: $SCRIPT_START_DISPLAY"
log "║ End:   $SCRIPT_END_DISPLAY"
log "║ Total Duration: ${TOTAL_HOURS}h ${TOTAL_MINUTES}m ${TOTAL_SECONDS}s"
log "╚════════════════════════════════════════════════════════════════════════╝"
log ""

log "Models saved to: $PROJECT_DIR/models/"
log "Predictions saved to: $PROJECT_DIR/predictions/"
log "Submission CSV: $PROJECT_DIR/submission.csv"
log "Logs saved to: $LOG_DIR/"

# Print model files
log ""
log "Trained models:"
ls -la "$PROJECT_DIR/models/"*.pth 2>/dev/null | while read line; do
    log "  $line"
done

log ""
log "Pipeline finished at $(date)"
