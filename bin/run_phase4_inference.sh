#!/bin/bash
#===============================================================================
# PHASE 4: INFERENCE ONLY
#===============================================================================
# This script runs only Phase 4 (inference) from the full pipeline
# Generates predictions on test_forged_100, test_authentic_100, and validation_images
#
# Usage:
#   nohup bash bin/run_phase4_inference.sh > log/phase4_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#
#===============================================================================

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/log"

mkdir -p "$LOG_DIR"
mkdir -p "$PROJECT_DIR/predictions"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
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

#===============================================================================
# PHASE 4: VALIDATION & INFERENCE
#===============================================================================

echo "========================================================================"
echo " PHASE 4: INFERENCE ONLY"
echo "========================================================================"
echo ""

phase_start

log "Processing test_forged_100..."
python "$SCRIPT_DIR/script_4_two_pass_pipeline.py" \
    --input "$PROJECT_DIR/test_forged_100" \
    --output "$PROJECT_DIR/predictions/test_forged_100" \
    --csv "$PROJECT_DIR/predictions/test_forged_100_results.csv" \
    --json "$PROJECT_DIR/predictions/test_forged_100_results.json" \
    2>&1 | tee -a "$LOG_DIR/inference_forged_${TIMESTAMP}.log"

log "Processing test_authentic_100..."
python "$SCRIPT_DIR/script_4_two_pass_pipeline.py" \
    --input "$PROJECT_DIR/test_authentic_100" \
    --output "$PROJECT_DIR/predictions/test_authentic_100" \
    --csv "$PROJECT_DIR/predictions/test_authentic_100_results.csv" \
    --json "$PROJECT_DIR/predictions/test_authentic_100_results.json" \
    2>&1 | tee -a "$LOG_DIR/inference_authentic_${TIMESTAMP}.log"

log "Processing validation_images..."
python "$SCRIPT_DIR/script_4_two_pass_pipeline.py" \
    --input "$PROJECT_DIR/validation_images" \
    --output "$PROJECT_DIR/predictions/validation_images" \
    --csv "$PROJECT_DIR/predictions/validation_images_results.csv" \
    --json "$PROJECT_DIR/predictions/validation_images_results.json" \
    2>&1 | tee -a "$LOG_DIR/inference_validation_${TIMESTAMP}.log"

phase_end "PHASE 4: INFERENCE"

echo ""
log "Predictions saved to: $PROJECT_DIR/predictions/"
log "Phase 4 finished at $(date)"
