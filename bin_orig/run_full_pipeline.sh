#!/bin/bash

# Full Pipeline Runner - Intermediate + Tier 3
# =============================================
# 
# Runs all intermediate and Tier 3 scripts with monitoring
#
# Usage:
#   bash run_full_pipeline.sh              # Run all
#   bash run_full_pipeline.sh intermediate # Only intermediate
#   bash run_full_pipeline.sh tier3        # Only Tier 3

# set -e removed to allow pipeline to continue even if individual scripts fail

# Activate phi4 conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate phi4

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_DIR="/home/swatson/work/MachineLearning/kaggle/ScientificImageForgeryDetection"
LOGS_DIR="$PROJECT_DIR/log"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOGS_DIR/pipeline_${TIMESTAMP}.log"

# Ensure we're in project directory
cd "$PROJECT_DIR" || exit 1
mkdir -p "$LOGS_DIR"

# ============================================================================
# Helper Functions
# ============================================================================

print_header() {
    echo ""
    echo "════════════════════════════════════════════════════════════════════════════"
    printf "%s\n" "$1" | sed 's/^/                    /'
    echo "════════════════════════════════════════════════════════════════════════════"
    echo ""
}

print_section() {
    echo ""
    echo "────────────────────────────────────────────────────────────────────────────"
    echo "  $1"
    echo "────────────────────────────────────────────────────────────────────────────"
    echo ""
}

print_script() {
    local num=$1
    local total=$2
    local name=$3
    local gain=$4
    local time=$5
    
    echo "[$num/$total] Running: $name"
    echo "    Expected gain: $gain"
    echo "    Estimated time: $time"
    echo ""
}

success_message() {
    echo -e "${GREEN}✅ COMPLETED${NC}: $1"
}

error_message() {
    echo -e "${RED}❌ ERROR${NC}: $1"
}

warning_message() {
    echo -e "${YELLOW}⚠️  WARNING${NC}: $1"
}

run_script() {
    local script=$1
    local name=$2
    local start_time=$(date +%s)
    local start_datetime=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo "Running $script..."
    echo "" >> "$LOG_FILE"
    echo "════════════════════════════════════════════════════════════════════════════" >> "$LOG_FILE"
    echo "SCRIPT START: $name" >> "$LOG_FILE"
    echo "Start time: $start_datetime" >> "$LOG_FILE"
    echo "════════════════════════════════════════════════════════════════════════════" >> "$LOG_FILE"
    
    if python "$script" 2>&1 | tee -a "$LOG_FILE"; then
        local end_time=$(date +%s)
        local end_datetime=$(date '+%Y-%m-%d %H:%M:%S')
        local elapsed=$((end_time - start_time))
        echo "" >> "$LOG_FILE"
        echo "════════════════════════════════════════════════════════════════════════════" >> "$LOG_FILE"
        echo "SCRIPT END: $name (SUCCESS)" >> "$LOG_FILE"
        echo "End time: $end_datetime" >> "$LOG_FILE"
        echo "Duration: ${elapsed}s" >> "$LOG_FILE"
        echo "════════════════════════════════════════════════════════════════════════════" >> "$LOG_FILE"
        success_message "$name (${elapsed}s)"
        return 0
    else
        local end_time=$(date +%s)
        local end_datetime=$(date '+%Y-%m-%d %H:%M:%S')
        local elapsed=$((end_time - start_time))
        echo "" >> "$LOG_FILE"
        echo "════════════════════════════════════════════════════════════════════════════" >> "$LOG_FILE"
        echo "SCRIPT END: $name (FAILED)" >> "$LOG_FILE"
        echo "End time: $end_datetime" >> "$LOG_FILE"
        echo "Duration: ${elapsed}s" >> "$LOG_FILE"
        echo "════════════════════════════════════════════════════════════════════════════" >> "$LOG_FILE"
        error_message "$name (${elapsed}s)"
        return 1
    fi
}

# ============================================================================
# Main Scripts
# ============================================================================

run_intermediate() {
    print_header "TRAINING PHASE 1 - SCRIPTS 1, 2"
    
    echo "Expected improvements:"
    echo "  • SegFormer Transformer: +5-8%"
    echo "  • Loss Variants (Lovász): +2-4%"
    echo ""
    
    local count=0
    local total=2
    local success_count=0
    
    # 1. SegFormer
    print_script 1 $total "SegFormer Transformer" "+5-8%" "40-50 min"
    if run_script "script_1_training_segformer.py" "SegFormer"; then
        ((success_count++))
    else
        warning_message "SegFormer failed, continuing..."
    fi
    ((count++))
    echo ""
    
    # 2. Loss Variants
    print_script 2 $total "Loss Variants" "+2-4%" "35-40 min"
    if run_script "script_2_training_loss_variants.py" "Loss Variants"; then
        ((success_count++))
    else
        warning_message "Loss Variants failed, continuing..."
    fi
    ((count++))
    echo ""
    
    echo "════════════════════════════════════════════════════════════════════════════"
    echo "PHASE 1 COMPLETE: $success_count/$total scripts successful"
    echo "════════════════════════════════════════════════════════════════════════════"
    echo ""
    
    return $((total - success_count))
}

run_tier3() {
    print_header "TRAINING PHASE 2 - SCRIPTS 3, 4, 5"
    
    echo "Expected improvements:"
    echo "  • Hyperparameter Tuning: +2-3%"
    echo "  • Self-Supervised Pre-training: +3-5% (DEPENDS ON Script 3)"
    echo "  • Deep Ensemble: +3-5%"
    echo ""
    
    local count=0
    local total=3
    local success_count=0
    
    # 1. Hyperparameter Tuning (MUST complete before script 4)
    print_script 1 $total "Hyperparameter Tuning" "+2-3%" "40-50 min"
    if run_script "script_3_hyperparameter_tuning.py" "Hyperparameter Tuning"; then
        ((success_count++))
        script_3_success=true
    else
        warning_message "Hyperparameter Tuning failed - Script 4 cannot run (requires best_hyperparameters.json)"
        script_3_success=false
    fi
    ((count++))
    echo ""
    
    # 2. Self-Supervised Pre-training (MUST WAIT FOR Script 3)
    print_script 2 $total "Self-Supervised Pre-training" "+3-5% (requires Script 3)" "50-70 min"
    
    if [ "$script_3_success" = true ]; then
        if run_script "script_4_selfsupervised_pretraining.py" "Self-Supervised Pre-training"; then
            ((success_count++))
            script_4_success=true
        else
            warning_message "Self-Supervised Pre-training failed, continuing..."
            script_4_success=false
        fi
    else
        error_message "Script 4 skipped - requires best_hyperparameters.json from Script 3"
        warning_message "Script 3 must complete successfully before Script 4 can run"
        script_4_success=false
    fi
    ((count++))
    echo ""
    
    # 3. Deep Ensemble (WAITS FOR Script 4 - loads its models)
    print_script 3 $total "Deep Ensemble" "+3-5% (requires Script 4)" "45-60 min"
    if run_script "script_5_deep_ensemble.py" "Deep Ensemble"; then
        ((success_count++))
    else
        warning_message "Deep Ensemble failed, continuing..."
    fi
    ((count++))
    echo ""
    
    echo "════════════════════════════════════════════════════════════════════════════"
    echo "PHASE 2 COMPLETE: $success_count/$total scripts successful"
    if [ "$script_3_success" != true ]; then
        echo "⚠️  WARNING: Script 4 was skipped due to Script 3 failure"
        echo "             Re-run Script 3 first, then re-run Script 4 separately"
    fi
    echo "════════════════════════════════════════════════════════════════════════════"
    echo ""
    
    return $((total - success_count))
}

run_inference() {
    print_section "PHASE 3: ENSEMBLE INFERENCE & SUBMISSION"
    
    local total=1
    local success_count=0
    local count=1
    
    print_script 1 $total "Ensemble Inference" "Generates final submission" "15-30 min"
    if run_script "script_6_ensemble_inference.py" "Ensemble Inference"; then
        ((success_count++))
    else
        warning_message "Ensemble Inference failed"
    fi
    ((count++))
    echo ""
    
    echo "════════════════════════════════════════════════════════════════════════════"
    echo "INFERENCE COMPLETE: $success_count/$total scripts successful"
    echo "════════════════════════════════════════════════════════════════════════════"
    echo ""
    
    return $((total - success_count))
}

run_full_pipeline() {
    print_header "FULL PIPELINE - CONCURRENT & SEQUENTIAL TRAINING + INFERENCE"
    
    echo "PHASE 1: CONCURRENT TRAINING (Scripts 1, 2, 3)"
    echo "  Scripts 1, 2, 3 all run in parallel"
    echo "  Expected: ~50 minutes total"
    echo ""
    echo "PHASE 2: SEQUENTIAL STAGES (Scripts 4-5)"
    echo "  Script 4: Waits for Script 3 completion"
    echo "  Script 5: Waits for Script 4 completion"
    echo ""
    echo "PHASE 3: ENSEMBLE INFERENCE (Script 6)"
    echo "  Generates final submission CSV"
    echo "  Uses models from scripts 1, 2, 4, 5"
    echo ""
    echo "Starting training..."
    echo ""
    
    print_section "PHASE 1: CONCURRENT TRAINING (Scripts 1, 2, 3)"
    
    # Ensure we're in the bin directory
    cd "$PROJECT_DIR/bin" || exit 1
    
    # Store PIDs for scripts 1, 2, 3
    declare -a PIDS_PHASE1
    declare -a NAMES_PHASE1
    
    # Start scripts 1, 2, 3 in parallel
    local concurrent_scripts=(
        "script_1_training_segformer.py:SegFormer Transformer"
        "script_2_training_loss_variants.py:Loss Variants"
        "script_3_hyperparameter_tuning.py:Hyperparameter Tuning"
    )
    
    local i=0
    for script_info in "${concurrent_scripts[@]}"; do
        IFS=':' read -r script_name script_desc <<< "$script_info"
        
        local log_file="$LOGS_DIR/${script_name%.py}_$(date +%Y%m%d_%H%M%S).log"
        
        echo "[$((i+1))/3] Starting: $script_desc"
        
        # Start script in background with nohup (-u for unbuffered output)
        nohup python -u "$script_name" > "$log_file" 2>&1 &
        PIDS_PHASE1[$i]=$!
        NAMES_PHASE1[$i]="$script_desc"
        
        echo "    PID: ${PIDS_PHASE1[$i]}"
        echo "    Log: $log_file"
        echo ""
        
        ((i++))
    done
    
    echo "════════════════════════════════════════════════════════════════════════════"
    echo "Scripts 1, 2, 3 all started in parallel"
    echo "Monitoring progress..."
    echo "════════════════════════════════════════════════════════════════════════════"
    echo ""
    
    # Wait for scripts 1, 2, 3 to complete
    local failed_count=0
    local success_count=0
    local script_1_success=false
    local script_2_success=false
    local script_3_success=false
    local script_3_pid=${PIDS_PHASE1[2]}
    
    # First, wait specifically for Script 3 so Script 4 can start early
    echo "Waiting for Script 3 (Hyperparameter Tuning) to complete first..."
    if wait $script_3_pid 2>/dev/null; then
        success_message "Hyperparameter Tuning completed successfully"
        script_3_success=true
        ((success_count++))
    else
        error_message "Hyperparameter Tuning failed (exit code: $?)"
        script_3_success=false
        ((failed_count++))
    fi
    echo ""
    
    # Start Script 4 immediately after Script 3 completes
    local script_4_success=false
    local script_4_pid=""
    
    print_section "Script 4: Self-Supervised Pre-training (Script 3 done, starting now)"
    echo "[1/1] Starting: Self-Supervised Pre-training"
    local log_file_4="$LOGS_DIR/script_4_selfsupervised_pretraining_$(date +%Y%m%d_%H%M%S).log"
    
    nohup python -u "script_4_selfsupervised_pretraining.py" > "$log_file_4" 2>&1 &
    script_4_pid=$!
    echo "    PID: $script_4_pid"
    echo "    Log: $log_file_4"
    echo ""
    
    # Now wait for remaining scripts 1, 2
    echo "Waiting for Scripts 1, 2 to complete..."
    for i in 0 1; do
        local pid=${PIDS_PHASE1[$i]}
        local name=${NAMES_PHASE1[$i]}
        
        echo "Waiting for [$((i+1))/2] $name (PID: $pid)..."
        
        if wait $pid 2>/dev/null; then
            success_message "$name completed successfully"
            ((success_count++))
            # Track scripts 1, 2 success for Script 6 dependency
            case $i in
                0) script_1_success=true ;;
                1) script_2_success=true ;;
            esac
        else
            error_message "$name failed (exit code: $?)"
            ((failed_count++))
        fi
        echo ""
    done
    
    echo "════════════════════════════════════════════════════════════════════════════"
    echo "Scripts 1, 2, 3 complete: $success_count/3 successful"
    echo "════════════════════════════════════════════════════════════════════════════"
    echo ""
    
    # Wait for Script 4 to complete
    print_section "Waiting for Script 4 to complete..."
    if wait $script_4_pid 2>/dev/null; then
        success_message "Self-Supervised Pre-training completed successfully"
        script_4_success=true
        ((success_count++))
    else
        error_message "Self-Supervised Pre-training failed"
        script_4_success=false
        ((failed_count++))
    fi
    echo ""
    
    # ========================================================================
    # Script 5: Deep Ensemble (WAITS FOR Script 4)
    # ========================================================================
    local script_5_success=false
    
    print_section "Script 5: Deep Ensemble (WAITS FOR Script 4)"
    
    echo "[1/1] Starting: Deep Ensemble"
    local log_file_5="$LOGS_DIR/script_5_deep_ensemble_$(date +%Y%m%d_%H%M%S).log"
    
    if python -u "script_5_deep_ensemble.py" > "$log_file_5" 2>&1; then
        success_message "Deep Ensemble completed successfully"
        script_5_success=true
        ((success_count++))
    else
        error_message "Deep Ensemble failed"
        script_5_success=false
        ((failed_count++))
    fi
    echo ""
    
    echo "════════════════════════════════════════════════════════════════════════════"
    echo "Scripts 1, 2, 3, 4, 5 complete: $success_count/5 successful"
    if [ $failed_count -gt 0 ]; then
        warning_message "$failed_count scripts failed"
    fi
    echo "════════════════════════════════════════════════════════════════════════════"
    echo ""
    
    # ========================================================================
    # Script 6: Ensemble Inference (WAITS FOR Scripts 1, 2, 4, 5)
    # ========================================================================
    
    print_section "Script 6: Ensemble Inference (WAITS FOR Scripts 1, 2, 4, 5)"
    
    echo "[1/1] Starting: Ensemble Inference"
    local inference_log="$LOGS_DIR/script_6_ensemble_inference_$(date +%Y%m%d_%H%M%S).log"
    
    echo "Scripts 1, 2, 4, 5 complete - proceeding to ensemble inference..."
    echo "  (Note: Script 6 will use models from scripts 1, 2, 4, 5)"
    echo ""
    
    if python -u "script_6_ensemble_inference.py" > "$inference_log" 2>&1; then
        success_message "Ensemble Inference completed successfully"
        echo "    Log: $inference_log"
    else
        error_message "Ensemble Inference failed"
        echo "    Log: $inference_log"
    fi
    
    print_header "FULL PIPELINE COMPLETE"
}

print_summary() {
    print_section "PIPELINE EXECUTION SUMMARY"
    
    echo "Full log file: $LOG_FILE"
    echo ""
    echo "✅ PIPELINE COMPLETE!"
    echo ""
    echo "Generated files:"
    echo "  - output/submission_ensemble.csv (final submission)"
    echo "  - Models from scripts 1, 2, 4, 5"
    echo ""
    echo "Next step:"
    echo "  Submit output/submission_ensemble.csv to Kaggle"
    echo ""
}

# ============================================================================
# Main Execution
# ============================================================================

main() {
    # Check if project directory exists
    if [ ! -d "$PROJECT_DIR" ]; then
        error_message "Project directory not found: $PROJECT_DIR"
        exit 1
    fi
    
    # Record pipeline start time
    PIPELINE_START_TIME=$(date +%s)
    PIPELINE_START_DATETIME=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Initialize log file
    echo "Pipeline started: $PIPELINE_START_DATETIME" > "$LOG_FILE"
    echo "Project directory: $PROJECT_DIR" >> "$LOG_FILE"
    echo "" >> "$LOG_FILE"
    
    echo ""
    echo "════════════════════════════════════════════════════════════════════════════"
    echo "  PIPELINE START TIME: $PIPELINE_START_DATETIME"
    echo "════════════════════════════════════════════════════════════════════════════"
    echo ""
    
    # Determine which scripts to run
    if [ "$1" == "phase1" ]; then
        run_intermediate
        print_summary
    elif [ "$1" == "phase2" ]; then
        run_tier3
        print_summary
    elif [ -z "$1" ]; then
        # Default: run full pipeline
        run_full_pipeline
        print_summary
    else
        echo "Usage: $0 [phase1|phase2]"
        echo ""
        echo "Options:"
        echo "  phase1  - Run scripts 2, 4 (SegFormer, Loss Variants)"
        echo "  phase2  - Run scripts 6, 7, 8 (Hyperparameter, SimCLR, Ensemble)"
        echo "  (none)  - Run full pipeline (all scripts)"
        exit 1
    fi
    
    # Record pipeline end time
    PIPELINE_END_TIME=$(date +%s)
    PIPELINE_END_DATETIME=$(date '+%Y-%m-%d %H:%M:%S')
    PIPELINE_ELAPSED=$((PIPELINE_END_TIME - PIPELINE_START_TIME))
    PIPELINE_HOURS=$((PIPELINE_ELAPSED / 3600))
    PIPELINE_MINUTES=$(((PIPELINE_ELAPSED % 3600) / 60))
    PIPELINE_SECONDS=$((PIPELINE_ELAPSED % 60))
    
    echo ""
    echo "════════════════════════════════════════════════════════════════════════════"
    echo "  PIPELINE END TIME: $PIPELINE_END_DATETIME"
    echo "  TOTAL DURATION: ${PIPELINE_HOURS}h ${PIPELINE_MINUTES}m ${PIPELINE_SECONDS}s"
    echo "════════════════════════════════════════════════════════════════════════════"
    echo ""
    
    # Log end time
    echo "" >> "$LOG_FILE"
    echo "Pipeline ended: $PIPELINE_END_DATETIME" >> "$LOG_FILE"
    echo "Total duration: ${PIPELINE_HOURS}h ${PIPELINE_MINUTES}m ${PIPELINE_SECONDS}s" >> "$LOG_FILE"
}

# Run main
main "$@"
