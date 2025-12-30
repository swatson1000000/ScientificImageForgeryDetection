#!/usr/bin/env python3
"""
Complete Training Pipeline with Hard Negative Mining

This script orchestrates the full training workflow:

    Phase 1: Initial Training
        - Train binary classifier (for fast filtering)
        - Train segmentation ensemble (without hard negatives)
    
    Phase 2: Hard Negative Mining  
        - Run inference on authentic test images
        - Identify false positives (authentic images incorrectly flagged as forged)
        - Save hard negative IDs for retraining
    
    Phase 3: Retraining with Hard Negatives
        - Retrain segmentation models with hard negatives as additional training data
        - This teaches the model to not flag these tricky authentic images
    
    Phase 4: Inference
        - Run two-pass pipeline on test images
        - Generate submission file

Usage:
    python full_pipeline.py                     # Run full pipeline
    python full_pipeline.py --skip-initial      # Skip initial training
    python full_pipeline.py --phase 2           # Run only phase 2
    python full_pipeline.py --inference-only    # Only run inference
"""

import os
import sys
import subprocess
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
MODELS_DIR = PROJECT_DIR / 'models'
LOG_DIR = SCRIPT_DIR / 'logs'

# Default thresholds for hard negative mining
DEFAULT_CLASSIFIER_THRESHOLD = 0.25
DEFAULT_SEG_THRESHOLD = 0.35
DEFAULT_MIN_AREA = 300


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

class PipelineLogger:
    """Logger that writes to both console and file."""
    
    def __init__(self, log_file: Optional[Path] = None):
        self.log_file = log_file
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
    
    def log(self, msg: str, level: str = "INFO"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"[{timestamp}] [{level}] {msg}"
        print(formatted)
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(formatted + "\n")
    
    def section(self, title: str):
        separator = "=" * 70
        self.log("")
        self.log(separator)
        self.log(f" {title}")
        self.log(separator)
    
    def error(self, msg: str):
        self.log(msg, "ERROR")
    
    def warning(self, msg: str):
        self.log(msg, "WARNING")


def run_script(script_path: Path, args: List[str] = None, logger: PipelineLogger = None) -> bool:
    """Run a Python script and return success status."""
    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)
    
    if logger:
        logger.log(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_DIR),
            capture_output=False,  # Show output in real-time
            text=True
        )
        return result.returncode == 0
    except Exception as e:
        if logger:
            logger.error(f"Failed to run {script_path.name}: {e}")
        return False


def check_models_exist(model_names: List[str]) -> List[str]:
    """Check which models exist and return list of missing ones."""
    missing = []
    for name in model_names:
        if not (MODELS_DIR / name).exists():
            missing.append(name)
    return missing


# ============================================================================
# PIPELINE PHASES
# ============================================================================

def phase1_initial_training(logger: PipelineLogger, skip_classifier: bool = False, skip_segmentation: bool = False) -> bool:
    """
    Phase 1: Initial model training.
    
    Trains (in order):
    1. Segmentation ensemble (script_1 or script_3) - needed for hard negative mining
    2. Binary classifier (script_2)
    """
    logger.section("PHASE 1: INITIAL MODEL TRAINING")
    
    # Step 1a: Train segmentation models FIRST (needed for hard negative mining)
    if not skip_segmentation:
        logger.log("Training initial segmentation models...")
        
        try:
            # Try v4 first, fall back to v3
            seg_script = SCRIPT_DIR / 'script_3_train_v4.py'
            if not seg_script.exists():
                seg_script = SCRIPT_DIR / 'script_1_train_v3.py'
            
            if not seg_script.exists():
                logger.error("No segmentation training script found!")
                return False
            
            # Pass --no-hard-negatives for initial training
            if not run_script(seg_script, args=['--no-hard-negatives'], logger=logger):
                logger.error("Segmentation model training failed!")
                return False
            
            logger.log("Initial segmentation training complete.")
            
        except Exception as e:
            logger.error(f"Error during segmentation training: {e}")
            return False
    else:
        logger.log("Skipping segmentation training (already exists)")
    
    # Step 1b: Train binary classifier SECOND
    if not skip_classifier:
        logger.log("Training binary classifier...")
        classifier_script = SCRIPT_DIR / 'script_2_train_binary_classifier.py'
        
        if not classifier_script.exists():
            logger.error(f"Classifier script not found: {classifier_script}")
            return False
        
        # For initial training, skip hard negatives in classifier too
        args = ['--no-hard-negatives']
        
        if not run_script(classifier_script, args=args, logger=logger):
            logger.error("Binary classifier training failed!")
            return False
        
        if not (MODELS_DIR / 'binary_classifier_best.pth').exists():
            logger.error("Binary classifier model not created!")
            return False
        
        logger.log("Binary classifier training complete.")
    else:
        logger.log("Skipping classifier training (already exists)")
    
    return True


def phase2_hard_negative_mining(
    logger: PipelineLogger,
    authentic_dir: Optional[Path] = None,
    classifier_threshold: float = DEFAULT_CLASSIFIER_THRESHOLD,
    seg_threshold: float = DEFAULT_SEG_THRESHOLD,
    min_area: int = DEFAULT_MIN_AREA
) -> bool:
    """
    Phase 2: Hard negative mining.
    
    Runs inference on known-authentic images to find false positives.
    These become "hard negatives" for retraining.
    """
    logger.section("PHASE 2: HARD NEGATIVE MINING")
    
    if authentic_dir is None:
        authentic_dir = PROJECT_DIR / 'test_authentic_100'
    
    if not authentic_dir.exists():
        logger.error(f"Authentic images directory not found: {authentic_dir}")
        logger.log("Skipping hard negative mining...")
        return True  # Not fatal, can continue without
    
    # Check required models exist
    required_models = ['binary_classifier_best.pth']
    seg_models = ['highres_no_ela_v4_best.pth', 'hard_negative_v4_best.pth', 
                  'high_recall_v4_best.pth', 'enhanced_aug_v4_best.pth']
    
    # Check if we have v4 models, otherwise try v3
    has_v4 = all((MODELS_DIR / m).exists() for m in seg_models)
    if not has_v4:
        seg_models = [m.replace('v4', 'v3') for m in seg_models]
    
    missing = check_models_exist(required_models + seg_models)
    if missing:
        logger.warning(f"Missing models for mining: {missing}")
        logger.log("Will proceed with available models...")
    
    # Run hard negative finder
    mining_script = SCRIPT_DIR / 'find_hard_negatives.py'
    if not mining_script.exists():
        logger.error(f"Mining script not found: {mining_script}")
        return False
    
    args = [
        '--authentic-dir', str(authentic_dir),
        '--output', 'hard_negative_ids.txt',
        '--classifier-threshold', str(classifier_threshold),
        '--seg-threshold', str(seg_threshold),
        '--min-area', str(min_area)
    ]
    
    logger.log(f"Mining hard negatives from {authentic_dir}...")
    if not run_script(mining_script, args=args, logger=logger):
        logger.warning("Hard negative mining had issues, but continuing...")
    
    # Check results
    hn_file = PROJECT_DIR / 'hard_negative_ids.txt'
    if hn_file.exists():
        with open(hn_file, 'r') as f:
            count = sum(1 for line in f if line.strip())
        logger.log(f"Found {count} hard negatives")
    else:
        logger.warning("No hard negatives file created")
        # Create empty file so training doesn't fail
        hn_file.touch()
    
    return True


def phase3_retrain_with_hard_negatives(logger: PipelineLogger) -> bool:
    """
    Phase 3: Retrain models with hard negatives included.
    
    Retrains (in order):
    1. Segmentation models (with hard negatives)
    2. Binary classifier (with hard negatives)
    """
    logger.section("PHASE 3: RETRAINING WITH HARD NEGATIVES")
    
    hn_file = PROJECT_DIR / 'hard_negative_ids.txt'
    if hn_file.exists():
        with open(hn_file, 'r') as f:
            count = sum(1 for line in f if line.strip())
        logger.log(f"Retraining with {count} hard negatives...")
    else:
        logger.log("No hard negatives file found, using defaults...")
    
    # Step 3a: Retrain segmentation models (WITH hard negatives)
    seg_script = SCRIPT_DIR / 'script_3_train_v4.py'
    if not seg_script.exists():
        seg_script = SCRIPT_DIR / 'script_1_train_v3.py'
    
    if not seg_script.exists():
        logger.error("No segmentation training script found!")
        return False
    
    logger.log(f"Retraining segmentation: {seg_script.name}...")
    if not run_script(seg_script, logger=logger):
        logger.error("Segmentation retraining failed!")
        return False
    
    logger.log("Segmentation retraining complete.")
    
    # Step 3b: Retrain classifier (WITH hard negatives)
    classifier_script = SCRIPT_DIR / 'script_2_train_binary_classifier.py'
    if classifier_script.exists():
        logger.log("Retraining binary classifier with hard negatives...")
        # Don't pass --no-hard-negatives, so it uses them
        if not run_script(classifier_script, logger=logger):
            logger.error("Classifier retraining failed!")
            return False
        logger.log("Classifier retraining complete.")
    else:
        logger.warning("Classifier script not found, skipping classifier retraining")
    
    logger.log("All retraining complete.")
    return True


def phase4_inference(
    logger: PipelineLogger,
    test_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None
) -> bool:
    """
    Phase 4: Run inference and generate submission.
    """
    logger.section("PHASE 4: INFERENCE")
    
    if test_dir is None:
        test_dir = PROJECT_DIR / 'validation_images'
    
    if output_dir is None:
        output_dir = PROJECT_DIR / 'predictions'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use two-pass pipeline
    pipeline_script = SCRIPT_DIR / 'two_pass_pipeline.py'
    
    if not pipeline_script.exists():
        # Fall back to original submission script
        pipeline_script = SCRIPT_DIR / 'script_4_two_stage_submission.py'
    
    if not pipeline_script.exists():
        logger.error("No inference script found!")
        return False
    
    if pipeline_script.name == 'two_pass_pipeline.py':
        args = [
            '--input', str(test_dir),
            '--output', str(output_dir),
            '--csv', str(PROJECT_DIR / 'submission.csv'),
            '--json', str(output_dir / 'results.json')
        ]
    else:
        args = []  # Original script may not need args
    
    logger.log(f"Running inference with {pipeline_script.name}...")
    if not run_script(pipeline_script, args=args, logger=logger):
        logger.error("Inference failed!")
        return False
    
    logger.log("Inference complete.")
    
    # Check outputs
    submission_file = PROJECT_DIR / 'submission.csv'
    if submission_file.exists():
        with open(submission_file, 'r') as f:
            line_count = sum(1 for _ in f) - 1  # Subtract header
        logger.log(f"Generated submission with {line_count} predictions")
    
    return True


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Complete Training Pipeline with Hard Negative Mining',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--phase', type=int, choices=[1, 2, 3, 4],
                        help='Run only specific phase (1-4)')
    parser.add_argument('--skip-initial', action='store_true',
                        help='Skip phase 1 (initial training)')
    parser.add_argument('--skip-mining', action='store_true',
                        help='Skip phase 2 (hard negative mining)')
    parser.add_argument('--skip-retrain', action='store_true',
                        help='Skip phase 3 (retraining)')
    parser.add_argument('--inference-only', action='store_true',
                        help='Only run phase 4 (inference)')
    
    # Mining options
    parser.add_argument('--authentic-dir', type=str, default=None,
                        help='Directory with authentic images for mining')
    parser.add_argument('--classifier-threshold', type=float, 
                        default=DEFAULT_CLASSIFIER_THRESHOLD)
    parser.add_argument('--seg-threshold', type=float,
                        default=DEFAULT_SEG_THRESHOLD)
    parser.add_argument('--min-area', type=int,
                        default=DEFAULT_MIN_AREA)
    
    # Inference options
    parser.add_argument('--test-dir', type=str, default=None,
                        help='Test images directory')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for predictions')
    
    args = parser.parse_args()
    
    # Setup logging
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f'full_pipeline_{timestamp}.log'
    logger = PipelineLogger(log_file)
    
    logger.section("FULL TRAINING PIPELINE WITH HARD NEGATIVE MINING")
    logger.log(f"Project directory: {PROJECT_DIR}")
    logger.log(f"Log file: {log_file}")
    logger.log("")
    
    success = True
    
    # Determine which phases to run
    run_phase = {1: True, 2: True, 3: True, 4: True}
    
    if args.phase:
        run_phase = {i: (i == args.phase) for i in range(1, 5)}
    elif args.inference_only:
        run_phase = {1: False, 2: False, 3: False, 4: True}
    else:
        if args.skip_initial:
            run_phase[1] = False
        if args.skip_mining:
            run_phase[2] = False
        if args.skip_retrain:
            run_phase[3] = False
    
    # Phase 1: Initial training
    if run_phase[1]:
        skip_classifier = (MODELS_DIR / 'binary_classifier_best.pth').exists()
        if not phase1_initial_training(logger, skip_classifier=skip_classifier):
            logger.error("Phase 1 failed!")
            success = False
            if not args.phase:  # Stop if running full pipeline
                sys.exit(1)
    
    # Phase 2: Hard negative mining
    if run_phase[2] and success:
        authentic_dir = Path(args.authentic_dir) if args.authentic_dir else None
        if not phase2_hard_negative_mining(
            logger,
            authentic_dir=authentic_dir,
            classifier_threshold=args.classifier_threshold,
            seg_threshold=args.seg_threshold,
            min_area=args.min_area
        ):
            logger.warning("Phase 2 had issues, continuing...")
    
    # Phase 3: Retrain with hard negatives
    if run_phase[3] and success:
        if not phase3_retrain_with_hard_negatives(logger):
            logger.error("Phase 3 failed!")
            success = False
            if not args.phase:
                sys.exit(1)
    
    # Phase 4: Inference
    if run_phase[4]:
        test_dir = Path(args.test_dir) if args.test_dir else None
        output_dir = Path(args.output_dir) if args.output_dir else None
        if not phase4_inference(logger, test_dir=test_dir, output_dir=output_dir):
            logger.error("Phase 4 failed!")
            success = False
    
    # Summary
    logger.section("PIPELINE COMPLETE")
    
    if success:
        logger.log("All phases completed successfully!")
    else:
        logger.log("Pipeline completed with some errors")
    
    logger.log(f"Models: {MODELS_DIR}")
    logger.log(f"Predictions: {PROJECT_DIR / 'predictions'}")
    logger.log(f"Submission: {PROJECT_DIR / 'submission.csv'}")
    logger.log(f"Logs: {log_file}")
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
