#!/usr/bin/env python3
"""
Full Pipeline Runner - Intermediate + Tier 3
==============================================

Runs all intermediate and Tier 3 scripts sequentially with:
- Progress tracking
- Error handling
- Resource monitoring
- Timing information
- Summary reporting

Usage:
    python run_full_pipeline.py              # Run all scripts
    python run_full_pipeline.py --intermediate  # Run only intermediate
    python run_full_pipeline.py --tier3      # Run only Tier 3
    python run_full_pipeline.py --skip-loss  # Skip loss variants
"""

import os
import sys
import subprocess
import time
import json
import argparse
import threading
import queue
from datetime import datetime
from pathlib import Path


class PipelineRunner:
    def __init__(self, project_dir):
        self.project_dir = Path(project_dir)
        self.models_dir = self.project_dir / "models"
        self.logs_dir = self.project_dir / "pipeline_logs"
        self.results = []
        self.start_time = None
        self.end_time = None
        
        # Create directories
        self.models_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        
        # Scripts configuration - all scripts are in the bin/ directory
        self.bin_dir = self.project_dir / "bin"
        
        self.intermediate_scripts = [
            {
                "name": "SegFormer Transformer",
                "script": "script_1_training_segformer.py",
                "expected_gain": "+5-8%",
                "time_estimate": "40-50 min"
            },
            {
                "name": "Loss Variants (Lovász, Focal, Tversky)",
                "script": "script_2_training_loss_variants.py",
                "expected_gain": "+2-4%",
                "time_estimate": "35-40 min"
            },
            {
                "name": "Hyperparameter Tuning",
                "script": "script_3_hyperparameter_tuning.py",
                "expected_gain": "+2-3%",
                "time_estimate": "40-50 min"
            },
            {
                "name": "Self-Supervised Pre-training (SimCLR)",
                "script": "script_4_selfsupervised_pretraining.py",
                "expected_gain": "+3-5%",
                "time_estimate": "50-70 min"
            }
        ]
        
        self.tier3_scripts = [
            {
                "name": "Deep Ensemble (4 members)",
                "script": "script_5_deep_ensemble.py",
                "expected_gain": "+3-5%",
                "time_estimate": "45-60 min"
            }
        ]
        
        self.inference_scripts = [
            {
                "name": "Ensemble Inference",
                "script": "script_6_ensemble_inference.py",
                "expected_gain": "Combined model predictions",
                "time_estimate": "30-45 min"
            }
        ]
    
    def print_header(self, text, width=80):
        """Print formatted header"""
        print("\n" + "="*width)
        print(text.center(width))
        print("="*width + "\n")
    
    def print_section(self, text, width=80):
        """Print formatted section"""
        print("\n" + "-"*width)
        print(f"  {text}")
        print("-"*width + "\n")
    
    def run_script(self, script_info, script_num, total):
        """Run a single script with error handling"""
        script_name = script_info["script"]
        script_path = self.bin_dir / script_name
        
        if not script_path.exists():
            print(f"❌ Script not found: {script_path}")
            return False
        
        print(f"\n[{script_num}/{total}] Running: {script_info['name']}")
        print(f"    Script: {script_name}")
        print(f"    Expected gain: {script_info['expected_gain']}")
        print(f"    Estimated time: {script_info['time_estimate']}")
        print()
        
        start_time_local = datetime.now()
        start = time.time()
        print(f"    START TIME: {start_time_local.strftime('%Y-%m-%d %H:%M:%S')}")
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(self.bin_dir),
                capture_output=False,
                text=True,
                timeout=72000  # 20 hour timeout (increased from 10 hours)
            )
            
            elapsed = time.time() - start
            end_time_local = datetime.now()
            
            if result.returncode == 0:
                print(f"\n✅ COMPLETED: {script_info['name']}")
                print(f"   END TIME: {end_time_local.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"   Time elapsed: {self.format_time(elapsed)}")
                
                self.results.append({
                    "script": script_name,
                    "name": script_info['name'],
                    "status": "SUCCESS",
                    "time": elapsed,
                    "start_time": start_time_local.isoformat(),
                    "end_time": end_time_local.isoformat(),
                    "expected_gain": script_info['expected_gain']
                })
                return True
            else:
                print(f"\n❌ FAILED: {script_info['name']} (exit code: {result.returncode})")
                print(f"   END TIME: {end_time_local.strftime('%Y-%m-%d %H:%M:%S')}")
                self.results.append({
                    "script": script_name,
                    "name": script_info['name'],
                    "status": "FAILED",
                    "time": elapsed,
                    "start_time": start_time_local.isoformat(),
                    "end_time": end_time_local.isoformat(),
                    "expected_gain": script_info['expected_gain']
                })
                return False
                
        except subprocess.TimeoutExpired:
            end_time_local = datetime.now()
            elapsed = time.time() - start
            print(f"\n❌ TIMEOUT: {script_info['name']} (exceeded 10 hours)")
            print(f"   END TIME: {end_time_local.strftime('%Y-%m-%d %H:%M:%S')}")
            self.results.append({
                "script": script_name,
                "name": script_info['name'],
                "status": "TIMEOUT",
                "time": elapsed,
                "start_time": start_time_local.isoformat(),
                "end_time": end_time_local.isoformat(),
                "expected_gain": script_info['expected_gain']
            })
            return False
        except Exception as e:
            end_time_local = datetime.now()
            elapsed = time.time() - start
            print(f"\n❌ ERROR: {script_info['name']}")
            print(f"   Error: {str(e)}")
            print(f"   END TIME: {end_time_local.strftime('%Y-%m-%d %H:%M:%S')}")
            self.results.append({
                "script": script_name,
                "name": script_info['name'],
                "status": "ERROR",
                "time": elapsed,
                "start_time": start_time_local.isoformat(),
                "end_time": end_time_local.isoformat(),
                "expected_gain": script_info['expected_gain']
            })
            return False
    
    def format_time(self, seconds):
        """Format seconds to human-readable time"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        parts = []
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        if secs > 0 or not parts:
            parts.append(f"{secs}s")
        
        return " ".join(parts)
    
    def run_script_async(self, script_info, results_queue):
        """Run a script asynchronously and put result in queue"""
        script_name = script_info["script"]
        script_path = self.bin_dir / script_name
        log_file = self.logs_dir / f"{script_name.replace('.py', '')}.log"
        
        if not script_path.exists():
            print(f"❌ Script not found: {script_path}")
            results_queue.put({
                "script": script_name,
                "name": script_info['name'],
                "status": "NOT_FOUND",
                "time": 0,
                "expected_gain": script_info['expected_gain']
            })
            return
        
        print(f"🚀 STARTING: {script_info['name']}")
        print(f"    Script: {script_name}")
        print(f"    Log: {log_file}")
        sys.stdout.flush()
        
        start_time_local = datetime.now()
        start = time.time()
        
        try:
            with open(log_file, 'w') as f:
                result = subprocess.run(
                    [sys.executable, str(script_path)],
                    cwd=str(self.bin_dir),
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=72000
                )
            
            elapsed = time.time() - start
            end_time_local = datetime.now()
            
            if result.returncode == 0:
                print(f"✅ COMPLETED: {script_info['name']} ({self.format_time(elapsed)})")
                results_queue.put({
                    "script": script_name,
                    "name": script_info['name'],
                    "status": "SUCCESS",
                    "time": elapsed,
                    "start_time": start_time_local.isoformat(),
                    "end_time": end_time_local.isoformat(),
                    "expected_gain": script_info['expected_gain']
                })
            else:
                print(f"❌ FAILED: {script_info['name']} (exit code: {result.returncode})")
                results_queue.put({
                    "script": script_name,
                    "name": script_info['name'],
                    "status": "FAILED",
                    "time": elapsed,
                    "start_time": start_time_local.isoformat(),
                    "end_time": end_time_local.isoformat(),
                    "expected_gain": script_info['expected_gain']
                })
        except Exception as e:
            elapsed = time.time() - start
            print(f"❌ ERROR: {script_info['name']} - {str(e)}")
            results_queue.put({
                "script": script_name,
                "name": script_info['name'],
                "status": "ERROR",
                "time": elapsed,
                "expected_gain": script_info['expected_gain']
            })
        
        sys.stdout.flush()
    
    def print_summary(self):
        """Print execution summary"""
        self.print_header("PIPELINE EXECUTION SUMMARY")
        
        # Statistics
        total = len(self.results)
        successful = sum(1 for r in self.results if r["status"] == "SUCCESS")
        failed = sum(1 for r in self.results if r["status"] != "SUCCESS")
        total_time = sum(r["time"] for r in self.results)
        
        print(f"Total Scripts Run: {total}")
        print(f"Successful: {successful} ✅")
        print(f"Failed: {failed} ❌")
        print(f"Total Time: {self.format_time(total_time)}")
        
        # Pipeline timing
        if self.results:
            first_start = self.results[0].get("start_time", "N/A")
            last_end = self.results[-1].get("end_time", "N/A")
            print(f"\nPipeline Start: {first_start}")
            print(f"Pipeline End:   {last_end}")
        print()
        
        # Detailed results
        self.print_section("DETAILED RESULTS")
        for i, result in enumerate(self.results, 1):
            status_icon = "✅" if result["status"] == "SUCCESS" else "❌"
            print(f"{i}. {status_icon} {result['name']}")
            print(f"   Script: {result['script']}")
            print(f"   Status: {result['status']}")
            print(f"   Time: {self.format_time(result['time'])}")
            if "start_time" in result:
                print(f"   Start: {result['start_time']}")
            if "end_time" in result:
                print(f"   End:   {result['end_time']}")
            print(f"   Expected gain: {result['expected_gain']}")
            print()
        
        # Expected improvements
        self.print_section("EXPECTED IMPROVEMENTS")
        
        if successful > 0:
            intermediate_count = sum(1 for r in self.results[:4] if r["status"] == "SUCCESS")
            tier3_count = sum(1 for r in self.results[4:] if r["status"] == "SUCCESS")
            
            if intermediate_count == 4:
                print("✅ All Intermediate Scripts Completed")
                print("   Expected improvement: +9-14%")
                print("   New ranking: 87-95th percentile")
                print()
            elif intermediate_count > 0:
                print(f"⚠️  {intermediate_count}/4 Intermediate Scripts Completed")
                print()
            
            if tier3_count == 4:
                print("✅ All Tier 3 Scripts Completed")
                print("   Additional improvement: +10-18%")
                print("   Final ranking: 92-98th percentile")
                print("   TOTAL: +49-78% over baseline")
                print()
            elif tier3_count > 0:
                print(f"⚠️  {tier3_count}/4 Tier 3 Scripts Completed")
                print()
        
        # Next steps
        self.print_section("NEXT STEPS")
        print("1. Ensemble predictions created with script_8_deep_ensemble.py")
        print()
        print("2. Submit to Kaggle:")
        print("   Check ranking improvement")
        print()
        
        # Save results
        results_file = self.logs_dir / f"pipeline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "total_scripts": total,
                "successful": successful,
                "failed": failed,
                "total_time": total_time,
                "results": self.results
            }, f, indent=2)
        
        print(f"Results saved: {results_file}")
    
    def run_intermediate(self):
        """Run intermediate scripts with parallel execution
        
        Dependency graph:
        - Scripts 1, 2, 3 run in parallel
        - Script 4 waits for script 3
        """
        self.print_header("PHASE 1: TRAINING SCRIPTS 1-4")
        
        print("Execution plan:")
        print("  • Scripts 1, 2, 3: Run in PARALLEL")
        print("  • Script 4: Waits for Script 3 to complete")
        print()
        print("Scripts:")
        for i, script in enumerate(self.intermediate_scripts, 1):
            print(f"  {i}. {script['name']} ({script['expected_gain']})")
        print()
        sys.stdout.flush()
        
        results_queue = queue.Queue()
        
        # Phase 1a: Run scripts 1, 2, 3 in parallel
        print("=" * 60)
        print("STARTING SCRIPTS 1, 2, 3 IN PARALLEL")
        print("=" * 60)
        sys.stdout.flush()
        
        threads_123 = []
        for script in self.intermediate_scripts[:3]:  # Scripts 1, 2, 3
            t = threading.Thread(target=self.run_script_async, args=(script, results_queue))
            t.start()
            threads_123.append(t)
        
        # Wait for script 3 specifically (needed for script 4)
        # We'll wait for all 3 to complete, but could optimize if needed
        for t in threads_123:
            t.join()
        
        # Collect results from scripts 1, 2, 3
        while not results_queue.empty():
            self.results.append(results_queue.get())
        
        print()
        print("=" * 60)
        print("SCRIPTS 1, 2, 3 COMPLETE - STARTING SCRIPT 4")
        print("=" * 60)
        sys.stdout.flush()
        
        # Phase 1b: Run script 4 (depends on script 3)
        script_4 = self.intermediate_scripts[3]
        t4 = threading.Thread(target=self.run_script_async, args=(script_4, results_queue))
        t4.start()
        t4.join()
        
        # Collect result from script 4
        while not results_queue.empty():
            self.results.append(results_queue.get())
    
    def run_tier3(self):
        """Run Tier 3 scripts (script 5 depends on script 4)"""
        self.print_header("PHASE 2: DEEP ENSEMBLE (SCRIPT 5)")
        
        print("Script 5 requires Script 4 models to be complete")
        print()
        sys.stdout.flush()
        
        results_queue = queue.Queue()
        
        for script in self.tier3_scripts:
            t = threading.Thread(target=self.run_script_async, args=(script, results_queue))
            t.start()
            t.join()
        
        while not results_queue.empty():
            self.results.append(results_queue.get())
    
    def run_inference(self):
        """Run inference scripts (waits for all training scripts 1-5)"""
        self.print_header("PHASE 3: ENSEMBLE INFERENCE (SCRIPT 6)")
        
        print("Script 6 combines predictions from all trained models")
        print("Requires: Scripts 1, 2, 3, 4, 5 to be complete")
        print()
        sys.stdout.flush()
        
        results_queue = queue.Queue()
        
        for script in self.inference_scripts:
            t = threading.Thread(target=self.run_script_async, args=(script, results_queue))
            t.start()
            t.join()
        
        while not results_queue.empty():
            self.results.append(results_queue.get())
    
    def run_full_pipeline(self):
        """Run complete pipeline with parallel execution and dependencies
        
        Dependency graph:
        - Scripts 1, 2, 3: Run in PARALLEL
        - Script 4: Waits for Script 3
        - Script 5: Waits for Script 4
        - Script 6: Waits for Scripts 1, 2, 3, 4, 5
        """
        self.print_header("FULL PIPELINE - PARALLEL EXECUTION")
        
        print("EXECUTION PLAN:")
        print("  ┌─ Script 1 (SegFormer) ─────────────────┐")
        print("  ├─ Script 2 (Loss Variants) ─────────────┼─┐")
        print("  └─ Script 3 (Hyperparameter) ─┬──────────┘ │")
        print("                                ↓            │")
        print("                         Script 4 (SimCLR)   │")
        print("                                ↓            │")
        print("                         Script 5 (Ensemble) │")
        print("                                ↓            │")
        print("                         Script 6 (Inference)←┘")
        print()
        print("Scripts 1, 2, 3 run in PARALLEL for faster execution")
        print()
        
        # Record overall pipeline start time
        self.start_time = datetime.now()
        print(f"PIPELINE START TIME: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("Starting pipeline...\n")
        sys.stdout.flush()
        
        # Phase 1: Scripts 1, 2, 3 (parallel) + Script 4 (after 3)
        self.run_intermediate()
        
        print("\n" + "="*80)
        print("PHASE 1 COMPLETE - Training scripts 1-4 finished".center(80))
        print("="*80)
        print("\nContinuing with Script 5 (Deep Ensemble)...\n")
        
        # Phase 2: Script 5 (depends on script 4)
        self.run_tier3()
        
        print("\n" + "="*80)
        print("PHASE 2 COMPLETE - Script 5 finished".center(80))
        print("="*80)
        print("\nContinuing with ensemble inference...\n")
        
        self.print_section("PHASE 3: ENSEMBLE INFERENCE")
        self.run_inference()
        
        # Record overall pipeline end time
        self.end_time = datetime.now()
        total_pipeline_time = (self.end_time - self.start_time).total_seconds()
        
        print("\n" + "="*80)
        print("PHASE 3 COMPLETE - Ensemble inference finished".center(80))
        print("="*80)
        print(f"\nPIPELINE END TIME: {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"TOTAL PIPELINE TIME: {self.format_time(total_pipeline_time)}")
        print("="*80)
        sys.stdout.flush()
        sys.stderr.flush()


def main():
    parser = argparse.ArgumentParser(
        description="Full Pipeline Runner - Intermediate + Tier 3 Scripts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_full_pipeline.py              # Run full pipeline
  python run_full_pipeline.py --intermediate  # Only intermediate
  python run_full_pipeline.py --tier3      # Only Tier 3
  python run_full_pipeline.py --skip-loss  # Skip loss variants
        """
    )
    
    parser.add_argument(
        "--intermediate",
        action="store_true",
        help="Run only intermediate scripts"
    )
    parser.add_argument(
        "--tier3",
        action="store_true",
        help="Run only Tier 3 scripts"
    )
    parser.add_argument(
        "--skip-loss",
        action="store_true",
        help="Skip loss variants script"
    )
    parser.add_argument(
        "--project-dir",
        default="/home/swatson/work/MachineLearning/kaggle/ScientificImageForgeryDetection",
        help="Path to project directory"
    )
    
    args = parser.parse_args()
    
    # Verify project directory
    if not Path(args.project_dir).exists():
        print(f"❌ Project directory not found: {args.project_dir}")
        sys.exit(1)
    
    runner = PipelineRunner(args.project_dir)
    
    try:
        if args.intermediate:
            runner.run_intermediate()
        elif args.tier3:
            runner.run_tier3()
        else:
            # Default: run full pipeline
            runner.run_full_pipeline()
        
        # Always print summary
        runner.print_summary()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        runner.print_summary()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        runner.print_summary()
        sys.exit(1)


if __name__ == "__main__":
    main()
