"""
Calibration Validation Script
==============================

Analyzes predictions from Script 9 (pos_weight=5.0) against baseline (pos_weight=3.0)
to verify the calibration fix is working correctly.

Metrics:
- Mean prediction value
- Max prediction value
- % above threshold (0.50)
- Distribution statistics
"""

import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

DATASET_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VALIDATION_IMAGES_PATH = f"{DATASET_PATH}/validation_images"
MODELS_PATH = f"{DATASET_PATH}/models"
OUTPUT_CSV = os.path.join(DATASET_PATH, "output", "validation_predictions_pos_weight_5.csv")

# Ensure directories exist
os.makedirs(os.path.join(DATASET_PATH, "output"), exist_ok=True)

# Baseline metrics (pos_weight=3.0)
BASELINE_METRICS = {
    'pos_weight': 3.0,
    'threshold': 0.40,
    'mean': 0.2661,
    'max': 0.3667,
    'above_threshold_pct': 0.0,  # 0% - all below threshold
    'issue': '100% false negatives - predictions too low'
}

# Target metrics (pos_weight=5.0)
TARGET_METRICS = {
    'pos_weight': 5.0,
    'threshold': 0.50,
    'mean_range': (0.40, 0.50),
    'max_min': 0.50,
    'above_threshold_pct_min': 20,  # At least 20% should be above threshold
    'goal': 'Balanced fraud detection without extreme false positives'
}

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def load_predictions():
    """Load predictions from CSV file"""
    if not Path(OUTPUT_CSV).exists():
        print(f"❌ Predictions file not found: {OUTPUT_CSV}")
        return None
    
    df = pd.read_csv(OUTPUT_CSV)
    print(f"✅ Loaded {len(df)} predictions from {OUTPUT_CSV}")
    return df

def compute_metrics(predictions):
    """Compute key metrics from predictions"""
    preds = predictions.values
    
    metrics = {
        'count': len(preds),
        'mean': float(np.mean(preds)),
        'std': float(np.std(preds)),
        'min': float(np.min(preds)),
        'max': float(np.max(preds)),
        'median': float(np.median(preds)),
        'p25': float(np.percentile(preds, 25)),
        'p75': float(np.percentile(preds, 75)),
        'p90': float(np.percentile(preds, 90)),
        'p95': float(np.percentile(preds, 95)),
    }
    
    # Compute % above various thresholds
    for threshold in [0.30, 0.40, 0.50, 0.60, 0.70]:
        above = np.sum(preds > threshold)
        metrics[f'above_{threshold:.2f}_pct'] = float(100 * above / len(preds))
    
    return metrics

def validate_calibration(metrics):
    """Check if calibration improvements are sufficient"""
    print("\n" + "="*80)
    print("CALIBRATION VALIDATION RESULTS")
    print("="*80)
    
    print("\n📊 BASELINE (pos_weight=3.0):")
    print(f"  Mean: {BASELINE_METRICS['mean']:.4f}")
    print(f"  Max: {BASELINE_METRICS['max']:.4f}")
    print(f"  Above 0.50: {BASELINE_METRICS['above_threshold_pct']}%")
    print(f"  Issue: {BASELINE_METRICS['issue']}")
    
    print("\n📊 CURRENT RESULTS (pos_weight=5.0):")
    print(f"  Count: {metrics['count']}")
    print(f"  Mean: {metrics['mean']:.4f}")
    print(f"  Std: {metrics['std']:.4f}")
    print(f"  Min: {metrics['min']:.4f}")
    print(f"  Max: {metrics['max']:.4f}")
    print(f"  Median: {metrics['median']:.4f}")
    print(f"  P25-P75: [{metrics['p25']:.4f}, {metrics['p75']:.4f}]")
    print(f"  Above 0.30: {metrics['above_0.30_pct']:.1f}%")
    print(f"  Above 0.40: {metrics['above_0.40_pct']:.1f}%")
    print(f"  Above 0.50: {metrics['above_0.50_pct']:.1f}%")
    print(f"  Above 0.60: {metrics['above_0.60_pct']:.1f}%")
    print(f"  Above 0.70: {metrics['above_0.70_pct']:.1f}%")
    
    print("\n🎯 TARGET (pos_weight=5.0 Goal):")
    print(f"  Mean range: {TARGET_METRICS['mean_range']}")
    print(f"  Max minimum: >{TARGET_METRICS['max_min']}")
    print(f"  Above threshold: ≥{TARGET_METRICS['above_threshold_pct_min']}%")
    print(f"  Goal: {TARGET_METRICS['goal']}")
    
    # Validation checks
    print("\n✅ VALIDATION CHECKS:")
    checks_passed = 0
    checks_total = 4
    
    # Check 1: Mean improvement
    mean_improved = metrics['mean'] > BASELINE_METRICS['mean']
    check1 = "✓" if mean_improved else "✗"
    print(f"  {check1} Mean improved from {BASELINE_METRICS['mean']:.4f} → {metrics['mean']:.4f}")
    if mean_improved:
        checks_passed += 1
    
    # Check 2: Mean in target range
    mean_in_range = TARGET_METRICS['mean_range'][0] <= metrics['mean'] <= TARGET_METRICS['mean_range'][1]
    check2 = "✓" if mean_in_range else "⚠"
    status2 = "in target range" if mean_in_range else "outside target range"
    print(f"  {check2} Mean {status2} ({metrics['mean']:.4f})")
    if mean_in_range:
        checks_passed += 1
    
    # Check 3: Max above threshold
    max_ok = metrics['max'] > TARGET_METRICS['max_min']
    check3 = "✓" if max_ok else "✗"
    print(f"  {check3} Max {metrics['max']:.4f} > {TARGET_METRICS['max_min']}")
    if max_ok:
        checks_passed += 1
    
    # Check 4: % above threshold
    pct_ok = metrics['above_0.50_pct'] >= TARGET_METRICS['above_threshold_pct_min']
    check4 = "✓" if pct_ok else "⚠"
    pct_status = f"{metrics['above_0.50_pct']:.1f}%" if pct_ok else f"{metrics['above_0.50_pct']:.1f}% (target: ≥{TARGET_METRICS['above_threshold_pct_min']}%)"
    print(f"  {check4} {pct_status} above 0.50 threshold")
    if pct_ok:
        checks_passed += 1
    
    print(f"\n📈 PASSED: {checks_passed}/{checks_total} validation checks")
    
    return checks_passed >= 3  # Pass if at least 3/4 checks pass

def plot_distribution(predictions, metrics):
    """Create distribution plots"""
    output_file = f"{DATASET_PATH}/calibration_validation_plots.png"
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Calibration Validation - pos_weight=5.0 vs pos_weight=3.0', 
                 fontsize=16, fontweight='bold')
    
    preds = predictions.values
    
    # Plot 1: Histogram with baselines
    ax = axes[0, 0]
    ax.hist(preds, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(metrics['mean'], color='red', linestyle='--', linewidth=2, label=f'Mean: {metrics["mean"]:.4f}')
    ax.axvline(BASELINE_METRICS['mean'], color='orange', linestyle='--', linewidth=2, label=f'Baseline: {BASELINE_METRICS["mean"]:.4f}')
    ax.axvline(0.50, color='green', linestyle=':', linewidth=2, label='Threshold: 0.50')
    ax.set_xlabel('Prediction Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Prediction Distribution')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 2: CDF
    ax = axes[0, 1]
    sorted_preds = np.sort(preds)
    cdf = np.arange(1, len(sorted_preds) + 1) / len(sorted_preds)
    ax.plot(sorted_preds, cdf * 100, linewidth=2, label='CDF')
    ax.axvline(0.50, color='green', linestyle=':', linewidth=2, label='Threshold: 0.50')
    ax.fill_between(sorted_preds, 0, 100, where=(sorted_preds > 0.50), 
                     alpha=0.2, color='green', label=f'Above threshold: {metrics["above_0.50_pct"]:.1f}%')
    ax.set_xlabel('Prediction Value')
    ax.set_ylabel('Cumulative Percentage')
    ax.set_title('Cumulative Distribution')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 3: Box plot comparison
    ax = axes[1, 0]
    data_to_plot = [
        preds,
        np.full_like(preds, BASELINE_METRICS['mean']),  # Baseline mean line
    ]
    bp = ax.boxplot([preds], labels=['pos_weight=5.0'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    ax.axhline(BASELINE_METRICS['mean'], color='orange', linestyle='--', linewidth=2, label='Baseline mean')
    ax.axhline(0.50, color='green', linestyle=':', linewidth=2, label='Threshold')
    ax.set_ylabel('Prediction Value')
    ax.set_title('Prediction Box Plot')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    # Plot 4: Comparison metrics
    ax = axes[1, 1]
    ax.axis('off')
    comparison_text = f"""
CALIBRATION IMPROVEMENT SUMMARY

Baseline (pos_weight=3.0):
  Mean:      {BASELINE_METRICS['mean']:.4f}
  Max:       {BASELINE_METRICS['max']:.4f}
  >0.50:     {BASELINE_METRICS['above_threshold_pct']:.1f}%

Current (pos_weight=5.0):
  Mean:      {metrics['mean']:.4f} (↑ {metrics['mean'] - BASELINE_METRICS['mean']:.4f})
  Max:       {metrics['max']:.4f} (↑ {metrics['max'] - BASELINE_METRICS['max']:.4f})
  >0.50:     {metrics['above_0.50_pct']:.1f}% (↑ {metrics['above_0.50_pct'] - BASELINE_METRICS['above_threshold_pct']:.1f}%)

Target (pos_weight=5.0 Goal):
  Mean:      {TARGET_METRICS['mean_range']}
  >0.50:     ≥{TARGET_METRICS['above_threshold_pct_min']}%
    """
    ax.text(0.1, 0.5, comparison_text, fontsize=11, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n📊 Plots saved to: {output_file}")
    plt.close()

def main():
    """Main validation pipeline"""
    print("\n" + "="*80)
    print("CALIBRATION VALIDATION - pos_weight=5.0")
    print("="*80)
    
    # Load predictions
    predictions = load_predictions()
    if predictions is None:
        return
    
    # Compute metrics
    metrics = compute_metrics(predictions['prediction'])
    
    # Validate calibration
    is_valid = validate_calibration(metrics)
    
    # Create plots
    plot_distribution(predictions['prediction'], metrics)
    
    # Save results
    results_file = f"{DATASET_PATH}/calibration_validation_results.json"
    results = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'baseline': BASELINE_METRICS,
        'current': metrics,
        'target': TARGET_METRICS,
        'validation_passed': is_valid,
        'summary': {
            'mean_improvement': metrics['mean'] - BASELINE_METRICS['mean'],
            'max_improvement': metrics['max'] - BASELINE_METRICS['max'],
            'detection_rate_improvement_pct': metrics['above_0.50_pct'] - BASELINE_METRICS['above_threshold_pct'],
        }
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {results_file}")
    
    if is_valid:
        print("\n🎉 CALIBRATION FIX VALIDATED - Ready for final submission!")
        return 0
    else:
        print("\n⚠️  CALIBRATION NEEDS ADJUSTMENT - Consider further tuning")
        return 1

if __name__ == "__main__":
    exit(main())
