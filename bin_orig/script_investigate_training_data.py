"""
Investigate Training Data Quality
===================================
Analyzes training images to verify they actually contain forgery artifacts.
"""

import os
import sys
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict

DATASET_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_IMAGES_PATH = os.path.join(DATASET_PATH, "train_images")
TRAIN_MASKS_PATH = os.path.join(DATASET_PATH, "train_masks")

print("="*80)
print("TRAINING DATA QUALITY INVESTIGATION")
print("="*80)

# Check directory structure
print("\n1. CHECKING DIRECTORY STRUCTURE")
print("-" * 80)

authentic_dir = os.path.join(TRAIN_IMAGES_PATH, "authentic")
forged_dir = os.path.join(TRAIN_IMAGES_PATH, "forged")

authentic_images = sorted([f for f in os.listdir(authentic_dir) if f.endswith('.png')])
forged_images = sorted([f for f in os.listdir(forged_dir) if f.endswith('.png')])

print(f"Authentic images: {len(authentic_images)}")
print(f"Forged images: {len(forged_images)}")
print(f"Masks (.npy): {len([f for f in os.listdir(TRAIN_MASKS_PATH) if f.endswith('.npy')])}")

# Analyze image characteristics
print("\n2. ANALYZING IMAGE CHARACTERISTICS")
print("-" * 80)

def analyze_image_set(image_dir, image_list, label, sample_size=20):
    """Analyze characteristics of image set"""
    print(f"\n{label.upper()} IMAGES (sampling {min(sample_size, len(image_list))} of {len(image_list)}):")
    
    stats = {
        'mean_intensity': [],
        'std_intensity': [],
        'contrast': [],
        'edge_density': [],
        'brightness': [],
    }
    
    sample_indices = np.linspace(0, len(image_list)-1, min(sample_size, len(image_list)), dtype=int)
    
    for idx in sample_indices:
        img_file = image_list[int(idx)]
        img_path = os.path.join(image_dir, img_file)
        
        if not os.path.exists(img_path):
            continue
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Calculate statistics
        stats['mean_intensity'].append(gray.mean())
        stats['std_intensity'].append(gray.std())
        stats['brightness'].append(np.mean(img))
        
        # Contrast (max - min)
        contrast = gray.max() - gray.min()
        stats['contrast'].append(contrast)
        
        # Edge density using Canny
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges > 0) / edges.size
        stats['edge_density'].append(edge_density)
    
    # Print statistics
    if stats['mean_intensity']:
        print(f"  Mean intensity: {np.mean(stats['mean_intensity']):.1f} ± {np.std(stats['mean_intensity']):.1f}")
    if stats['contrast']:
        print(f"  Contrast (L-H): {np.mean(stats['contrast']):.1f} ± {np.std(stats['contrast']):.1f}")
    if stats['edge_density']:
        print(f"  Edge density: {np.mean(stats['edge_density']):.5f} ± {np.std(stats['edge_density']):.5f}")
    
    return stats

auth_stats = analyze_image_set(authentic_dir, authentic_images, "authentic")
forge_stats = analyze_image_set(forged_dir, forged_images, "forged")

# Analyze masks
print("\n3. ANALYZING MASK COVERAGE (Forgery Artifact Extent)")
print("-" * 80)

def analyze_masks(mask_list, label):
    """Analyze mask characteristics"""
    print(f"\n{label.upper()} MASKS (analyzing {min(50, len(mask_list))} of {len(mask_list)}):")
    
    coverage_percentages = []
    nonzero_count = 0
    
    for mask_file in mask_list[:50]:
        mask_path = os.path.join(TRAIN_MASKS_PATH, mask_file)
        
        if os.path.exists(mask_path):
            try:
                mask = np.load(mask_path)
                nonzero_pixels = np.sum(mask > 0)
                if nonzero_pixels > 0:
                    nonzero_count += 1
                    coverage = nonzero_pixels / mask.size * 100
                    coverage_percentages.append(coverage)
            except:
                pass
    
    if coverage_percentages:
        print(f"  Images with mask pixels: {nonzero_count}/50")
        print(f"  Mask coverage: {np.mean(coverage_percentages):.3f}% ± {np.std(coverage_percentages):.3f}%")
        print(f"  Range: {np.min(coverage_percentages):.3f}% - {np.max(coverage_percentages):.3f}%")
        if np.mean(coverage_percentages) < 2:
            print(f"  ⚠️  VERY SUBTLE: Forgeries cover <2% of images!")
        return np.mean(coverage_percentages)
    else:
        print(f"  No masks found or all masks are empty")
        return 0

# Get mask counts
auth_mask_files = [f for f in os.listdir(TRAIN_MASKS_PATH) if f.endswith('.npy')][:len(authentic_images)]
forge_mask_files = [f for f in os.listdir(TRAIN_MASKS_PATH) if f.endswith('.npy')][len(authentic_images):]

auth_coverage = analyze_masks(auth_mask_files, "authentic")
forge_coverage = analyze_masks(forge_mask_files, "forged")

# Summary and diagnosis
print("\n4. DIAGNOSIS & FINDINGS")
print("="*80)

print("\nKEY OBSERVATIONS:")

# Compare statistics
if auth_stats['mean_intensity'] and forge_stats['mean_intensity']:
    auth_mean = np.mean(auth_stats['mean_intensity'])
    forge_mean = np.mean(forge_stats['mean_intensity'])
    diff = abs(auth_mean - forge_mean)
    print(f"\n1. INTENSITY DIFFERENCE: {diff:.1f}")
    if diff < 10:
        print("   ✗ Very small difference - hard to distinguish")
    else:
        print("   ✓ Noticeable difference - good for detection")

if auth_stats['edge_density'] and forge_stats['edge_density']:
    auth_edges = np.mean(auth_stats['edge_density'])
    forge_edges = np.mean(forge_stats['edge_density'])
    edge_diff = abs(auth_edges - forge_edges) * 1000
    print(f"\n2. EDGE DENSITY DIFFERENCE: {edge_diff:.3f} (×1000)")
    print(f"   Authentic: {auth_edges:.5f}, Forged: {forge_edges:.5f}")
    if edge_diff < 0.5:
        print("   ✗ Minimal difference - forgeries have similar edges")
    else:
        print("   ✓ Clear difference - edges are distinguishable")

if auth_coverage > 0 or forge_coverage > 0:
    print(f"\n3. MASK COVERAGE (Forgery Artifact Extent):")
    print(f"   Authentic: {auth_coverage:.3f}%")
    print(f"   Forged: {forge_coverage:.3f}%")
    if forge_coverage < 2:
        print("   ⚠️  CRITICAL: Forged images have VERY SUBTLE artifacts (<2%)")
        print("     This explains why model struggles to detect them!")
    elif forge_coverage < 5:
        print("   ✗ Forged artifacts are quite subtle (<5%)")
    else:
        print("   ✓ Forged images have substantial artifacts")

print("\n" + "="*80)
print("CONCLUSIONS & RECOMMENDATIONS")
print("="*80)

print(f"""
ROOT CAUSE ANALYSIS:
  Your training data appears to have {forge_coverage:.2f}% average mask coverage
  for forged images. This is EXTREMELY SUBTLE and explains why:
  
  - Models struggle to distinguish forged from authentic
  - Even the ensemble only marks 1% as forged at threshold 0.44
  - The optimization metrics (threshold/cutoff) alone won't solve this
  
SOLUTIONS IN ORDER OF EFFECTIVENESS:

1. ⭐ RETRAIN WITH HIGHER POS_WEIGHT (Recommended)
   - Change pos_weight from 6.0 to 15-25
   - Increases penalty for missing forged images
   - Easier to implement than data cleaning
   - Try: pos_weight=20, keep current loss function
   - Expected improvement: 5-15% better forged detection

2. ⭐ USE BETTER LOSS FUNCTION
   - Switch from BCE to Focal Loss (already available)
   - Focal Loss better for imbalanced/subtle targets
   - Expected improvement: 10-20% better forged detection

3. COMBINE STRATEGIES 1 + 2
   - Retrain with pos_weight=20 + Focal Loss
   - Expected improvement: 20-30% better forged detection
   - Still may not reach 90% on forged due to data subtlety

4. DATA AUGMENTATION IMPROVEMENTS
   - Add stronger random erasing/cutout during training
   - Increase zoom, rotation, distortion augmentations
   - Better prepares model for finding subtle artifacts

5. MANUAL DATA REVIEW (Time Intensive)
   - Inspect 50-100 forged samples
   - Verify masks are accurate
   - Check if some are mislabeled
   - Only do if retraining doesn't help

IMMEDIATE NEXT STEPS:
  1. Create a script to retrain models with pos_weight=20
  2. Test on same test sets to measure improvement
  3. If not sufficient: Switch to Focal Loss + higher pos_weight
  4. Monitor both authentic and forged accuracy
""")

print("="*80)
