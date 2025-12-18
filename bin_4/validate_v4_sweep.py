#!/usr/bin/env python3
"""
V4 Ensemble Validation with:
1. Threshold sweep (multiple threshold/min_area combinations)
2. Multi-scale inference (512px + 768px)
3. More TTA (8x with rotations)
"""

import os
import sys
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from pathlib import Path
from tqdm import tqdm
from itertools import product

# ============================================================================
# CONFIGURATION
# ============================================================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ENCODER = 'timm-efficientnet-b2'

# Multi-scale sizes
SCALES = [768]  # Try 768px only

# Threshold sweep parameters
THRESHOLDS = [0.25, 0.30, 0.35, 0.40, 0.45]
MIN_AREAS = [200, 300, 400, 500]

# V4 model names
V4_MODELS = [
    'highres_no_ela_v4_best.pth',
    'hard_negative_v4_best.pth',
    'high_recall_v4_best.pth',
    'enhanced_aug_v4_best.pth',
]

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent

# ============================================================================
# MODEL
# ============================================================================

class AttentionGate(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        hidden1 = max(8, in_channels // 2)
        hidden2 = max(4, in_channels // 4)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden1, kernel_size=1),
            nn.BatchNorm2d(hidden1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden1, hidden2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden2, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.conv(x)


class AttentionFPN(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        self.attention = AttentionGate(1)
    
    def forward(self, x):
        return self.attention(self.base(x))


def load_model(path):
    base = smp.FPN(
        encoder_name=ENCODER,
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    model = AttentionFPN(base)
    checkpoint = torch.load(path, map_location=DEVICE)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()
    return model


# ============================================================================
# MULTI-SCALE + 8x TTA INFERENCE
# ============================================================================

def get_transform(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def apply_tta_8x(model, image, img_size, device):
    """Apply 8x TTA: 4 flips x 2 (original + 90° rotation)."""
    transform = get_transform(img_size)
    
    predictions = []
    
    # Original orientation
    for flip_h in [False, True]:
        for flip_v in [False, True]:
            img = image.copy()
            if flip_h:
                img = cv2.flip(img, 1)
            if flip_v:
                img = cv2.flip(img, 0)
            
            transformed = transform(image=img)
            tensor = transformed['image'].unsqueeze(0).to(device)
            
            with torch.no_grad():
                pred = torch.sigmoid(model(tensor))
            
            pred = pred.cpu().numpy()[0, 0]
            
            # Undo flips
            if flip_v:
                pred = np.flip(pred, axis=0)
            if flip_h:
                pred = np.flip(pred, axis=1)
            
            predictions.append(pred.copy())
    
    # 90° rotation
    img_rot = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    for flip_h in [False, True]:
        for flip_v in [False, True]:
            img = img_rot.copy()
            if flip_h:
                img = cv2.flip(img, 1)
            if flip_v:
                img = cv2.flip(img, 0)
            
            transformed = transform(image=img)
            tensor = transformed['image'].unsqueeze(0).to(device)
            
            with torch.no_grad():
                pred = torch.sigmoid(model(tensor))
            
            pred = pred.cpu().numpy()[0, 0]
            
            # Undo flips
            if flip_v:
                pred = np.flip(pred, axis=0)
            if flip_h:
                pred = np.flip(pred, axis=1)
            
            # Undo rotation
            pred = cv2.rotate(pred.copy(), cv2.ROTATE_90_COUNTERCLOCKWISE)
            predictions.append(pred)
    
    # MEAN aggregation (more conservative)
    return np.mean(np.stack(predictions, axis=0), axis=0)


def predict_multiscale_tta(models, image, device):
    """Multi-scale + 8x TTA + ensemble prediction."""
    orig_h, orig_w = image.shape[:2]
    all_preds = []
    
    for scale in SCALES:
        for model in models:
            pred = apply_tta_8x(model, image, scale, device)
            # Resize to original size
            pred = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            all_preds.append(pred)
    
    # MEAN across all scales and models (more conservative)
    return np.mean(np.stack(all_preds, axis=0), axis=0)


# ============================================================================
# POSTPROCESSING
# ============================================================================

def postprocess(mask, image, threshold, min_area):
    """Postprocess with brightness-based adaptive threshold."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    brightness = np.mean(gray) / 255.0
    
    if brightness < 0.3:
        adaptive_thresh = threshold * 0.8
    elif brightness > 0.7:
        adaptive_thresh = threshold * 1.2
    else:
        adaptive_thresh = threshold
    
    binary = (mask > adaptive_thresh).astype(np.uint8)
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    clean = np.zeros_like(binary)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 1
    
    return clean


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("V4 SWEEP: Threshold + Multi-scale + 8x TTA")
    print("=" * 60)
    print(f"Scales: {SCALES}")
    print(f"Thresholds: {THRESHOLDS}")
    print(f"Min areas: {MIN_AREAS}")
    print(f"TTA: 8x (4 flips × 2 orientations)")
    print(f"Device: {DEVICE}")
    print()
    
    # Load models
    model_dir = PROJECT_DIR / 'models'
    models = []
    for name in V4_MODELS:
        path = model_dir / name
        if path.exists():
            models.append(load_model(str(path)))
            print(f"  ✓ {name}")
    print(f"Loaded {len(models)} models")
    
    # Get images
    forged_dir = PROJECT_DIR / 'train_images' / 'forged'
    authentic_dir = PROJECT_DIR / 'train_images' / 'authentic'
    forged_images = sorted(list(forged_dir.glob('*.png')))
    authentic_images = sorted(list(authentic_dir.glob('*.png')))
    print(f"\nForged: {len(forged_images)}, Authentic: {len(authentic_images)}")
    
    # Pre-compute predictions (most expensive part)
    print("\n" + "=" * 60)
    print("Computing predictions (multi-scale + 8x TTA)...")
    print("=" * 60)
    
    forged_preds = []
    forged_imgs = []
    for img_path in tqdm(forged_images, desc="Forged"):
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pred = predict_multiscale_tta(models, img, DEVICE)
        forged_preds.append(pred)
        forged_imgs.append(img)
    
    authentic_preds = []
    authentic_imgs = []
    for img_path in tqdm(authentic_images, desc="Authentic"):
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pred = predict_multiscale_tta(models, img, DEVICE)
        authentic_preds.append(pred)
        authentic_imgs.append(img)
    
    # Sweep through thresholds and min_areas
    print("\n" + "=" * 60)
    print("Threshold Sweep Results")
    print("=" * 60)
    print(f"{'Thresh':<8} {'MinArea':<8} {'TP':<8} {'FP':<8} {'Net':<8}")
    print("-" * 40)
    
    best_net = -float('inf')
    best_params = None
    results = []
    
    for thresh, min_area in product(THRESHOLDS, MIN_AREAS):
        tp = 0
        for pred, img in zip(forged_preds, forged_imgs):
            mask = postprocess(pred, img, thresh, min_area)
            if mask.sum() > 0:
                tp += 1
        
        fp = 0
        for pred, img in zip(authentic_preds, authentic_imgs):
            mask = postprocess(pred, img, thresh, min_area)
            if mask.sum() > 0:
                fp += 1
        
        net = tp - fp
        results.append((thresh, min_area, tp, fp, net))
        print(f"{thresh:<8.2f} {min_area:<8} {tp:<8} {fp:<8} {net:<8}")
        
        if net > best_net:
            best_net = net
            best_params = (thresh, min_area, tp, fp)
    
    # Summary
    print("\n" + "=" * 60)
    print("BEST RESULT")
    print("=" * 60)
    thresh, min_area, tp, fp = best_params
    print(f"Threshold: {thresh}, Min Area: {min_area}")
    print(f"TP: {tp}/{len(forged_images)} ({tp/len(forged_images)*100:.1f}%)")
    print(f"FP: {fp}/{len(authentic_images)} ({fp/len(authentic_images)*100:.1f}%)")
    print(f"Net Score: {best_net}")
    print()
    print("COMPARISON TO BASELINE (V4 @ thresh=0.35, min_area=300, 4x TTA):")
    print(f"  Baseline: 2235 TP, 207 FP, Net 2028")
    print(f"  New:      {tp} TP, {fp} FP, Net {best_net}")
    print(f"  Improvement: {best_net - 2028:+d}")


if __name__ == '__main__':
    main()
