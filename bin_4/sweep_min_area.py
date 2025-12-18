#!/usr/bin/env python3
"""
Script 3: Min Area Sweep
Test higher min_area thresholds (500-2000) to reduce FP

Based on error analysis:
- 73/551 FP have <500px detections (13%)
- 165/551 FP have <1000px detections (30%)
- Need to find optimal min_area without hurting TP
"""

import os
import sys
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
import torch
import torch.nn as nn
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from pathlib import Path
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 512
ENCODER = 'timm-efficientnet-b2'
THRESHOLD = 0.35

# Sweep these min_area values
MIN_AREAS = [300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1500, 2000]

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
    def __init__(self, in_channels=3):
        super().__init__()
        self.base = smp.FPN(
            encoder_name=ENCODER,
            encoder_weights=None,
            in_channels=in_channels,
            classes=1,
        )
        self.attention = AttentionGate(1)
    
    def forward(self, x):
        return self.attention(self.base(x))


def load_model(path):
    model = AttentionFPN()
    checkpoint = torch.load(path, map_location=DEVICE)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()
    return model


# ============================================================================
# PREDICTION
# ============================================================================

def get_transform():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def predict_with_tta(models, image):
    """4x TTA with flips + MAX aggregation."""
    transform = get_transform()
    orig_h, orig_w = image.shape[:2]
    
    all_preds = []
    
    for model in models:
        preds = []
        for flip_h in [False, True]:
            for flip_v in [False, True]:
                img = image.copy()
                if flip_h:
                    img = cv2.flip(img, 1)
                if flip_v:
                    img = cv2.flip(img, 0)
                
                transformed = transform(image=img)
                tensor = transformed['image'].unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    pred = torch.sigmoid(model(tensor))
                
                pred = pred.cpu().numpy()[0, 0]
                
                if flip_v:
                    pred = np.flip(pred, axis=0)
                if flip_h:
                    pred = np.flip(pred, axis=1)
                
                preds.append(pred.copy())
        
        model_pred = np.max(np.stack(preds, axis=0), axis=0)
        all_preds.append(model_pred)
    
    final = np.max(np.stack(all_preds, axis=0), axis=0)
    return cv2.resize(final, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)


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
    print("MIN AREA SWEEP")
    print("=" * 60)
    print(f"Threshold: {THRESHOLD}")
    print(f"Min areas to test: {MIN_AREAS}")
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
    print(f"Loaded {len(models)} models\n")
    
    # Get images
    forged_dir = PROJECT_DIR / 'train_images' / 'forged'
    authentic_dir = PROJECT_DIR / 'train_images' / 'authentic'
    
    forged_images = sorted(list(forged_dir.glob('*.png')))
    authentic_images = sorted(list(authentic_dir.glob('*.png')))
    
    print(f"Forged: {len(forged_images)}, Authentic: {len(authentic_images)}\n")
    
    # Pre-compute predictions
    print("Computing predictions...")
    forged_preds = []
    forged_imgs = []
    for img_path in tqdm(forged_images, desc="Forged"):
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pred = predict_with_tta(models, img)
        forged_preds.append(pred)
        forged_imgs.append(img)
    
    authentic_preds = []
    authentic_imgs = []
    for img_path in tqdm(authentic_images, desc="Authentic"):
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pred = predict_with_tta(models, img)
        authentic_preds.append(pred)
        authentic_imgs.append(img)
    
    # Sweep min_area
    print("\n" + "=" * 60)
    print("MIN AREA SWEEP RESULTS")
    print("=" * 60)
    print(f"{'MinArea':<10} {'TP':<10} {'FP':<10} {'Net':<10} {'TP%':<10} {'FP%':<10}")
    print("-" * 60)
    
    best_net = -float('inf')
    best_min_area = 0
    results = []
    
    for min_area in MIN_AREAS:
        tp = 0
        for pred, img in zip(forged_preds, forged_imgs):
            mask = postprocess(pred, img, THRESHOLD, min_area)
            if mask.sum() > 0:
                tp += 1
        
        fp = 0
        for pred, img in zip(authentic_preds, authentic_imgs):
            mask = postprocess(pred, img, THRESHOLD, min_area)
            if mask.sum() > 0:
                fp += 1
        
        net = tp - fp
        tp_pct = tp / len(forged_images) * 100
        fp_pct = fp / len(authentic_images) * 100
        
        results.append((min_area, tp, fp, net, tp_pct, fp_pct))
        print(f"{min_area:<10} {tp:<10} {fp:<10} {net:<10} {tp_pct:<10.1f} {fp_pct:<10.1f}")
        
        if net > best_net:
            best_net = net
            best_min_area = min_area
    
    # Summary
    print("\n" + "=" * 60)
    print("BEST RESULT")
    print("=" * 60)
    
    best_result = [r for r in results if r[0] == best_min_area][0]
    min_area, tp, fp, net, tp_pct, fp_pct = best_result
    
    print(f"Min Area: {min_area}")
    print(f"TP: {tp}/{len(forged_images)} ({tp_pct:.1f}%)")
    print(f"FP: {fp}/{len(authentic_images)} ({fp_pct:.1f}%)")
    print(f"Net Score: {net}")
    
    # Baseline comparison (min_area=300)
    baseline = [r for r in results if r[0] == 300][0]
    _, base_tp, base_fp, base_net, _, _ = baseline
    
    print(f"\nComparison to baseline (min_area=300):")
    print(f"  TP change: {tp - base_tp:+d}")
    print(f"  FP change: {fp - base_fp:+d}")
    print(f"  Net change: {net - base_net:+d}")
    
    # Show tradeoffs
    print("\n" + "=" * 60)
    print("TRADEOFF ANALYSIS")
    print("=" * 60)
    print("Each row shows: how many more FP removed vs TP lost when increasing min_area")
    print()
    
    for i in range(1, len(results)):
        prev = results[i-1]
        curr = results[i]
        tp_lost = prev[1] - curr[1]
        fp_removed = prev[2] - curr[2]
        ratio = fp_removed / tp_lost if tp_lost > 0 else float('inf')
        
        print(f"{prev[0]} -> {curr[0]}: TP lost = {tp_lost}, FP removed = {fp_removed}, "
              f"Ratio = {ratio:.2f} (>1 is good)")


if __name__ == '__main__':
    main()
