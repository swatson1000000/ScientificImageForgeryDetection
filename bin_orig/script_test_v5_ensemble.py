#!/usr/bin/env python3
"""
Test V5 ensemble vs V4 ensemble on training data
Quick validation to see if hard negative mining helped or hurt
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import cv2
import segmentation_models_pytorch as smp
from pathlib import Path
import pandas as pd

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 512

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
MODELS_DIR = PROJECT_DIR / 'models'
TRAIN_IMAGES = PROJECT_DIR / 'train_images'
TRAIN_MASKS = PROJECT_DIR / 'train_masks'

# V4 ensemble (original - 4 models)
V4_MODELS = [
    'highres_no_ela_v4_best.pth',
    'hard_negative_v4_best.pth',
    'high_recall_v4_best.pth',
    'enhanced_aug_v4_best.pth'
]

# V5 ensemble (5 models - adds FP-aware model)
V5_MODELS = [
    'highres_no_ela_v4_best.pth',
    'hard_negative_v4_best.pth',
    'high_recall_v4_best.pth',
    'enhanced_aug_v4_best.pth',
    'highres_no_ela_v5_best.pth',  # Fine-tuned on 198 FPs - adds FP awareness
]


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
        features = self.base(x)
        return self.attention(features)


def load_model(model_path):
    """Load a trained model."""
    base_model = smp.FPN(
        encoder_name='timm-efficientnet-b2',
        encoder_weights=None,
        in_channels=3,
        classes=1
    )
    model = AttentionFPN(base_model).to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model


def load_ensemble(model_names):
    """Load multiple models for ensemble."""
    models = []
    for name in model_names:
        path = MODELS_DIR / name
        if path.exists():
            models.append(load_model(str(path)))
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name} not found")
    return models


def predict_with_tta(models, image_tensor):
    """Ensemble prediction with 4x TTA (identity, hflip, vflip, rot180)."""
    batch = image_tensor.unsqueeze(0).to(DEVICE)
    preds = []
    
    with torch.no_grad():
        for model in models:
            # Identity
            p0 = torch.sigmoid(model(batch))
            # Horizontal flip
            p1 = torch.sigmoid(model(torch.flip(batch, [3])))
            p1 = torch.flip(p1, [3])
            # Vertical flip
            p2 = torch.sigmoid(model(torch.flip(batch, [2])))
            p2 = torch.flip(p2, [2])
            # 180 rotation
            p3 = torch.sigmoid(model(torch.flip(batch, [2, 3])))
            p3 = torch.flip(p3, [2, 3])
            
            avg = (p0 + p1 + p2 + p3) / 4.0
            preds.append(avg)
    
    # Ensemble average
    ensemble = torch.stack(preds).mean(dim=0)
    return ensemble.squeeze().cpu().numpy()


def preprocess_image(img_path):
    """Load and preprocess image."""
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    return torch.from_numpy(img)


def evaluate_on_forged(models, threshold=0.35, min_area=300):
    """Evaluate TP rate on forged images."""
    forged_dir = TRAIN_IMAGES / 'forged'
    forged_images = sorted(list(forged_dir.glob('*.png')))
    
    tp_count = 0
    total = 0
    
    for i, img_path in enumerate(forged_images):
        if i % 500 == 0:
            print(f"  Forged: {i}/{len(forged_images)}")
        image_tensor = preprocess_image(img_path)
        if image_tensor is None:
            continue
        
        prob = predict_with_tta(models, image_tensor)
        binary = (prob >= threshold).astype(np.uint8)
        
        # Check if any significant region
        from scipy import ndimage
        labeled, num_features = ndimage.label(binary)
        has_detection = False
        for i in range(1, num_features + 1):
            if np.sum(labeled == i) >= min_area:
                has_detection = True
                break
        
        if has_detection:
            tp_count += 1
        total += 1
    
    return tp_count, total


def evaluate_on_authentic(models, threshold=0.35, min_area=300):
    """Evaluate FP rate on authentic images."""
    auth_dir = TRAIN_IMAGES / 'authentic'
    auth_images = sorted(list(auth_dir.glob('*.png')))
    
    fp_count = 0
    total = 0
    
    for i, img_path in enumerate(auth_images):
        if i % 500 == 0:
            print(f"  Authentic: {i}/{len(auth_images)}")
        image_tensor = preprocess_image(img_path)
        if image_tensor is None:
            continue
        
        prob = predict_with_tta(models, image_tensor)
        binary = (prob >= threshold).astype(np.uint8)
        
        # Check if any significant region
        from scipy import ndimage
        labeled, num_features = ndimage.label(binary)
        has_detection = False
        for i in range(1, num_features + 1):
            if np.sum(labeled == i) >= min_area:
                has_detection = True
                break
        
        if has_detection:
            fp_count += 1
        total += 1
    
    return fp_count, total


def main():
    print("=" * 60)
    print("V5 ENSEMBLE TEST (with hard negative training)")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Threshold: 0.35, Min area: 300")
    print()
    
    # Load V4 ensemble
    print("Loading V4 ensemble...")
    v4_models = load_ensemble(V4_MODELS)
    print(f"Loaded {len(v4_models)} V4 models")
    print()
    
    # Load V5 ensemble  
    print("Loading V5 ensemble...")
    v5_models = load_ensemble(V5_MODELS)
    print(f"Loaded {len(v5_models)} V5 models")
    print()
    
    # Test V4
    print("-" * 60)
    print("Testing V4 ensemble...")
    tp4, total_forged = evaluate_on_forged(v4_models, threshold=0.35, min_area=300)
    fp4, total_auth = evaluate_on_authentic(v4_models, threshold=0.35, min_area=300)
    net4 = tp4 - fp4
    print(f"V4: TP={tp4}/{total_forged} ({tp4/total_forged*100:.1f}%), FP={fp4}/{total_auth} ({fp4/total_auth*100:.1f}%), Net={net4}")
    print()
    
    # Test V5
    print("-" * 60)
    print("Testing V5 ensemble...")
    tp5, _ = evaluate_on_forged(v5_models, threshold=0.35, min_area=300)
    fp5, _ = evaluate_on_authentic(v5_models, threshold=0.35, min_area=300)
    net5 = tp5 - fp5
    print(f"V5: TP={tp5}/{total_forged} ({tp5/total_forged*100:.1f}%), FP={fp5}/{total_auth} ({fp5/total_auth*100:.1f}%), Net={net5}")
    print()
    
    # Compare
    print("=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"V4: TP={tp4}, FP={fp4}, Net={net4}")
    print(f"V5: TP={tp5}, FP={fp5}, Net={net5}")
    print(f"Delta: TP={tp5-tp4:+d}, FP={fp5-fp4:+d}, Net={net5-net4:+d}")
    
    if net5 > net4:
        print(f"\n✓ V5 is BETTER by {net5-net4} points!")
    elif net5 < net4:
        print(f"\n✗ V5 is WORSE by {net4-net5} points")
    else:
        print("\n= V5 is the same as V4")


if __name__ == '__main__':
    main()
