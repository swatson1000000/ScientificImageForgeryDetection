#!/usr/bin/env python3
"""
Two-Stage Validation Pipeline:
1. Binary classifier filters images (forged vs authentic)
2. Segmentation model only runs on images classified as potentially forged

Sweeps classifier threshold to find optimal balance.
"""

import os
import sys
import cv2
import torch
import numpy as np
import timm
import segmentation_models_pytorch as smp
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "train_images"
MODELS_DIR = PROJECT_ROOT / "models"

# Image size
CLASSIFIER_SIZE = 384
SEG_SIZE = 512

class ForgeryClassifier(torch.nn.Module):
    """Binary classifier: forged (1) vs authentic (0)."""
    
    def __init__(self, backbone='efficientnet_b2'):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=False, num_classes=0)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.backbone.num_features, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 1)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features).squeeze(-1)

class AttentionGate(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        hidden1 = max(8, in_channels // 2)
        hidden2 = max(4, in_channels // 4)
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, hidden1, kernel_size=1),
            torch.nn.BatchNorm2d(hidden1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(hidden1, hidden2, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(hidden2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(hidden2, 1, kernel_size=1),
            torch.nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.conv(x)


class AttentionFPN(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        self.attention = AttentionGate(1)
    
    def forward(self, x):
        return self.attention(self.base(x))


def load_models(device):
    """Load both classifier and segmentation models."""
    # Load binary classifier
    classifier = ForgeryClassifier('efficientnet_b2').to(device)
    classifier_path = MODELS_DIR / "binary_classifier_best.pth"
    classifier_ckpt = torch.load(classifier_path, map_location=device)
    classifier.load_state_dict(classifier_ckpt['model_state_dict'])
    classifier.eval()
    print(f"Loaded classifier: {classifier_path}")
    
    # Load segmentation model (V4)
    base = smp.FPN(
        encoder_name="timm-efficientnet-b2",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    seg_model = AttentionFPN(base).to(device)
    seg_path = MODELS_DIR / "highres_no_ela_v4_best.pth"
    seg_ckpt = torch.load(seg_path, map_location=device)
    if 'model_state_dict' in seg_ckpt:
        seg_model.load_state_dict(seg_ckpt['model_state_dict'])
    else:
        seg_model.load_state_dict(seg_ckpt)
    seg_model.eval()
    print(f"Loaded segmentation model: {seg_path}")
    
    return classifier, seg_model


def preprocess_for_classifier(img, device):
    """Preprocess image for classifier."""
    img_resized = cv2.resize(img, (CLASSIFIER_SIZE, CLASSIFIER_SIZE))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_norm = (img_norm - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    return tensor


def preprocess_for_segmentation(img, device):
    """Preprocess image for segmentation."""
    img_resized = cv2.resize(img, (SEG_SIZE, SEG_SIZE))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_norm = (img_norm - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    return tensor


def apply_tta(model, img_tensor):
    """Apply 4x TTA (flips) with MAX aggregation."""
    preds = []
    
    # Original
    with torch.no_grad():
        pred = torch.sigmoid(model(img_tensor))
    preds.append(pred)
    
    # Horizontal flip
    with torch.no_grad():
        pred = torch.sigmoid(model(torch.flip(img_tensor, dims=[3])))
        pred = torch.flip(pred, dims=[3])
    preds.append(pred)
    
    # Vertical flip
    with torch.no_grad():
        pred = torch.sigmoid(model(torch.flip(img_tensor, dims=[2])))
        pred = torch.flip(pred, dims=[2])
    preds.append(pred)
    
    # Both flips
    with torch.no_grad():
        pred = torch.sigmoid(model(torch.flip(img_tensor, dims=[2, 3])))
        pred = torch.flip(pred, dims=[2, 3])
    preds.append(pred)
    
    # MAX aggregation
    return torch.stack(preds).max(dim=0)[0]


def get_adaptive_threshold(img, base_threshold=0.35):
    """Adaptive threshold based on image brightness."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = gray.mean() / 255.0
    
    if brightness < 0.3:
        return base_threshold * 0.8
    elif brightness > 0.7:
        return base_threshold * 1.2
    return base_threshold


def run_segmentation(seg_model, img, device, seg_threshold=0.35, min_area=300):
    """Run segmentation and return whether forgery was detected."""
    img_tensor = preprocess_for_segmentation(img, device)
    
    # TTA prediction
    pred = apply_tta(seg_model, img_tensor)
    pred = pred.squeeze().cpu().numpy()
    
    # Adaptive threshold
    threshold = get_adaptive_threshold(img, seg_threshold)
    binary = (pred > threshold).astype(np.uint8)
    
    # Check for valid regions
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            return True  # Forgery detected
    
    return False  # No forgery


def evaluate_two_stage(classifier, seg_model, device, classifier_threshold=0.5, 
                       seg_threshold=0.35, min_area=300):
    """
    Two-stage evaluation:
    1. Classifier predicts forged probability
    2. If prob >= classifier_threshold, run segmentation
    3. If segmentation finds regions, count as detection
    """
    forged_dir = DATA_DIR / "forged"
    authentic_dir = DATA_DIR / "authentic"
    
    results = {
        'tp': 0,  # Forged image correctly detected
        'fp': 0,  # Authentic image incorrectly detected
        'fn': 0,  # Forged image missed
        'tn': 0,  # Authentic image correctly rejected
        'filtered_by_classifier': 0,  # Images skipped by classifier
        'passed_to_segmentation': 0,  # Images sent to segmentation
    }
    
    # Process forged images
    forged_images = list(forged_dir.glob("*.jpg")) + list(forged_dir.glob("*.png"))
    for img_path in tqdm(forged_images, desc="Forged"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        # Stage 1: Classifier
        cls_tensor = preprocess_for_classifier(img, device)
        with torch.no_grad():
            cls_prob = torch.sigmoid(classifier(cls_tensor)).item()
        
        if cls_prob < classifier_threshold:
            # Classifier says authentic - this is FN
            results['fn'] += 1
            results['filtered_by_classifier'] += 1
        else:
            # Classifier says possibly forged - run segmentation
            results['passed_to_segmentation'] += 1
            if run_segmentation(seg_model, img, device, seg_threshold, min_area):
                results['tp'] += 1
            else:
                results['fn'] += 1
    
    # Process authentic images
    authentic_images = list(authentic_dir.glob("*.jpg")) + list(authentic_dir.glob("*.png"))
    for img_path in tqdm(authentic_images, desc="Authentic"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        # Stage 1: Classifier
        cls_tensor = preprocess_for_classifier(img, device)
        with torch.no_grad():
            cls_prob = torch.sigmoid(classifier(cls_tensor)).item()
        
        if cls_prob < classifier_threshold:
            # Classifier says authentic - correct TN
            results['tn'] += 1
            results['filtered_by_classifier'] += 1
        else:
            # Classifier says possibly forged - run segmentation
            results['passed_to_segmentation'] += 1
            if run_segmentation(seg_model, img, device, seg_threshold, min_area):
                results['fp'] += 1
            else:
                results['tn'] += 1
    
    return results


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load models
    classifier, seg_model = load_models(device)
    
    print("\n" + "="*70)
    print("TWO-STAGE PIPELINE VALIDATION")
    print("="*70)
    
    # Baseline: segmentation only (no classifier filter)
    print("\n--- BASELINE: Segmentation Only ---")
    baseline = evaluate_two_stage(classifier, seg_model, device, 
                                   classifier_threshold=0.0,  # Pass everything
                                   seg_threshold=0.35, min_area=300)
    print(f"TP: {baseline['tp']}, FP: {baseline['fp']}, Net: {baseline['tp'] - baseline['fp']}")
    
    # Sweep classifier thresholds
    print("\n--- CLASSIFIER THRESHOLD SWEEP ---")
    print(f"{'Cls_Thresh':<12} {'TP':<8} {'FP':<8} {'Net':<8} {'Filtered':<10} {'Passed':<10} {'vs Baseline'}")
    print("-"*80)
    
    best_net = baseline['tp'] - baseline['fp']
    best_threshold = 0.0
    
    for cls_thresh in [0.20, 0.25, 0.28, 0.30, 0.32, 0.35, 0.38, 0.40]:
        results = evaluate_two_stage(classifier, seg_model, device,
                                      classifier_threshold=cls_thresh,
                                      seg_threshold=0.35, min_area=300)
        
        net = results['tp'] - results['fp']
        diff = net - (baseline['tp'] - baseline['fp'])
        diff_str = f"+{diff}" if diff > 0 else str(diff)
        
        print(f"{cls_thresh:<12.1f} {results['tp']:<8} {results['fp']:<8} {net:<8} "
              f"{results['filtered_by_classifier']:<10} {results['passed_to_segmentation']:<10} {diff_str}")
        
        if net > best_net:
            best_net = net
            best_threshold = cls_thresh
    
    print("\n" + "="*70)
    print(f"BEST CONFIG: classifier_threshold={best_threshold}, Net={best_net}")
    print(f"Baseline Net: {baseline['tp'] - baseline['fp']}")
    print(f"Improvement: {best_net - (baseline['tp'] - baseline['fp'])}")
    print("="*70)


if __name__ == "__main__":
    main()
