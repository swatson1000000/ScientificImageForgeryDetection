#!/usr/bin/env python3
"""
Compare Single V4 vs 4-Model Ensemble with Classifier Gate

Tests whether the ensemble improves net score when combined with binary classifier.
"""

import numpy as np
import torch
import torch.nn as nn
import cv2
import timm
import segmentation_models_pytorch as smp
from pathlib import Path
from tqdm import tqdm

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "train_images"
MODELS_DIR = PROJECT_ROOT / "models"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CLASSIFIER_SIZE = 384
SEG_SIZE = 512

# V4 models for ensemble
V4_MODELS = [
    'highres_no_ela_v4_best.pth',
    'hard_negative_v4_best.pth',
    'high_recall_v4_best.pth',
    'enhanced_aug_v4_best.pth',
]


class ForgeryClassifier(nn.Module):
    def __init__(self, backbone='efficientnet_b2'):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=False, num_classes=0)
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features).squeeze(-1)


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


def load_model(model_path, device):
    """Load a single segmentation model."""
    base = smp.FPN(
        encoder_name='timm-efficientnet-b2',
        encoder_weights=None,
        in_channels=3,
        classes=1
    )
    model = AttentionFPN(base).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model


def preprocess(img, size, device):
    img_resized = cv2.resize(img, (size, size))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_norm = (img_norm - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    return torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).float().to(device)


def apply_tta(model, img_tensor):
    """4x TTA with MAX aggregation."""
    preds = []
    with torch.no_grad():
        preds.append(torch.sigmoid(model(img_tensor)))
        pred = torch.sigmoid(model(torch.flip(img_tensor, dims=[3])))
        preds.append(torch.flip(pred, dims=[3]))
        pred = torch.sigmoid(model(torch.flip(img_tensor, dims=[2])))
        preds.append(torch.flip(pred, dims=[2]))
        pred = torch.sigmoid(model(torch.flip(img_tensor, dims=[2, 3])))
        preds.append(torch.flip(pred, dims=[2, 3]))
    return torch.stack(preds).max(dim=0)[0]


def get_adaptive_threshold(img, base_threshold=0.35):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    if brightness < 50:
        return max(0.20, base_threshold - 0.10)
    elif brightness < 80:
        return max(0.25, base_threshold - 0.05)
    elif brightness > 200:
        return min(0.50, base_threshold + 0.05)
    return base_threshold


def run_single_model(model, img, device, seg_threshold=0.35, min_area=300):
    """Run single model with TTA."""
    threshold = get_adaptive_threshold(img, seg_threshold)
    img_tensor = preprocess(img, SEG_SIZE, device)
    pred = apply_tta(model, img_tensor)
    pred_np = pred.cpu().numpy()[0, 0]
    binary = (pred_np > threshold).astype(np.uint8)
    
    if min_area > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                binary[labels == i] = 0
    
    return binary.max() > 0


def run_ensemble(models, img, device, seg_threshold=0.35, min_area=300):
    """Run ensemble with TTA and MEAN aggregation."""
    threshold = get_adaptive_threshold(img, seg_threshold)
    img_tensor = preprocess(img, SEG_SIZE, device)
    
    all_preds = []
    for model in models:
        pred = apply_tta(model, img_tensor)
        all_preds.append(pred)
    
    # Mean of all models
    avg_pred = torch.stack(all_preds).mean(dim=0)
    pred_np = avg_pred.cpu().numpy()[0, 0]
    binary = (pred_np > threshold).astype(np.uint8)
    
    if min_area > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                binary[labels == i] = 0
    
    return binary.max() > 0


def main():
    print(f"Device: {DEVICE}")
    
    # Load classifier
    classifier = ForgeryClassifier('efficientnet_b2').to(DEVICE)
    classifier_ckpt = torch.load(MODELS_DIR / "binary_classifier_best.pth", map_location=DEVICE)
    classifier.load_state_dict(classifier_ckpt['model_state_dict'])
    classifier.eval()
    print("Loaded classifier")
    
    # Load single V4 model
    single_model = load_model(MODELS_DIR / "highres_no_ela_v4_best.pth", DEVICE)
    print("Loaded single V4 model")
    
    # Load ensemble
    ensemble_models = []
    for model_name in V4_MODELS:
        model_path = MODELS_DIR / model_name
        if model_path.exists():
            ensemble_models.append(load_model(model_path, DEVICE))
            print(f"  ✓ {model_name}")
        else:
            print(f"  ✗ {model_name} not found")
    print(f"Loaded {len(ensemble_models)} ensemble models")
    
    # Get image paths
    forged_dir = DATA_DIR / "forged"
    authentic_dir = DATA_DIR / "authentic"
    forged_images = sorted(forged_dir.glob("*"))
    authentic_images = sorted(authentic_dir.glob("*"))
    
    classifier_threshold = 0.25  # Optimal from previous sweep
    
    print(f"\n{'='*70}")
    print(f"COMPARING SINGLE V4 vs ENSEMBLE (classifier_threshold={classifier_threshold})")
    print(f"{'='*70}\n")
    
    # Results
    single_tp, single_fp = 0, 0
    ensemble_tp, ensemble_fp = 0, 0
    
    # Process forged images
    print("Processing forged images...")
    for img_path in tqdm(forged_images):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        # Classifier
        img_cls = preprocess(img, CLASSIFIER_SIZE, DEVICE)
        with torch.no_grad():
            cls_prob = torch.sigmoid(classifier(img_cls)).item()
        
        if cls_prob < classifier_threshold:
            # Filtered - neither detects
            continue
        
        # Single model
        if run_single_model(single_model, img, DEVICE):
            single_tp += 1
        
        # Ensemble
        if run_ensemble(ensemble_models, img, DEVICE):
            ensemble_tp += 1
    
    # Process authentic images
    print("Processing authentic images...")
    for img_path in tqdm(authentic_images):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        # Classifier
        img_cls = preprocess(img, CLASSIFIER_SIZE, DEVICE)
        with torch.no_grad():
            cls_prob = torch.sigmoid(classifier(img_cls)).item()
        
        if cls_prob < classifier_threshold:
            # Filtered - neither detects (good)
            continue
        
        # Single model
        if run_single_model(single_model, img, DEVICE):
            single_fp += 1
        
        # Ensemble
        if run_ensemble(ensemble_models, img, DEVICE):
            ensemble_fp += 1
    
    # Results
    print(f"\n{'='*70}")
    print("RESULTS (with classifier threshold=0.25)")
    print(f"{'='*70}")
    print(f"{'Config':<20} {'TP':>8} {'FP':>8} {'Net':>8}")
    print("-"*50)
    print(f"{'Single V4':<20} {single_tp:>8} {single_fp:>8} {single_tp - single_fp:>8}")
    print(f"{'4-Model Ensemble':<20} {ensemble_tp:>8} {ensemble_fp:>8} {ensemble_tp - ensemble_fp:>8}")
    print(f"\nDifference: {(ensemble_tp - ensemble_fp) - (single_tp - single_fp):+d}")


if __name__ == "__main__":
    main()
