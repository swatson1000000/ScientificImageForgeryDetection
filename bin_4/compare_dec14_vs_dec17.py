#!/usr/bin/env python3
"""
Head-to-head comparison: Dec 14 single-stage vs Dec 17 two-stage

Dec 14 config (from script_2):
  - 4-model ensemble
  - threshold=0.35, min_area=300
  - TTA 4x, adaptive, min_confidence=0.45

Dec 17 config (from script_3):
  - Classifier (threshold=0.25) + 4-model ensemble
  - seg_threshold=0.35, min_area=300
  - TTA 4x, adaptive
"""

import numpy as np
import torch
import torch.nn as nn
import cv2
import timm
import segmentation_models_pytorch as smp
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "train_images"
MODELS_DIR = PROJECT_ROOT / "models"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CLASSIFIER_SIZE = 384
SEG_SIZE = 512

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
        return max(0.15, base_threshold - 0.08)
    elif brightness < 80:
        return max(0.20, base_threshold - 0.04)
    elif brightness > 200:
        return min(0.50, base_threshold + 0.05)
    return base_threshold


def run_ensemble_dec14(models, img, device, threshold=0.35, min_area=300, min_confidence=0.45):
    """Dec 14 config: ensemble + threshold + min_area + min_confidence"""
    threshold = get_adaptive_threshold(img, threshold)
    img_tensor = preprocess(img, SEG_SIZE, device)
    
    all_preds = []
    for model in models:
        pred = apply_tta(model, img_tensor)
        all_preds.append(pred)
    
    avg_pred = torch.stack(all_preds).mean(dim=0)
    pred_np = avg_pred.cpu().numpy()[0, 0]
    binary = (pred_np > threshold).astype(np.uint8)
    
    # Min area filter
    if min_area > 0 and binary.max() > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                binary[labels == i] = 0
    
    # Min confidence filter (from Dec 14 config)
    if min_confidence > 0 and binary.max() > 0:
        max_conf = pred_np[binary > 0].max()
        if max_conf < min_confidence:
            binary = np.zeros_like(binary)
    
    return binary.max() > 0


def run_two_stage_dec17(classifier, models, img, device, cls_threshold=0.25, seg_threshold=0.35, min_area=300):
    """Dec 17 config: classifier gate + ensemble"""
    # Stage 1: Classifier
    img_cls = preprocess(img, CLASSIFIER_SIZE, device)
    with torch.no_grad():
        cls_prob = torch.sigmoid(classifier(img_cls)).item()
    
    if cls_prob < cls_threshold:
        return False  # Filtered as authentic
    
    # Stage 2: Ensemble segmentation
    seg_threshold = get_adaptive_threshold(img, seg_threshold)
    img_tensor = preprocess(img, SEG_SIZE, device)
    
    all_preds = []
    for model in models:
        pred = apply_tta(model, img_tensor)
        all_preds.append(pred)
    
    avg_pred = torch.stack(all_preds).mean(dim=0)
    pred_np = avg_pred.cpu().numpy()[0, 0]
    binary = (pred_np > seg_threshold).astype(np.uint8)
    
    if min_area > 0 and binary.max() > 0:
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
    
    # Load ensemble
    models = []
    for model_name in V4_MODELS:
        model_path = MODELS_DIR / model_name
        if model_path.exists():
            models.append(load_model(model_path, DEVICE))
            print(f"  ✓ {model_name}")
    print(f"Loaded {len(models)} ensemble models")
    
    # Get images
    forged_dir = DATA_DIR / "forged"
    authentic_dir = DATA_DIR / "authentic"
    forged_images = sorted(forged_dir.glob("*"))
    authentic_images = sorted(authentic_dir.glob("*"))
    
    print(f"\n{'='*70}")
    print("HEAD-TO-HEAD: Dec 14 Single-Stage vs Dec 17 Two-Stage")
    print(f"{'='*70}\n")
    
    # Results
    dec14_tp, dec14_fp = 0, 0
    dec17_tp, dec17_fp = 0, 0
    
    # Process forged
    print("Processing forged images...")
    for img_path in tqdm(forged_images):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        if run_ensemble_dec14(models, img, DEVICE, threshold=0.35, min_area=300, min_confidence=0.45):
            dec14_tp += 1
        
        if run_two_stage_dec17(classifier, models, img, DEVICE, cls_threshold=0.25, seg_threshold=0.35, min_area=300):
            dec17_tp += 1
    
    # Process authentic
    print("Processing authentic images...")
    for img_path in tqdm(authentic_images):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        if run_ensemble_dec14(models, img, DEVICE, threshold=0.35, min_area=300, min_confidence=0.45):
            dec14_fp += 1
        
        if run_two_stage_dec17(classifier, models, img, DEVICE, cls_threshold=0.25, seg_threshold=0.35, min_area=300):
            dec17_fp += 1
    
    # Results
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"{'Config':<30} {'TP':>8} {'FP':>8} {'Net':>8} {'Recall':>8} {'FP Rate':>8}")
    print("-"*70)
    
    dec14_net = dec14_tp - dec14_fp
    dec17_net = dec17_tp - dec17_fp
    dec14_recall = dec14_tp / len(forged_images) * 100
    dec17_recall = dec17_tp / len(forged_images) * 100
    dec14_fpr = dec14_fp / len(authentic_images) * 100
    dec17_fpr = dec17_fp / len(authentic_images) * 100
    
    print(f"{'Dec 14 Single-Stage':<30} {dec14_tp:>8} {dec14_fp:>8} {dec14_net:>8} {dec14_recall:>7.1f}% {dec14_fpr:>7.1f}%")
    print(f"{'Dec 17 Two-Stage':<30} {dec17_tp:>8} {dec17_fp:>8} {dec17_net:>8} {dec17_recall:>7.1f}% {dec17_fpr:>7.1f}%")
    print("-"*70)
    
    diff = dec17_net - dec14_net
    print(f"\nDifference (Dec17 - Dec14): {diff:+d}")
    
    if diff > 0:
        print(f"\n✓ Dec 17 Two-Stage is BETTER by {diff} points")
    elif diff < 0:
        print(f"\n✓ Dec 14 Single-Stage is BETTER by {-diff} points")
    else:
        print(f"\n= Both configs are TIED")


if __name__ == "__main__":
    main()
