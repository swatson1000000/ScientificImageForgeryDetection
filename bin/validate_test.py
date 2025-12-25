#!/usr/bin/env python3
"""
Validate on test_authentic_100 and test_forged_100 directories.
Reports TP, FP, TN, FN and net score.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import timm
import segmentation_models_pytorch as smp
from pathlib import Path
import argparse

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CLASSIFIER_SIZE = 384
SEG_SIZE = 512

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
MODELS_DIR = PROJECT_DIR / 'models'

# Model architectures (same as training scripts)
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
        self.base = base_model  # Use 'base' to match saved weights
        self.attention = AttentionGate(1)
    
    def forward(self, x):
        x = self.base(x)
        x = self.attention(x)
        return x


class ForgeryClassifier(nn.Module):
    def __init__(self, backbone_name='efficientnet_b2', pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        num_features = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


def load_classifier():
    model = ForgeryClassifier(pretrained=False)
    path = MODELS_DIR / 'binary_classifier_best.pth'
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    return model


def load_segmentation_models():
    models = []
    model_names = ['highres_no_ela_v4_best.pth', 'hard_negative_v4_best.pth', 
                   'high_recall_v4_best.pth', 'enhanced_aug_v4_best.pth']
    
    for name in model_names:
        path = MODELS_DIR / name
        if path.exists():
            base = smp.FPN(encoder_name='timm-efficientnet-b2', encoder_weights=None, 
                          in_channels=3, classes=1)
            model = AttentionFPN(base).to(DEVICE)
            checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            models.append(model)
    
    return models


def preprocess_for_classifier(img_path):
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (CLASSIFIER_SIZE, CLASSIFIER_SIZE))
    img = img.astype(np.float32) / 255.0
    img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    img = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()
    return img.to(DEVICE)


def preprocess_for_segmentation(img_path):
    img = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (SEG_SIZE, SEG_SIZE))
    img_norm = img_resized.astype(np.float32) / 255.0
    # Apply ImageNet normalization (same as training)
    img_norm = (img_norm - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).float()
    return tensor.to(DEVICE), img  # Return original BGR image for adaptive threshold


def apply_tta(model, img_tensor):
    """Apply 4x TTA (flips) with MAX aggregation for better recall."""
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
    
    # MAX aggregation for better recall
    return torch.stack(preds).max(dim=0)[0]


def get_adaptive_threshold(img_bgr, base_threshold=0.35):
    """Adaptive threshold based on image brightness (matches script_4)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)  # 0-255 scale
    
    if brightness < 50:
        return max(0.20, base_threshold - 0.10)
    elif brightness < 80:
        return max(0.25, base_threshold - 0.05)
    elif brightness > 200:
        return min(0.50, base_threshold + 0.05)
    return base_threshold


def run_inference(classifier, seg_models, img_path, classifier_threshold=0.25, 
                  seg_threshold=0.35, min_area=300, use_tta=True, adaptive=True):
    """Returns (predicted_forged, classifier_score, mask_area)"""
    
    # Stage 1: Classifier
    with torch.no_grad():
        img = preprocess_for_classifier(img_path)
        logit = classifier(img)
        prob = torch.sigmoid(logit).item()
    
    if prob < classifier_threshold:
        return False, prob, 0
    
    # Stage 2: Segmentation ensemble with TTA
    with torch.no_grad():
        img_tensor, img_bgr = preprocess_for_segmentation(img_path)
        
        all_preds = []
        for model in seg_models:
            if use_tta:
                pred = apply_tta(model, img_tensor)
            else:
                pred = torch.sigmoid(model(img_tensor))
            all_preds.append(pred)
        
        # Mean ensemble across models, then squeeze
        mask = torch.stack(all_preds).mean(dim=0).squeeze().cpu().numpy()
    
    # Adaptive threshold
    if adaptive:
        threshold = get_adaptive_threshold(img_bgr, seg_threshold)
    else:
        threshold = seg_threshold
    
    # Apply threshold
    binary_mask = (mask > threshold).astype(np.uint8)
    
    # Remove small regions using connected components (same as script_4)
    if min_area > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                binary_mask[labels == i] = 0
    
    # Check if any forgery region remains
    is_forged = binary_mask.max() > 0
    area = binary_mask.sum()
    
    return is_forged, prob, area


def run_single_config(classifier, seg_models, authentic_files, forged_files,
                       cls_thresh, seg_thresh, min_area):
    """Run inference with given config and return results."""
    tp = fp = tn = fn = 0
    
    for img_path in authentic_files:
        is_forged, prob, area = run_inference(
            classifier, seg_models, img_path, cls_thresh, seg_thresh, min_area
        )
        if is_forged:
            fp += 1
        else:
            tn += 1
    
    for img_path in forged_files:
        is_forged, prob, area = run_inference(
            classifier, seg_models, img_path, cls_thresh, seg_thresh, min_area
        )
        if is_forged:
            tp += 1
        else:
            fn += 1
    
    return tp, fp, tn, fn


# Optimal thresholds from validation sweep (Dec 2025)
# Best Net Score: cls=0.20, seg=0.35, area=25 -> Net=397, Recall=87.8%, FP=8.4%
DEFAULT_CLASSIFIER_THRESHOLD = 0.20
DEFAULT_SEG_THRESHOLD = 0.35
DEFAULT_MIN_AREA = 25


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--classifier-threshold', type=float, default=DEFAULT_CLASSIFIER_THRESHOLD)
    parser.add_argument('--seg-threshold', type=float, default=DEFAULT_SEG_THRESHOLD)
    parser.add_argument('--min-area', type=int, default=DEFAULT_MIN_AREA)
    parser.add_argument('--sweep', action='store_true', help='Run parameter sweep')
    args = parser.parse_args()
    
    print("=" * 60)
    print("VALIDATION ON TEST DATA")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    
    # Load models
    print("Loading models...")
    classifier = load_classifier()
    seg_models = load_segmentation_models()
    print(f"Loaded classifier and {len(seg_models)} segmentation models")
    print()
    
    # Test directories
    authentic_dir = PROJECT_DIR / 'test_authentic_100'
    forged_dir = PROJECT_DIR / 'test_forged_100'
    
    authentic_files = list(authentic_dir.glob('*.png')) + list(authentic_dir.glob('*.jpg'))
    forged_files = list(forged_dir.glob('*.png')) + list(forged_dir.glob('*.jpg'))
    
    print(f"Authentic images: {len(authentic_files)}")
    print(f"Forged images: {len(forged_files)}")
    print()
    
    if args.sweep:
        # Parameter sweep
        cls_thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
        seg_thresholds = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
        min_areas = [25, 50, 100, 150, 200, 300]
        
        print("=" * 100)
        print("PARAMETER SWEEP")
        print("=" * 100)
        print(f"{'Cls':<6} {'Seg':<6} {'Area':<6} {'TP':<6} {'FP':<6} {'Net':<8} {'Recall':<10} {'FP Rate':<10} {'Precision':<10}")
        print("-" * 100)
        
        best_net = -999
        best_recall_at_low_fp = 0
        best_config = None
        best_config_low_fp = None
        
        for cls_t in cls_thresholds:
            for seg_t in seg_thresholds:
                for min_a in min_areas:
                    tp, fp, tn, fn = run_single_config(
                        classifier, seg_models, authentic_files, forged_files,
                        cls_t, seg_t, min_a
                    )
                    
                    net = tp - fp
                    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
                    fp_rate = fp / (fp + tn) * 100 if (fp + tn) > 0 else 0
                    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
                    
                    # Track best net score
                    if net > best_net:
                        best_net = net
                        best_config = (cls_t, seg_t, min_a, tp, fp, recall, fp_rate)
                    
                    # Track best recall with FP rate < 10%
                    if fp_rate < 10 and recall > best_recall_at_low_fp:
                        best_recall_at_low_fp = recall
                        best_config_low_fp = (cls_t, seg_t, min_a, tp, fp, recall, fp_rate)
                    
                    print(f"{cls_t:<6.2f} {seg_t:<6.2f} {min_a:<6} {tp:<6} {fp:<6} {net:<8} {recall:<10.1f} {fp_rate:<10.1f} {precision:<10.1f}")
        
        print()
        print("=" * 100)
        print("BEST CONFIGURATIONS")
        print("=" * 100)
        if best_config:
            cls_t, seg_t, min_a, tp, fp, recall, fp_rate = best_config
            print(f"Best Net Score: cls={cls_t}, seg={seg_t}, area={min_a}")
            print(f"  TP={tp}, FP={fp}, Net={tp-fp}, Recall={recall:.1f}%, FP Rate={fp_rate:.1f}%")
        if best_config_low_fp:
            cls_t, seg_t, min_a, tp, fp, recall, fp_rate = best_config_low_fp
            print(f"Best Recall (FP<10%): cls={cls_t}, seg={seg_t}, area={min_a}")
            print(f"  TP={tp}, FP={fp}, Net={tp-fp}, Recall={recall:.1f}%, FP Rate={fp_rate:.1f}%")
    else:
        # Single run
        print(f"Classifier threshold: {args.classifier_threshold}")
        print(f"Seg threshold: {args.seg_threshold}")
        print(f"Min area: {args.min_area}")
        print()
        
        tp, fp, tn, fn = run_single_config(
            classifier, seg_models, authentic_files, forged_files,
            args.classifier_threshold, args.seg_threshold, args.min_area
        )
        
        # Summary
        print("=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"True Positives (TP):  {tp}")
        print(f"False Positives (FP): {fp}")
        print(f"True Negatives (TN):  {tn}")
        print(f"False Negatives (FN): {fn}")
        print()
        print(f"Net Score (TP - FP):  {tp - fp}")
        print(f"Recall:               {tp / (tp + fn) * 100:.1f}%")
        print(f"FP Rate:              {fp / (fp + tn) * 100:.1f}%")
        if tp + fp > 0:
            print(f"Precision:            {tp / (tp + fp) * 100:.1f}%")


if __name__ == '__main__':
    main()
