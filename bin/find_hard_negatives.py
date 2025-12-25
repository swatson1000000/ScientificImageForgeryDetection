#!/usr/bin/env python3
"""
Find Hard Negatives: Identify authentic images that produce false positives.

This script runs inference on authentic images and identifies which ones
are incorrectly classified as forged. These "hard negatives" can then be
used for retraining to reduce false positive rate.

Usage:
    python bin/find_hard_negatives.py --output bin_4/fp_images_new.txt
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
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


# ============================================================================
# MODEL ARCHITECTURES (must match training)
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


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_classifier():
    model = ForgeryClassifier(pretrained=False)
    path = MODELS_DIR / 'binary_classifier_best.pth'
    if not path.exists():
        print(f"Warning: Classifier not found at {path}")
        return None
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    return model


def load_segmentation_models():
    models = []
    model_names = ['highres_no_ela_v4_best.pth', 'hard_negative_v4_best.pth', 
                   'high_recall_v4_best.pth', 'enhanced_aug_v4_best.pth']
    
    # Try v4 first, then fall back to v3
    for i, name in enumerate(model_names):
        path = MODELS_DIR / name
        if not path.exists():
            # Try v3 variant
            v3_name = name.replace('_v4_', '_v3_')
            path = MODELS_DIR / v3_name
            if path.exists():
                name = v3_name
        
        if path.exists():
            base = smp.FPN(encoder_name='timm-efficientnet-b2', encoder_weights=None, 
                          in_channels=3, classes=1)
            model = AttentionFPN(base).to(DEVICE)
            checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            models.append(model)
            print(f"  Loaded: {name}")
    
    return models


# ============================================================================
# PREPROCESSING (must match inference pipeline)
# ============================================================================

def preprocess_for_classifier(img_path):
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (CLASSIFIER_SIZE, CLASSIFIER_SIZE))
    img = img.astype(np.float32) / 255.0
    # ImageNet normalization
    img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    img = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()
    return img.to(DEVICE)


def preprocess_for_segmentation(img_path):
    img = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (SEG_SIZE, SEG_SIZE))
    img_norm = img_resized.astype(np.float32) / 255.0
    # ImageNet normalization (CRITICAL!)
    img_norm = (img_norm - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).float()
    return tensor.to(DEVICE), img  # Return original BGR for adaptive threshold


def apply_tta(model, img_tensor):
    """Apply 4x TTA (flips) with MAX aggregation."""
    preds = []
    
    with torch.no_grad():
        pred = torch.sigmoid(model(img_tensor))
    preds.append(pred)
    
    with torch.no_grad():
        pred = torch.sigmoid(model(torch.flip(img_tensor, dims=[3])))
        pred = torch.flip(pred, dims=[3])
    preds.append(pred)
    
    with torch.no_grad():
        pred = torch.sigmoid(model(torch.flip(img_tensor, dims=[2])))
        pred = torch.flip(pred, dims=[2])
    preds.append(pred)
    
    with torch.no_grad():
        pred = torch.sigmoid(model(torch.flip(img_tensor, dims=[2, 3])))
        pred = torch.flip(pred, dims=[2, 3])
    preds.append(pred)
    
    return torch.stack(preds).max(dim=0)[0]


def get_adaptive_threshold(img_bgr, base_threshold=0.35):
    """Adaptive threshold based on image brightness."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    
    if brightness < 50:
        return max(0.20, base_threshold - 0.10)
    elif brightness < 80:
        return max(0.25, base_threshold - 0.05)
    elif brightness > 200:
        return min(0.50, base_threshold + 0.05)
    return base_threshold


# ============================================================================
# INFERENCE
# ============================================================================

def run_inference(classifier, seg_models, img_path, classifier_threshold=0.25, 
                  seg_threshold=0.35, min_area=300):
    """Returns (is_forged, classifier_prob, mask_area)"""
    
    # Stage 1: Classifier
    if classifier is not None:
        with torch.no_grad():
            img = preprocess_for_classifier(img_path)
            logit = classifier(img)
            prob = torch.sigmoid(logit).item()
        
        if prob < classifier_threshold:
            return False, prob, 0
    else:
        prob = 1.0  # No classifier, proceed to segmentation
    
    # Stage 2: Segmentation ensemble with TTA
    with torch.no_grad():
        img_tensor, img_bgr = preprocess_for_segmentation(img_path)
        
        all_preds = []
        for model in seg_models:
            pred = apply_tta(model, img_tensor)
            all_preds.append(pred)
        
        mask = torch.stack(all_preds).mean(dim=0).squeeze().cpu().numpy()
    
    # Adaptive threshold
    threshold = get_adaptive_threshold(img_bgr, seg_threshold)
    
    # Apply threshold
    binary_mask = (mask > threshold).astype(np.uint8)
    
    # Remove small regions using connected components
    if min_area > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                binary_mask[labels == i] = 0
    
    is_forged = binary_mask.max() > 0
    area = binary_mask.sum()
    
    return is_forged, prob, area


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Find hard negatives (false positives on authentic images)')
    parser.add_argument('--authentic-dir', type=str, default=None,
                        help='Directory containing authentic images (default: test_authentic_100)')
    parser.add_argument('--output', type=str, default='hard_negative_ids.txt',
                        help='Output file for hard negative IDs (relative to project root)')
    parser.add_argument('--classifier-threshold', type=float, default=0.25)
    parser.add_argument('--seg-threshold', type=float, default=0.35)
    parser.add_argument('--min-area', type=int, default=300)
    args = parser.parse_args()
    
    # Default authentic directory
    if args.authentic_dir is None:
        authentic_dir = PROJECT_DIR / 'test_authentic_100'
    else:
        authentic_dir = Path(args.authentic_dir)
    
    print("=" * 70)
    print("HARD NEGATIVE MINING")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Authentic images: {authentic_dir}")
    print(f"Classifier threshold: {args.classifier_threshold}")
    print(f"Seg threshold: {args.seg_threshold}")
    print(f"Min area: {args.min_area}")
    print()
    
    # Load models
    print("Loading models...")
    classifier = load_classifier()
    seg_models = load_segmentation_models()
    print(f"Loaded {len(seg_models)} segmentation models")
    print()
    
    if len(seg_models) == 0:
        print("ERROR: No segmentation models found. Train models first.")
        sys.exit(1)
    
    # Get all authentic images
    image_files = list(authentic_dir.glob('*.png')) + list(authentic_dir.glob('*.jpg'))
    print(f"Found {len(image_files)} authentic images")
    print()
    
    # Find false positives
    false_positives = []
    
    print("Running inference on authentic images...")
    for i, img_path in enumerate(image_files):
        is_forged, prob, area = run_inference(
            classifier, seg_models, img_path,
            args.classifier_threshold, args.seg_threshold, args.min_area
        )
        
        if is_forged:
            # This authentic image was incorrectly classified as forged
            img_id = img_path.stem  # Get filename without extension
            false_positives.append(img_id)
            print(f"  FP: {img_path.name} (prob={prob:.3f}, area={area})")
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(image_files)} images, FPs so far: {len(false_positives)}")
    
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Total authentic images: {len(image_files)}")
    print(f"False positives found: {len(false_positives)}")
    print(f"FP rate: {len(false_positives) / len(image_files) * 100:.1f}%")
    print()
    
    # Save to file
    output_path = PROJECT_DIR / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for img_id in sorted(false_positives, key=lambda x: int(x) if x.isdigit() else x):
            f.write(f"{img_id}\n")
    
    print(f"Saved {len(false_positives)} hard negative IDs to: {output_path}")
    print()
    
    # Also print as Python list for easy copy-paste
    print("Python list format (for script_1_train_v3.py):")
    print("HARD_NEGATIVE_IDS = [")
    ids_sorted = sorted([int(x) if x.isdigit() else x for x in false_positives])
    for i in range(0, len(ids_sorted), 10):
        chunk = ids_sorted[i:i+10]
        print(f"    {', '.join(str(x) for x in chunk)},")
    print("]")


if __name__ == '__main__':
    main()
