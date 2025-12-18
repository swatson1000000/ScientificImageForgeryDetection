#!/usr/bin/env python3
"""
Validate V4 ensemble on training data with full pipeline (TTA + adaptive)
"""

import sys
sys.stdout.reconfigure(line_buffering=True)  # Force unbuffered output
import numpy as np
import torch
import torch.nn as nn
import cv2
import segmentation_models_pytorch as smp
from pathlib import Path

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 512
THRESHOLD = 0.35
MIN_AREA = 300

V4_MODELS = [
    'highres_no_ela_v4_best.pth',
    'hard_negative_v4_best.pth',
    'high_recall_v4_best.pth',
    'enhanced_aug_v4_best.pth',
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
    def __init__(self, in_channels=3):
        super().__init__()
        self.base = smp.FPN(
            encoder_name='timm-efficientnet-b2',
            encoder_weights=None,
            in_channels=in_channels,
            classes=1
        )
        self.attention = AttentionGate(1)
    
    def forward(self, x):
        x = self.base(x)
        x = self.attention(x)
        return x


def load_model(path):
    model = AttentionFPN(in_channels=3)
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()
    return model


def predict_with_tta(models, img, img_size=512):
    """4x TTA: original + 3 flips (matches script_2)"""
    h, w = img.shape[:2]
    
    # Preprocess
    img_resized = cv2.resize(img, (img_size, img_size))
    img_norm = img_resized.astype(np.float32) / 255.0
    img_norm = (img_norm - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    
    # TTA augmentations: original + 3 flips
    augmented = [
        (img_norm, 'orig'),
        (np.flip(img_norm, axis=1).copy(), 'hflip'),
        (np.flip(img_norm, axis=0).copy(), 'vflip'),
        (np.flip(np.flip(img_norm, axis=0), axis=1).copy(), 'hvflip'),
    ]
    
    predictions = []
    
    for aug_img, aug_type in augmented:
        tensor = torch.from_numpy(aug_img.transpose(2, 0, 1)).float().unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            # Average across all models
            model_preds = [torch.sigmoid(model(tensor)).cpu().numpy()[0, 0] for model in models]
            avg_pred = np.mean(model_preds, axis=0)
        
        # Reverse the augmentation
        if aug_type == 'hflip':
            avg_pred = np.flip(avg_pred, axis=1).copy()
        elif aug_type == 'vflip':
            avg_pred = np.flip(avg_pred, axis=0).copy()
        elif aug_type == 'hvflip':
            avg_pred = np.flip(np.flip(avg_pred, axis=0), axis=1).copy()
        
        predictions.append(avg_pred)
    
    # MAX of all TTA predictions (matches script_2 - boosts recall)
    max_pred = np.max(predictions, axis=0)
    
    # Resize to original
    max_pred = cv2.resize(max_pred, (w, h))
    
    return max_pred


def get_image_brightness(img):
    """Get average brightness of image (0-255)."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return gray.mean()


def postprocess(pred, img, threshold, min_area):
    """Post-processing with adaptive thresholding for dark images (matches script_2)."""
    # Adaptive thresholding based on image brightness (matches script_2)
    brightness = get_image_brightness(img)
    if brightness < 50:  # Very dark image
        threshold = max(0.15, threshold - 0.08)   # e.g., 0.35 -> 0.27
        min_area = max(100, int(min_area / 2.5))  # e.g., 300 -> 120
    elif brightness < 80:  # Moderately dark
        threshold = max(0.20, threshold - 0.04)   # e.g., 0.35 -> 0.31
        min_area = max(150, int(min_area / 1.5))  # e.g., 300 -> 200
    
    # Threshold
    mask = (pred > threshold).astype(np.uint8)
    
    # Remove small components (NO morphological cleanup - matches script_2 default)
    if min_area > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                mask[labels == i] = 0
    
    return mask


def main():
    print("=" * 60)
    print("V4 ENSEMBLE VALIDATION (Full Pipeline)")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Threshold: {THRESHOLD}, Min area: {MIN_AREA}")
    print(f"TTA: 4x, Adaptive threshold: Yes")
    
    # Paths
    project_dir = Path(__file__).parent.parent
    model_dir = project_dir / 'models'
    forged_dir = project_dir / 'train_images' / 'forged'
    authentic_dir = project_dir / 'train_images' / 'authentic'
    
    # Load models
    print(f"\nLoading V4 ensemble...")
    models = []
    for name in V4_MODELS:
        path = model_dir / name
        if path.exists():
            models.append(load_model(str(path)))
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name} not found")
    
    if not models:
        print("No models found!")
        return
    
    print(f"Loaded {len(models)} models")
    
    # Evaluate forged images
    print(f"\n" + "-" * 60)
    print("Evaluating forged images...")
    forged_images = list(forged_dir.glob('*.png'))
    tp = 0
    
    for i, img_path in enumerate(forged_images):
        if i % 500 == 0:
            print(f"  Forged: {i}/{len(forged_images)}")
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        pred = predict_with_tta(models, img)
        mask = postprocess(pred, img, THRESHOLD, MIN_AREA)
        
        if mask.sum() > 0:
            tp += 1
    
    # Evaluate authentic images
    print("Evaluating authentic images...")
    authentic_images = list(authentic_dir.glob('*.png'))
    fp = 0
    
    for i, img_path in enumerate(authentic_images):
        if i % 500 == 0:
            print(f"  Authentic: {i}/{len(authentic_images)}")
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        pred = predict_with_tta(models, img)
        mask = postprocess(pred, img, THRESHOLD, MIN_AREA)
        
        if mask.sum() > 0:
            fp += 1
    
    # Results
    total_forged = len(forged_images)
    total_authentic = len(authentic_images)
    
    print(f"\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"True Positives:  {tp}/{total_forged} ({100*tp/total_forged:.1f}%)")
    print(f"False Positives: {fp}/{total_authentic} ({100*fp/total_authentic:.1f}%)")
    print(f"Net Score:       {tp - fp}")
    print("=" * 60)


if __name__ == '__main__':
    main()
