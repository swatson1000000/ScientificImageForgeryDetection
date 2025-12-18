#!/usr/bin/env python3
"""
Validate V6 model (UNet++ B4 768px) on training data.
Uses same TTA and postprocessing as script_2 for fair comparison.
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
# CONFIGURATION - Match V6 training
# ============================================================================

IMG_SIZE = 768
ENCODER = 'timm-efficientnet-b4'
THRESHOLD = 0.35
MIN_AREA = 300
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
TRAIN_IMAGES_PATH = PROJECT_DIR / 'train_images'
TRAIN_MASKS_PATH = PROJECT_DIR / 'train_masks'
MODEL_PATH = PROJECT_DIR / 'models' / 'unetpp_b4_768_v6_best.pth'

# ============================================================================
# MODEL ARCHITECTURE - Must match training exactly
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


class AttentionUNetPP(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        self.attention = AttentionGate(1)
    
    def forward(self, x):
        return self.attention(self.base(x))


def create_model_v6(in_channels=3):
    base = smp.UnetPlusPlus(
        encoder_name=ENCODER,
        encoder_weights='imagenet',
        in_channels=in_channels,
        classes=1,
        decoder_attention_type='scse',
    )
    return AttentionUNetPP(base)


# ============================================================================
# TTA - Match script_2 exactly (flips + MAX aggregation)
# ============================================================================

def apply_tta(model, image_tensor, device):
    """Apply 4x TTA with flips and MAX aggregation (matching script_2)."""
    model.eval()
    
    with torch.no_grad():
        # Original
        pred0 = torch.sigmoid(model(image_tensor.to(device)))
        
        # Horizontal flip
        img_hflip = torch.flip(image_tensor, dims=[3])
        pred1 = torch.sigmoid(model(img_hflip.to(device)))
        pred1 = torch.flip(pred1, dims=[3])
        
        # Vertical flip
        img_vflip = torch.flip(image_tensor, dims=[2])
        pred2 = torch.sigmoid(model(img_vflip.to(device)))
        pred2 = torch.flip(pred2, dims=[2])
        
        # Both flips
        img_hvflip = torch.flip(image_tensor, dims=[2, 3])
        pred3 = torch.sigmoid(model(img_hvflip.to(device)))
        pred3 = torch.flip(pred3, dims=[2, 3])
        
        # MAX aggregation (matching script_2)
        pred = torch.max(torch.stack([pred0, pred1, pred2, pred3], dim=0), dim=0)[0]
    
    return pred.cpu().numpy()[0, 0]


# ============================================================================
# POSTPROCESSING - Match script_2 exactly (brightness-based adaptive threshold)
# ============================================================================

def postprocess(mask, image, threshold=THRESHOLD, min_area=MIN_AREA):
    """Postprocess mask with brightness-based adaptive threshold (matching script_2)."""
    # Calculate image brightness for adaptive thresholding
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    brightness = np.mean(gray) / 255.0
    
    # Adaptive threshold based on brightness
    if brightness < 0.3:
        adaptive_thresh = threshold * 0.8
    elif brightness > 0.7:
        adaptive_thresh = threshold * 1.2
    else:
        adaptive_thresh = threshold
    
    # Apply threshold
    binary = (mask > adaptive_thresh).astype(np.uint8)
    
    # Remove small regions
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    clean = np.zeros_like(binary)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 1
    
    return clean


# ============================================================================
# MAIN VALIDATION
# ============================================================================

def main():
    print("=" * 60)
    print("V6 MODEL VALIDATION")
    print("=" * 60)
    print(f"Model: UNet++ with EfficientNet-B4 @ 768px")
    print(f"Model path: {MODEL_PATH}")
    print(f"Threshold: {THRESHOLD}, Min area: {MIN_AREA}")
    print(f"TTA: 4x flips with MAX aggregation")
    print(f"Device: {DEVICE}")
    print()
    
    # Check model exists
    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}")
        return
    
    # Load model
    print("Loading V6 model...")
    model = create_model_v6()
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    # Handle both raw state_dict and checkpoint with model_state_dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print()
    
    # Transform
    transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # Get images from forged and authentic subdirectories
    forged_dir = TRAIN_IMAGES_PATH / 'forged'
    authentic_dir = TRAIN_IMAGES_PATH / 'authentic'
    
    forged = sorted(list(forged_dir.glob('*.png')))
    authentic = sorted(list(authentic_dir.glob('*.png')))
    
    print(f"Found {len(forged)} forged images, {len(authentic)} authentic images")
    print()
    
    # Validate
    true_positives = 0
    false_negatives = 0
    false_positives = 0
    true_negatives = 0
    
    print("Validating forged images...")
    for img_path in tqdm(forged, desc="Forged"):
        # Load image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]
        
        # Transform
        transformed = transform(image=image)
        img_tensor = transformed['image'].unsqueeze(0)
        
        # Predict with TTA
        pred = apply_tta(model, img_tensor, DEVICE)
        
        # Resize to original size
        pred = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        
        # Postprocess
        binary = postprocess(pred, image)
        
        # Check if any forgery detected
        if binary.sum() > 0:
            true_positives += 1
        else:
            false_negatives += 1
    
    print(f"\nValidating authentic images...")
    for img_path in tqdm(authentic, desc="Authentic"):
        # Load image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]
        
        # Transform
        transformed = transform(image=image)
        img_tensor = transformed['image'].unsqueeze(0)
        
        # Predict with TTA
        pred = apply_tta(model, img_tensor, DEVICE)
        
        # Resize to original size
        pred = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        
        # Postprocess
        binary = postprocess(pred, image)
        
        # Check if any forgery detected (should be none for authentic)
        if binary.sum() > 0:
            false_positives += 1
        else:
            true_negatives += 1
    
    # Results
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    tp_rate = true_positives / len(forged) * 100 if forged else 0
    fp_rate = false_positives / len(authentic) * 100 if authentic else 0
    net_score = true_positives - false_positives
    
    print(f"True Positives:  {true_positives}/{len(forged)} ({tp_rate:.1f}%)")
    print(f"False Positives: {false_positives}/{len(authentic)} ({fp_rate:.1f}%)")
    print(f"Net Score:       {net_score}")
    print("=" * 60)
    
    # Compare to V4 baseline
    print()
    print("COMPARISON TO V4 BASELINE:")
    print(f"  V4: 2235 TP, 207 FP, Net 2028")
    print(f"  V6: {true_positives} TP, {false_positives} FP, Net {net_score}")
    diff = net_score - 2028
    print(f"  Difference: {diff:+d}")


if __name__ == '__main__':
    main()
