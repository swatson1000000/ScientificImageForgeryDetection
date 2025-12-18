#!/usr/bin/env python3
"""
Error Analysis - Analyze FP and FN to understand patterns
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
from collections import defaultdict

# ============================================================================
# CONFIGURATION
# ============================================================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 512
ENCODER = 'timm-efficientnet-b2'
THRESHOLD = 0.35
MIN_AREA = 300

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
# PREDICTION (matching script_2)
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
        
        # MAX within model
        model_pred = np.max(np.stack(preds, axis=0), axis=0)
        all_preds.append(model_pred)
    
    # MAX across models
    final = np.max(np.stack(all_preds, axis=0), axis=0)
    return cv2.resize(final, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)


def postprocess(mask, image, threshold=THRESHOLD, min_area=MIN_AREA):
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
    
    return clean, adaptive_thresh


# ============================================================================
# IMAGE ANALYSIS FEATURES
# ============================================================================

def analyze_image(image):
    """Extract features from an image for error analysis."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    h, w = image.shape[:2]
    
    features = {}
    
    # Size
    features['width'] = w
    features['height'] = h
    features['area'] = w * h
    features['aspect_ratio'] = w / h
    
    # Brightness/contrast
    features['brightness'] = np.mean(gray) / 255.0
    features['contrast'] = np.std(gray) / 255.0
    
    # Color statistics
    features['mean_r'] = np.mean(image[:,:,0]) / 255.0
    features['mean_g'] = np.mean(image[:,:,1]) / 255.0
    features['mean_b'] = np.mean(image[:,:,2]) / 255.0
    features['color_variance'] = np.var(image) / (255.0**2)
    
    # Edge density (texture complexity)
    edges = cv2.Canny(gray, 50, 150)
    features['edge_density'] = np.sum(edges > 0) / (h * w)
    
    # Histogram entropy (image complexity)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    features['entropy'] = -np.sum(hist * np.log2(hist))
    
    # JPEG artifacts (block-based variance)
    if h >= 16 and w >= 16:
        block_vars = []
        for i in range(0, h-8, 8):
            for j in range(0, w-8, 8):
                block = gray[i:i+8, j:j+8]
                block_vars.append(np.var(block))
        features['block_variance'] = np.mean(block_vars) if block_vars else 0
    else:
        features['block_variance'] = 0
    
    return features


def analyze_mask(mask_path):
    """Analyze ground truth mask properties."""
    if not mask_path.exists():
        return None
    
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    
    binary = (mask > 127).astype(np.uint8)
    
    features = {}
    
    # Forgery size
    forgery_pixels = np.sum(binary)
    total_pixels = binary.shape[0] * binary.shape[1]
    features['forgery_ratio'] = forgery_pixels / total_pixels
    features['forgery_pixels'] = forgery_pixels
    
    # Number of forgery regions
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
    features['num_regions'] = num_labels - 1  # Exclude background
    
    if num_labels > 1:
        # Region sizes
        region_sizes = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)]
        features['max_region_size'] = max(region_sizes)
        features['min_region_size'] = min(region_sizes)
        features['mean_region_size'] = np.mean(region_sizes)
    else:
        features['max_region_size'] = 0
        features['min_region_size'] = 0
        features['mean_region_size'] = 0
    
    return features


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("ERROR ANALYSIS")
    print("=" * 60)
    print(f"Threshold: {THRESHOLD}, Min area: {MIN_AREA}")
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
    mask_dir = PROJECT_DIR / 'train_masks' / 'forged'
    
    forged_images = sorted(list(forged_dir.glob('*.png')))
    authentic_images = sorted(list(authentic_dir.glob('*.png')))
    
    print(f"Forged: {len(forged_images)}, Authentic: {len(authentic_images)}\n")
    
    # Collect errors
    true_positives = []
    false_negatives = []
    true_negatives = []
    false_positives = []
    
    # Process forged images
    print("Processing forged images...")
    for img_path in tqdm(forged_images, desc="Forged"):
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        pred = predict_with_tta(models, img)
        binary, thresh_used = postprocess(pred, img)
        
        img_features = analyze_image(img)
        mask_path = mask_dir / img_path.name
        mask_features = analyze_mask(mask_path)
        
        info = {
            'path': img_path,
            'img_features': img_features,
            'mask_features': mask_features,
            'max_pred': float(pred.max()),
            'mean_pred': float(pred.mean()),
            'thresh_used': thresh_used,
            'detected_pixels': int(binary.sum()),
        }
        
        if binary.sum() > 0:
            true_positives.append(info)
        else:
            false_negatives.append(info)
    
    # Process authentic images
    print("\nProcessing authentic images...")
    for img_path in tqdm(authentic_images, desc="Authentic"):
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        pred = predict_with_tta(models, img)
        binary, thresh_used = postprocess(pred, img)
        
        img_features = analyze_image(img)
        
        info = {
            'path': img_path,
            'img_features': img_features,
            'mask_features': None,
            'max_pred': float(pred.max()),
            'mean_pred': float(pred.mean()),
            'thresh_used': thresh_used,
            'detected_pixels': int(binary.sum()),
        }
        
        if binary.sum() > 0:
            false_positives.append(info)
        else:
            true_negatives.append(info)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"True Positives:  {len(true_positives)}")
    print(f"False Negatives: {len(false_negatives)}")
    print(f"True Negatives:  {len(true_negatives)}")
    print(f"False Positives: {len(false_positives)}")
    print(f"Net Score: {len(true_positives) - len(false_positives)}")
    
    # Analyze False Negatives (forged images not detected)
    print("\n" + "=" * 60)
    print("FALSE NEGATIVES ANALYSIS (Forged but not detected)")
    print("=" * 60)
    
    if false_negatives:
        fn_features = defaultdict(list)
        for item in false_negatives:
            for k, v in item['img_features'].items():
                fn_features[f'img_{k}'].append(v)
            if item['mask_features']:
                for k, v in item['mask_features'].items():
                    fn_features[f'mask_{k}'].append(v)
            fn_features['max_pred'].append(item['max_pred'])
            fn_features['mean_pred'].append(item['mean_pred'])
        
        print(f"\nImage features (mean ± std):")
        for k in ['img_brightness', 'img_contrast', 'img_edge_density', 'img_entropy']:
            if k in fn_features:
                vals = fn_features[k]
                print(f"  {k}: {np.mean(vals):.3f} ± {np.std(vals):.3f}")
        
        print(f"\nMask features (mean ± std):")
        for k in ['mask_forgery_ratio', 'mask_num_regions', 'mask_mean_region_size']:
            if k in fn_features:
                vals = fn_features[k]
                print(f"  {k}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
        
        print(f"\nPrediction stats:")
        print(f"  max_pred: {np.mean(fn_features['max_pred']):.3f} ± {np.std(fn_features['max_pred']):.3f}")
        print(f"  mean_pred: {np.mean(fn_features['mean_pred']):.4f} ± {np.std(fn_features['mean_pred']):.4f}")
        
        # Small forgeries
        small_forgeries = [x for x in false_negatives if x['mask_features'] and x['mask_features']['forgery_ratio'] < 0.01]
        print(f"\n  FN with tiny forgeries (<1% of image): {len(small_forgeries)}")
        
        # Low max pred
        low_pred = [x for x in false_negatives if x['max_pred'] < 0.3]
        print(f"  FN with low confidence (max_pred < 0.3): {len(low_pred)}")
    
    # Analyze False Positives (authentic images wrongly detected)
    print("\n" + "=" * 60)
    print("FALSE POSITIVES ANALYSIS (Authentic but detected as forged)")
    print("=" * 60)
    
    if false_positives:
        fp_features = defaultdict(list)
        for item in false_positives:
            for k, v in item['img_features'].items():
                fp_features[f'img_{k}'].append(v)
            fp_features['max_pred'].append(item['max_pred'])
            fp_features['mean_pred'].append(item['mean_pred'])
            fp_features['detected_pixels'].append(item['detected_pixels'])
        
        print(f"\nImage features (mean ± std):")
        for k in ['img_brightness', 'img_contrast', 'img_edge_density', 'img_entropy']:
            if k in fp_features:
                vals = fp_features[k]
                print(f"  {k}: {np.mean(vals):.3f} ± {np.std(vals):.3f}")
        
        print(f"\nPrediction stats:")
        print(f"  max_pred: {np.mean(fp_features['max_pred']):.3f} ± {np.std(fp_features['max_pred']):.3f}")
        print(f"  mean_pred: {np.mean(fp_features['mean_pred']):.4f} ± {np.std(fp_features['mean_pred']):.4f}")
        print(f"  detected_pixels: {np.mean(fp_features['detected_pixels']):.0f} ± {np.std(fp_features['detected_pixels']):.0f}")
        
        # Small detections
        small_detections = [x for x in false_positives if x['detected_pixels'] < 1000]
        print(f"\n  FP with small detections (<1000 px): {len(small_detections)}")
        
        # High confidence FPs (hardest to fix)
        high_conf_fp = [x for x in false_positives if x['max_pred'] > 0.7]
        print(f"  FP with high confidence (max_pred > 0.7): {len(high_conf_fp)}")
    
    # Compare TP vs FN
    print("\n" + "=" * 60)
    print("COMPARISON: TP vs FN")
    print("=" * 60)
    
    if true_positives and false_negatives:
        tp_forgery_ratios = [x['mask_features']['forgery_ratio'] for x in true_positives if x['mask_features']]
        fn_forgery_ratios = [x['mask_features']['forgery_ratio'] for x in false_negatives if x['mask_features']]
        
        print(f"\nForgery size (ratio of image):")
        print(f"  TP: {np.mean(tp_forgery_ratios):.4f} ± {np.std(tp_forgery_ratios):.4f}")
        print(f"  FN: {np.mean(fn_forgery_ratios):.4f} ± {np.std(fn_forgery_ratios):.4f}")
        
        tp_max_pred = [x['max_pred'] for x in true_positives]
        fn_max_pred = [x['max_pred'] for x in false_negatives]
        print(f"\nMax prediction confidence:")
        print(f"  TP: {np.mean(tp_max_pred):.3f} ± {np.std(tp_max_pred):.3f}")
        print(f"  FN: {np.mean(fn_max_pred):.3f} ± {np.std(fn_max_pred):.3f}")
    
    # Compare TN vs FP
    print("\n" + "=" * 60)
    print("COMPARISON: TN vs FP")
    print("=" * 60)
    
    if true_negatives and false_positives:
        tn_brightness = [x['img_features']['brightness'] for x in true_negatives]
        fp_brightness = [x['img_features']['brightness'] for x in false_positives]
        print(f"\nBrightness:")
        print(f"  TN: {np.mean(tn_brightness):.3f} ± {np.std(tn_brightness):.3f}")
        print(f"  FP: {np.mean(fp_brightness):.3f} ± {np.std(fp_brightness):.3f}")
        
        tn_edge = [x['img_features']['edge_density'] for x in true_negatives]
        fp_edge = [x['img_features']['edge_density'] for x in false_positives]
        print(f"\nEdge density:")
        print(f"  TN: {np.mean(tn_edge):.4f} ± {np.std(tn_edge):.4f}")
        print(f"  FP: {np.mean(fp_edge):.4f} ± {np.std(fp_edge):.4f}")
        
        tn_max_pred = [x['max_pred'] for x in true_negatives]
        fp_max_pred = [x['max_pred'] for x in false_positives]
        print(f"\nMax prediction:")
        print(f"  TN: {np.mean(tn_max_pred):.3f} ± {np.std(tn_max_pred):.3f}")
        print(f"  FP: {np.mean(fp_max_pred):.3f} ± {np.std(fp_max_pred):.3f}")
    
    # Actionable insights
    print("\n" + "=" * 60)
    print("ACTIONABLE INSIGHTS")
    print("=" * 60)
    
    if false_negatives:
        tiny_fn = len([x for x in false_negatives if x['mask_features'] and x['mask_features']['forgery_ratio'] < 0.005])
        print(f"\n1. Tiny forgeries (<0.5% of image) account for {tiny_fn}/{len(false_negatives)} FN ({tiny_fn/len(false_negatives)*100:.1f}%)")
    
    if false_positives:
        small_fp = len([x for x in false_positives if x['detected_pixels'] < 500])
        print(f"2. Small false detections (<500px) account for {small_fp}/{len(false_positives)} FP ({small_fp/len(false_positives)*100:.1f}%)")
        
        low_conf_fp = len([x for x in false_positives if x['max_pred'] < 0.5])
        print(f"3. Low confidence FP (max_pred < 0.5) account for {low_conf_fp}/{len(false_positives)} FP ({low_conf_fp/len(false_positives)*100:.1f}%)")


if __name__ == '__main__':
    main()
