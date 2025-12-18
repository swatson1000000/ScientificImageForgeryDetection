#!/usr/bin/env python3
"""
Two-Stage Submission Generator for Scientific Image Forgery Detection

Pipeline:
  Stage 1: Binary classifier filters images (forged probability > threshold → proceed)
  Stage 2: 4-model V4 ensemble generates mask for filtered images

Optimal config from validation:
  - Classifier threshold: 0.25
  - 4-model ensemble: Net 2033 (TP: 2173, FP: 140)
  - vs Single V4: Net 1934 (TP: 2079, FP: 145)
  - Ensemble improvement: +99 net points
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
import pandas as pd

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Image sizes
CLASSIFIER_SIZE = 384
SEG_SIZE = 512

# Optimal threshold from validation sweep
DEFAULT_CLASSIFIER_THRESHOLD = 0.25

# V4 ensemble models
V4_MODELS = [
    'highres_no_ela_v4_best.pth',
    'hard_negative_v4_best.pth',
    'high_recall_v4_best.pth',
    'enhanced_aug_v4_best.pth',
]


class ForgeryClassifier(nn.Module):
    """Binary classifier: forged (1) vs authentic (0)."""
    
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
    """Attention gate for refining predictions."""
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
    """FPN with attention gating."""
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        self.attention = AttentionGate(1)
    
    def forward(self, x):
        return self.attention(self.base(x))


def load_classifier(model_dir, device):
    """Load the binary classifier model."""
    classifier = ForgeryClassifier('efficientnet_b2').to(device)
    classifier_path = model_dir / "binary_classifier_best.pth"
    checkpoint = torch.load(classifier_path, map_location=device, weights_only=False)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier.eval()
    return classifier


def load_segmentation_model(model_path, device):
    """Load a single V4 segmentation model."""
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


def load_ensemble(model_dir, device):
    """Load all 4 V4 ensemble models."""
    models = []
    for model_name in V4_MODELS:
        model_path = model_dir / model_name
        if model_path.exists():
            model = load_segmentation_model(model_path, device)
            models.append(model)
            print(f"    ✓ {model_name}")
        else:
            print(f"    ✗ {model_name} not found")
    return models


def preprocess_for_classifier(image_bgr, device):
    """Preprocess image for classifier (384x384)."""
    img_resized = cv2.resize(image_bgr, (CLASSIFIER_SIZE, CLASSIFIER_SIZE))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_norm = (img_norm - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    return tensor


def preprocess_for_segmentation(image_bgr, device):
    """Preprocess image for segmentation (512x512)."""
    img_resized = cv2.resize(image_bgr, (SEG_SIZE, SEG_SIZE))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_norm = (img_norm - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    return tensor


def get_image_brightness(image_bgr):
    """Calculate mean brightness of image (0-255 scale)."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)


def get_adaptive_threshold(image_bgr, base_threshold=0.35):
    """Adaptive threshold based on image brightness."""
    brightness = get_image_brightness(image_bgr)
    if brightness < 50:
        return max(0.20, base_threshold - 0.10)
    elif brightness < 80:
        return max(0.25, base_threshold - 0.05)
    elif brightness > 200:
        return min(0.50, base_threshold + 0.05)
    return base_threshold


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


def mask_to_rle(mask):
    """Convert binary mask to RLE string."""
    flat = mask.flatten()
    
    if len(flat) == 0:
        return ""
    
    runs = []
    current_val = flat[0]
    count = 1
    
    for i in range(1, len(flat)):
        if flat[i] == current_val:
            count += 1
        else:
            runs.append(f"{count} {int(current_val)}")
            current_val = flat[i]
            count = 1
    
    runs.append(f"{count} {int(current_val)}")
    
    return " ".join(runs)


class TwoStageGenerator:
    """Two-stage pipeline: classifier + 4-model ensemble."""
    
    def __init__(self, model_dir=None, classifier_threshold=DEFAULT_CLASSIFIER_THRESHOLD):
        if model_dir is None:
            model_dir = Path(__file__).parent.parent / 'models'
        else:
            model_dir = Path(model_dir)
        
        self.classifier_threshold = classifier_threshold
        
        print("Loading two-stage pipeline...")
        print(f"  Classifier threshold: {classifier_threshold}")
        
        # Load classifier
        self.classifier = load_classifier(model_dir, DEVICE)
        print(f"  ✓ Binary classifier loaded")
        
        # Load 4-model ensemble
        print(f"  Loading 4-model ensemble:")
        self.seg_models = load_ensemble(model_dir, DEVICE)
        print(f"  ✓ Loaded {len(self.seg_models)} ensemble models")
    
    def classify(self, image_bgr):
        """Run binary classifier on image. Returns probability of being forged."""
        img_tensor = preprocess_for_classifier(image_bgr, DEVICE)
        with torch.no_grad():
            logit = self.classifier(img_tensor)
            prob = torch.sigmoid(logit).item()
        return prob
    
    def segment(self, image_bgr, seg_threshold=0.35, min_area=300, use_tta=True, adaptive=True):
        """Run 4-model ensemble segmentation. Returns binary mask and is_forged flag."""
        orig_h, orig_w = image_bgr.shape[:2]
        
        # Adaptive threshold
        if adaptive:
            seg_threshold = get_adaptive_threshold(image_bgr, seg_threshold)
        
        # Preprocess
        img_tensor = preprocess_for_segmentation(image_bgr, DEVICE)
        
        # Run ensemble with TTA
        all_preds = []
        for model in self.seg_models:
            if use_tta:
                pred = apply_tta(model, img_tensor)
            else:
                with torch.no_grad():
                    pred = torch.sigmoid(model(img_tensor))
            all_preds.append(pred)
        
        # Mean of all ensemble models
        avg_pred = torch.stack(all_preds).mean(dim=0)
        
        # Convert to numpy
        pred_np = avg_pred.cpu().numpy()[0, 0]
        
        # Threshold
        binary = (pred_np > seg_threshold).astype(np.uint8)
        
        # Remove small regions
        if min_area > 0:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] < min_area:
                    binary[labels == i] = 0
        
        # Resize to original resolution
        mask = cv2.resize(binary, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        is_forged = mask.max() > 0
        
        return mask, is_forged
    
    def generate_mask(self, image_path, seg_threshold=0.35, min_area=300, use_tta=True, adaptive=True):
        """
        Two-stage pipeline:
        1. Classifier filters likely authentic images
        2. Segmentation runs on remaining images
        
        Returns: (mask, is_forged, was_filtered)
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Stage 1: Classifier
        classifier_prob = self.classify(image)
        
        if classifier_prob < self.classifier_threshold:
            # Classifier says authentic - skip segmentation
            orig_h, orig_w = image.shape[:2]
            return np.zeros((orig_h, orig_w), dtype=np.uint8), False, True
        
        # Stage 2: Segmentation
        mask, is_forged = self.segment(image, seg_threshold, min_area, use_tta, adaptive)
        
        return mask, is_forged, False


def generate_submission(input_dir, output_path, classifier_threshold=DEFAULT_CLASSIFIER_THRESHOLD,
                        seg_threshold=0.35, min_area=300, model_dir=None, use_tta=True, adaptive=True):
    """Generate submission CSV using two-stage pipeline."""
    input_dir = Path(input_dir)
    output_path = Path(output_path)
    
    # Get all images
    image_files = sorted([
        f for f in input_dir.iterdir()
        if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
    ])
    
    print(f"\nFound {len(image_files)} images in {input_dir}")
    print(f"Classifier threshold: {classifier_threshold}")
    print(f"Seg threshold: {seg_threshold}, Min area: {min_area}")
    print(f"TTA: {use_tta}, Adaptive: {adaptive}")
    print()
    
    # Initialize generator
    generator = TwoStageGenerator(model_dir, classifier_threshold)
    print()
    
    # Process images
    results = []
    forged_count = 0
    filtered_count = 0
    
    for img_path in image_files:
        case_id = img_path.stem
        
        try:
            mask, is_forged, was_filtered = generator.generate_mask(
                img_path, seg_threshold, min_area, use_tta, adaptive
            )
            
            if was_filtered:
                filtered_count += 1
            
            if is_forged:
                annotation = mask_to_rle(mask)
                forged_count += 1
            else:
                annotation = "authentic"
            
            results.append({'case_id': case_id, 'annotation': annotation})
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            results.append({'case_id': case_id, 'annotation': 'authentic'})
    
    # Create DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    
    print()
    print("=" * 60)
    print("TWO-STAGE SUBMISSION GENERATED")
    print("=" * 60)
    print(f"Total images: {len(image_files)}")
    print(f"Filtered by classifier: {filtered_count} ({filtered_count/len(image_files)*100:.1f}%)")
    print(f"Passed to segmentation: {len(image_files) - filtered_count}")
    print(f"Forged detected: {forged_count} ({forged_count/len(image_files)*100:.1f}%)")
    print(f"Authentic: {len(image_files) - forged_count}")
    print(f"Saved to: {output_path}")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Two-Stage Submission Generator (Classifier + Segmentation)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Two-Stage Pipeline:
  Stage 1: Binary classifier filters likely authentic images (fast)
  Stage 2: 4-model V4 ensemble runs on remaining images (detailed)

Optimal config (from validation):
  - Classifier threshold: 0.25
  - 4-model ensemble Net: 2033 (TP: 2173, FP: 140)
  - Single V4 Net: 1934 (TP: 2079, FP: 145)
  - Ensemble improvement: +99 net points

Examples:
  python script_3_two_stage_submission.py --input test_images/ --output submission.csv
  python script_3_two_stage_submission.py --input test_images/ --output submission.csv --classifier-threshold 0.30
        """
    )
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input directory containing test images')
    parser.add_argument('--output', '-o', type=str, default='submission.csv',
                        help='Output CSV file path (default: submission.csv)')
    parser.add_argument('--classifier-threshold', '-c', type=float, default=DEFAULT_CLASSIFIER_THRESHOLD,
                        help=f'Classifier threshold for filtering (default: {DEFAULT_CLASSIFIER_THRESHOLD})')
    parser.add_argument('--seg-threshold', '-s', type=float, default=0.35,
                        help='Segmentation threshold (default: 0.35)')
    parser.add_argument('--min-area', type=int, default=300,
                        help='Minimum forgery area in pixels (default: 300)')
    parser.add_argument('--no-tta', action='store_true',
                        help='Disable test-time augmentation')
    parser.add_argument('--no-adaptive', action='store_true',
                        help='Disable adaptive thresholding')
    parser.add_argument('--model-dir', type=str, default=None,
                        help='Directory containing model files')
    
    args = parser.parse_args()
    
    generate_submission(
        args.input,
        args.output,
        classifier_threshold=args.classifier_threshold,
        seg_threshold=args.seg_threshold,
        min_area=args.min_area,
        model_dir=args.model_dir,
        use_tta=not args.no_tta,
        adaptive=not args.no_adaptive
    )


if __name__ == '__main__':
    main()
