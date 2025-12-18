#!/usr/bin/env python3
"""
Generate Submission for Scientific Image Forgery Detection
Uses v4 ensemble (4 models trained with hard negative mining + FP penalty)

Output format:
  case_id,annotation
  12345,authentic          # for clean images
  12345,10 0 5 1 3 0 ...   # RLE encoded mask for forged images
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import cv2
import segmentation_models_pytorch as smp
from pathlib import Path

import argparse
import pandas as pd

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 512
IMG_SIZE_768 = 768

# V4 models (trained with hard negatives + FP penalty)
V4_MODELS = [
    'highres_no_ela_v4_best.pth',
    'hard_negative_v4_best.pth',
    'high_recall_v4_best.pth',
    'enhanced_aug_v4_best.pth',
]

# Small forgery specialist (768px resolution, trained on missed forgeries)
SMALL_FORGERY_MODEL = 'small_forgery_v5_best.pth'


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


def prepare_image_rgb(image_bgr, size=IMG_SIZE):
    """Prepare 3-channel RGB image with ImageNet normalization."""
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (size, size))
    image = image.astype(np.float32) / 255.0
    image[..., 0] = (image[..., 0] - 0.485) / 0.229
    image[..., 1] = (image[..., 1] - 0.456) / 0.224
    image[..., 2] = (image[..., 2] - 0.406) / 0.225
    return torch.from_numpy(image.transpose(2, 0, 1)).float()


def get_image_brightness(image_bgr):
    """Calculate mean brightness of image (0-255 scale)."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)


def load_model(model_path):
    """Load a trained v4/v5 model."""
    base_model = smp.FPN(
        encoder_name='timm-efficientnet-b2',
        encoder_weights=None,
        in_channels=3,
        classes=1
    )
    model = AttentionFPN(base_model).to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    # Handle both formats: direct state_dict or wrapped checkpoint
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model


def load_small_forgery_model(model_path):
    """Load the small forgery specialist model (768px resolution)."""
    base_model = smp.FPN(
        encoder_name='timm-efficientnet-b2',
        encoder_weights=None,
        in_channels=3,
        classes=1
    )
    model = AttentionFPN(base_model).to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def mask_to_rle(mask):
    """
    Convert binary mask to RLE string.
    
    Args:
        mask: 2D numpy array (H, W) with 0s and 1s
        
    Returns:
        RLE string: "count0 val0 count1 val1 ..."
    """
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


class SubmissionGenerator:
    """V4 Ensemble for generating submission masks (4 models + optional specialist)."""
    
    def __init__(self, model_dir=None, use_specialist=True):
        if model_dir is None:
            model_dir = Path(__file__).parent.parent / 'models'
        else:
            model_dir = Path(model_dir)
        
        self.models = []
        self.specialist = None
        self.use_specialist = use_specialist
        
        print("Loading v4 ensemble (4 models)...")
        for model_name in V4_MODELS:
            model_path = model_dir / model_name
            if model_path.exists():
                model = load_model(str(model_path))
                self.models.append(model)
                print(f"  ✓ {model_name}")
            else:
                print(f"  ✗ {model_name} not found")
        
        if not self.models:
            raise RuntimeError("No v4 models found!")
        
        print(f"Loaded {len(self.models)} models")
        
        # Load small forgery specialist (768px resolution)
        if use_specialist:
            specialist_path = model_dir / SMALL_FORGERY_MODEL
            if specialist_path.exists():
                self.specialist = load_small_forgery_model(str(specialist_path))
                print(f"  ✓ {SMALL_FORGERY_MODEL} (specialist, 768px)")
            else:
                print(f"  ✗ {SMALL_FORGERY_MODEL} not found (specialist disabled)")
                self.use_specialist = False
    
    def predict(self, image_bgr, use_tta=False, use_multiscale=False, tta_mode='4x'):
        """Get ensemble prediction for an image.
        
        Args:
            image_bgr: BGR image array
            use_tta: If True, use test-time augmentation (flip/rotate)
            use_multiscale: If True, run at multiple resolutions (512 + 768)
            tta_mode: '4x' (flips only) or '8x' (flips + rotations)
        """
        if use_multiscale:
            return self._predict_multiscale(image_bgr, use_tta, tta_mode)
        if use_tta:
            return self._predict_tta(image_bgr, tta_mode)
        
        input_tensor = prepare_image_rgb(image_bgr).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            predictions = [torch.sigmoid(model(input_tensor)) for model in self.models]
            avg_pred = torch.stack(predictions).mean(0)
        
        return avg_pred.cpu().numpy()[0, 0]
    
    def _predict_at_scale(self, image_bgr, size):
        """Predict at a specific resolution."""
        input_tensor = prepare_image_rgb(image_bgr, size=size).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            predictions = [torch.sigmoid(model(input_tensor)) for model in self.models]
            avg_pred = torch.stack(predictions).mean(0).cpu().numpy()[0, 0]
        
        # Resize back to 512x512 for combining
        if size != 512:
            avg_pred = cv2.resize(avg_pred, (512, 512), interpolation=cv2.INTER_LINEAR)
        
        return avg_pred
    
    def _predict_multiscale(self, image_bgr, use_tta=False, tta_mode='4x'):
        """Predict at multiple scales (512 + 768) and combine."""
        scales = [512, 768]
        
        if use_tta:
            # Multi-scale + TTA - simplified, just run TTA at each scale
            all_preds = []
            for scale in scales:
                # Resize image to scale
                h, w = image_bgr.shape[:2]
                scaled_img = cv2.resize(image_bgr, (scale, scale))
                
                # Run TTA at this scale (temporarily override IMG_SIZE)
                pred = self._predict_tta(scaled_img, tta_mode)
                
                # Resize back to 512x512
                if scale != 512:
                    pred = cv2.resize(pred, (512, 512), interpolation=cv2.INTER_LINEAR)
                
                all_preds.append(pred)
            
            return np.max(all_preds, axis=0)
        else:
            # Multi-scale only
            preds = [self._predict_at_scale(image_bgr, s) for s in scales]
            return np.max(preds, axis=0)
    
    def _predict_tta(self, image_bgr, tta_mode='4x'):
        """Predict with test-time augmentation.
        
        Args:
            image_bgr: BGR image array
            tta_mode: '4x' (flips only) or '8x' (flips + 90/270 rotations)
        """
        # 4x TTA: original + 3 flips
        augmented = [
            (image_bgr, 'orig'),
            (cv2.flip(image_bgr, 1), 'hflip'),
            (cv2.flip(image_bgr, 0), 'vflip'),
            (cv2.flip(image_bgr, -1), 'hvflip'),
        ]
        
        # 8x TTA: add 90° and 270° rotations
        if tta_mode == '8x':
            rot90 = cv2.rotate(image_bgr, cv2.ROTATE_90_CLOCKWISE)
            rot270 = cv2.rotate(image_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
            augmented.extend([
                (rot90, 'rot90'),
                (cv2.flip(rot90, 1), 'rot90_hflip'),
                (rot270, 'rot270'),
                (cv2.flip(rot270, 1), 'rot270_hflip'),
            ])
        
        predictions = []
        for aug_img, aug_type in augmented:
            input_tensor = prepare_image_rgb(aug_img).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                preds = [torch.sigmoid(model(input_tensor)) for model in self.models]
                avg_pred = torch.stack(preds).mean(0).cpu().numpy()[0, 0]
            
            # Reverse the augmentation on the prediction
            if aug_type == 'hflip':
                avg_pred = np.flip(avg_pred, axis=1)
            elif aug_type == 'vflip':
                avg_pred = np.flip(avg_pred, axis=0)
            elif aug_type == 'hvflip':
                avg_pred = np.flip(np.flip(avg_pred, axis=0), axis=1)
            elif aug_type == 'rot90':
                avg_pred = cv2.rotate(avg_pred, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif aug_type == 'rot90_hflip':
                avg_pred = np.flip(avg_pred, axis=1)
                avg_pred = cv2.rotate(avg_pred, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif aug_type == 'rot270':
                avg_pred = cv2.rotate(avg_pred, cv2.ROTATE_90_CLOCKWISE)
            elif aug_type == 'rot270_hflip':
                avg_pred = np.flip(avg_pred, axis=1)
                avg_pred = cv2.rotate(avg_pred, cv2.ROTATE_90_CLOCKWISE)
            
            predictions.append(avg_pred)
        
        # Max of all TTA predictions (aggressive - boosts recall)
        return np.max(predictions, axis=0)
    
    def predict_specialist(self, image_bgr, use_tta=False):
        """Get specialist prediction at 768px resolution.
        
        The specialist is trained on missed forgeries and uses higher resolution
        to better detect small/subtle forgeries.
        
        Args:
            image_bgr: BGR image array
            use_tta: If True, use test-time augmentation (4x flips)
            
        Returns:
            Prediction probability map at 512x512 resolution
        """
        if self.specialist is None:
            return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
        
        if use_tta:
            # 4x TTA: original + 3 flips
            augmented = [
                (image_bgr, 'orig'),
                (cv2.flip(image_bgr, 1), 'hflip'),
                (cv2.flip(image_bgr, 0), 'vflip'),
                (cv2.flip(image_bgr, -1), 'hvflip'),
            ]
            
            predictions = []
            for aug_img, aug_type in augmented:
                # Resize to 768x768
                resized = cv2.resize(aug_img, (IMG_SIZE_768, IMG_SIZE_768))
                input_tensor = prepare_image_rgb(resized, size=IMG_SIZE_768).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    pred = torch.sigmoid(self.specialist(input_tensor)).cpu().numpy()[0, 0]
                
                # Resize to 512x512
                pred = cv2.resize(pred, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
                
                # Reverse augmentation
                if aug_type == 'hflip':
                    pred = np.flip(pred, axis=1)
                elif aug_type == 'vflip':
                    pred = np.flip(pred, axis=0)
                elif aug_type == 'hvflip':
                    pred = np.flip(np.flip(pred, axis=0), axis=1)
                
                predictions.append(pred)
            
            return np.max(predictions, axis=0)
        else:
            # Single prediction at 768px
            resized = cv2.resize(image_bgr, (IMG_SIZE_768, IMG_SIZE_768))
            input_tensor = prepare_image_rgb(resized, size=IMG_SIZE_768).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                pred = torch.sigmoid(self.specialist(input_tensor)).cpu().numpy()[0, 0]
            
            # Resize to 512x512 for combining with V4
            return cv2.resize(pred, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    
    def generate_mask(self, image_path, threshold=0.20, min_area=500, use_tta=False, use_multiscale=False, adaptive=True, min_confidence=0.0, tta_mode='4x', morphological=False, min_solidity=0.0):
        """
        Generate binary mask for an image.
        
        Args:
            image_path: Path to image file
            threshold: Detection threshold (default 0.20 for v4)
            min_area: Minimum forgery area in pixels (default 500)
            use_tta: Use test-time augmentation for better recall
            use_multiscale: Use multi-scale inference (512 + 768)
            adaptive: If True, use lower threshold/min_area for dark images
            min_confidence: Minimum max confidence to accept detection (0-1, default 0 = disabled)
            tta_mode: '4x' (flips only) or '8x' (flips + 90/270 rotations)
            morphological: If True, apply morphological cleanup to reduce noise
            min_solidity: Minimum solidity ratio (0-1) to accept region (0 = disabled)
        
        Returns:
            mask: Binary mask at original image resolution
            is_forged: Whether any forgery was detected
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        orig_h, orig_w = image.shape[:2]
        
        # Adaptive thresholding for dark images (they have smaller, harder to detect forgeries)
        # Tuned: Medium setting gives +15 net improvement over fixed threshold
        if adaptive:
            brightness = get_image_brightness(image)
            if brightness < 50:  # Very dark image
                # Lower threshold and min_area for dark images
                threshold = max(0.15, threshold - 0.08)   # e.g., 0.30 -> 0.22
                min_area = max(100, int(min_area / 2.5))  # e.g., 500 -> 200
            elif brightness < 80:  # Moderately dark
                threshold = max(0.20, threshold - 0.04)   # e.g., 0.30 -> 0.26
                min_area = max(150, int(min_area / 1.5))  # e.g., 500 -> 333
        
        # Get V4 ensemble prediction
        pred = self.predict(image, use_tta=use_tta, use_multiscale=use_multiscale, tta_mode=tta_mode)
        
        # Add specialist prediction (OR logic - catches forgeries V4 missed)
        if self.use_specialist and self.specialist is not None:
            specialist_pred = self.predict_specialist(image, use_tta=use_tta)
            # Use very high threshold for specialist to reduce FPs
            specialist_thresh = 0.95  # Very conservative - only high-confidence detections
            specialist_binary = (specialist_pred > specialist_thresh).astype(np.float32)
            # OR combination: add specialist detections to V4 prediction
            pred = np.maximum(pred, specialist_binary * (specialist_thresh + 0.01))
        
        # Threshold
        binary = (pred > threshold).astype(np.uint8)
        
        # Morphological cleanup: erode then dilate to remove noise, require solid regions
        if morphological and binary.max() > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            # Opening: erode then dilate - removes small noise
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            # Closing: dilate then erode - fills small holes
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Remove small regions and filter by solidity (at 512x512 resolution)
        if min_area > 0 or min_solidity > 0:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
            for i in range(1, num_labels):
                # Remove small regions
                if stats[i, cv2.CC_STAT_AREA] < min_area:
                    binary[labels == i] = 0
                    continue
                
                # Solidity filtering: reject fragmented/scattered detections
                if min_solidity > 0:
                    region_mask = (labels == i).astype(np.uint8)
                    contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        hull = cv2.convexHull(contours[0])
                        hull_area = cv2.contourArea(hull)
                        if hull_area > 0:
                            solidity = stats[i, cv2.CC_STAT_AREA] / hull_area
                            if solidity < min_solidity:
                                binary[labels == i] = 0
        
        # Confidence filtering: reject if max confidence in detected region is too low
        if min_confidence > 0 and binary.max() > 0:
            max_conf_in_detection = pred[binary > 0].max()
            if max_conf_in_detection < min_confidence:
                binary = np.zeros_like(binary)
        
        # Resize to original resolution
        mask = cv2.resize(binary, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        
        is_forged = mask.max() > 0
        
        return mask, is_forged


def generate_submission(input_dir, output_path, threshold=0.20, min_area=500, model_dir=None, use_tta=False, use_multiscale=False, adaptive=True, min_confidence=0.45, tta_mode='4x', morphological=False, use_specialist=True):
    """
    Generate submission CSV file.
    
    Args:
        input_dir: Directory containing test images
        output_path: Path to save submission CSV
        threshold: Detection threshold (default 0.20)
        min_area: Minimum forgery area in pixels (default 500)
        model_dir: Directory containing model files
        use_tta: Use test-time augmentation (4x slower but better recall)
        use_multiscale: Use multi-scale inference (512 + 768)
        adaptive: Use adaptive threshold/min_area for dark images (default True)
        min_confidence: Minimum confidence to accept detection (default 0.45)
        tta_mode: '4x' (flips) or '8x' (flips + rotations)
        morphological: Apply morphological cleanup (default False)
        use_specialist: Use small forgery specialist model (default True)
    """
    input_dir = Path(input_dir)
    output_path = Path(output_path)
    
    # Get all images
    image_files = sorted([
        f for f in input_dir.iterdir()
        if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
    ])
    
    print(f"Found {len(image_files)} images in {input_dir}")
    print(f"Threshold: {threshold}, Min area: {min_area}, TTA: {use_tta} ({tta_mode}), Adaptive: {adaptive}, Min conf: {min_confidence}, Morph: {morphological}, Specialist: {use_specialist}")
    print()
    
    # Initialize generator
    generator = SubmissionGenerator(model_dir, use_specialist=use_specialist)
    print()
    
    # Process images
    results = []
    forged_count = 0
    
    for img_path in image_files:
        case_id = img_path.stem
        
        try:
            mask, is_forged = generator.generate_mask(img_path, threshold, min_area, use_tta=use_tta, use_multiscale=use_multiscale, adaptive=adaptive, min_confidence=min_confidence, tta_mode=tta_mode, morphological=morphological)
            
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
    print("SUBMISSION GENERATED")
    print("=" * 60)
    print(f"Total images: {len(image_files)}")
    print(f"Forged detected: {forged_count} ({forged_count/len(image_files)*100:.1f}%)")
    print(f"Authentic: {len(image_files) - forged_count}")
    print(f"Saved to: {output_path}")
    
    return df


# Mode presets (calibrated for v4 + TTA)
MODES = {
    'high-precision': {'threshold': 0.45, 'min_area': 1000, 'description': '~65% recall, ~1.5% FP'},
    'balanced': {'threshold': 0.35, 'min_area': 500, 'description': '~68% recall, ~2.5% FP'},
    'default': {'threshold': 0.30, 'min_area': 500, 'description': '~70% recall, ~3.5% FP'},
    'high-recall': {'threshold': 0.20, 'min_area': 500, 'description': '~75% recall, ~12% FP'},
    'max-recall': {'threshold': 0.10, 'min_area': 300, 'description': '~76% recall, ~15% FP'},
}


def main():
    parser = argparse.ArgumentParser(
        description='Generate submission for Scientific Image Forgery Detection (v4 ensemble)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes (use --mode):
  high-precision : ~58%% recall, ~0%% FP (threshold=0.30, min_area=1000)
  balanced       : ~60%% recall, ~1%% FP (threshold=0.25, min_area=500)
  default        : ~63%% recall, ~4%% FP (threshold=0.20, min_area=500)
  high-recall    : ~64%% recall, ~6%% FP (threshold=0.15, min_area=500)
  max-recall     : ~66%% recall, ~8%% FP (threshold=0.10, min_area=300)

Examples:
  python script_11_generate_submission.py --input test_images/ --output submission.csv
  python script_2_generate_submission.py --input validation_images/ --output submission.csv --mode max-recall
  python script_2_generate_submission.py --input validation_images/ --output submission.csv --threshold 0.15 --min-area 400
        """
    )
    parser.add_argument('--input', '-i', type=str, default='validation_images',
                        help='Input directory containing test images (default: validation_images)')
    parser.add_argument('--output', '-o', type=str, default='submission.csv',
                        help='Output CSV file path (default: submission.csv)')
    parser.add_argument('--mode', '-m', type=str, choices=list(MODES.keys()),
                        help='Preset mode (overrides threshold and min-area)')
    parser.add_argument('--threshold', '-t', type=float, default=0.35,
                        help='Detection threshold (default: 0.35)')
    parser.add_argument('--min-area', type=int, default=300,
                        help='Minimum forgery area in pixels (default: 300)')
    parser.add_argument('--tta', action='store_true', default=True,
                        help='Use test-time augmentation (default: enabled)')
    parser.add_argument('--no-tta', action='store_true',
                        help='Disable test-time augmentation')
    parser.add_argument('--tta-mode', type=str, choices=['4x', '8x'], default='4x',
                        help='TTA mode: 4x (flips only) or 8x (flips + rotations)')
    parser.add_argument('--multiscale', action='store_true',
                        help='Use multi-scale inference (512 + 768)')
    parser.add_argument('--morphological', action='store_true',
                        help='Apply morphological cleanup to reduce noise')
    parser.add_argument('--no-adaptive', action='store_true',
                        help='Disable adaptive thresholding for dark images')
    parser.add_argument('--min-confidence', type=float, default=0.45,
                        help='Minimum confidence to accept detection (default: 0.45, 0=disabled)')
    parser.add_argument('--model-dir', type=str, default=None,
                        help='Directory containing model files')
    parser.add_argument('--specialist', action='store_true',
                        help='Enable small forgery specialist model (default: disabled)')
    
    args = parser.parse_args()
    
    # Handle TTA flag
    use_tta = args.tta and not args.no_tta
    
    # Apply mode preset if specified
    threshold = args.threshold
    min_area = args.min_area
    if args.mode:
        mode_cfg = MODES[args.mode]
        threshold = mode_cfg['threshold']
        min_area = mode_cfg['min_area']
        print(f"Mode: {args.mode} - {mode_cfg['description']}")
    
    generate_submission(
        args.input,
        args.output,
        threshold=threshold,
        min_area=min_area,
        model_dir=args.model_dir,
        use_tta=use_tta,
        use_multiscale=args.multiscale,
        adaptive=not args.no_adaptive,
        min_confidence=args.min_confidence,
        tta_mode=args.tta_mode,
        morphological=args.morphological,
        use_specialist=args.specialist  # Disabled by default
    )


if __name__ == '__main__':
    main()
