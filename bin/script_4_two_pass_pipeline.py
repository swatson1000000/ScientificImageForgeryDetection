#!/usr/bin/env python3
"""
Two-Pass Forgery Detection Pipeline

This script implements a two-stage inference pipeline for scientific image forgery detection:

Pass 1 (Classifier): Fast binary classification to filter out obvious authentic images
Pass 2 (Segmentation): Detailed segmentation ensemble with TTA for forgery localization

This approach provides:
- Fast inference by skipping segmentation for clearly authentic images
- High accuracy through ensemble segmentation with test-time augmentation
- Detailed forgery masks for localization

Usage:
    # Single image
    python bin/two_pass_pipeline.py --input image.png --output results/
    
    # Directory of images
    python bin/two_pass_pipeline.py --input images_dir/ --output results/
    
    # Batch processing with CSV output
    python bin/two_pass_pipeline.py --input images/ --output results/ --csv results.csv
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
import json
import csv
from datetime import datetime
from typing import Tuple, List, Dict, Optional, Union

# ============================================================================
# CONFIGURATION
# ============================================================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Image sizes
CLASSIFIER_SIZE = 384
SEG_SIZE = 512

# Default thresholds (can be overridden via CLI)
DEFAULT_CLASSIFIER_THRESHOLD = 0.25  # Pass to stage 2 if prob >= this
DEFAULT_SEG_THRESHOLD = 0.35         # Pixel-level forgery threshold
DEFAULT_MIN_AREA = 300               # Minimum connected component area

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
MODELS_DIR = PROJECT_DIR / 'models'

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


# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

class AttentionGate(nn.Module):
    """Spatial attention gate for focusing on forgery regions."""
    def __init__(self, in_channels: int):
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.conv(x)


class AttentionFPN(nn.Module):
    """FPN with attention gate for forgery segmentation."""
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base = base_model
        self.attention = AttentionGate(1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attention(self.base(x))


class ForgeryClassifier(nn.Module):
    """Binary classifier for authentic vs forged detection."""
    def __init__(self, backbone_name: str = 'efficientnet_b2', pretrained: bool = False):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        num_features = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)


# ============================================================================
# PIPELINE CLASS
# ============================================================================

class TwoPassPipeline:
    """
    Two-pass forgery detection pipeline.
    
    Pass 1: Binary classifier for fast filtering
    Pass 2: Ensemble segmentation with TTA for detailed detection
    """
    
    def __init__(
        self,
        classifier_threshold: float = DEFAULT_CLASSIFIER_THRESHOLD,
        seg_threshold: float = DEFAULT_SEG_THRESHOLD,
        min_area: int = DEFAULT_MIN_AREA,
        use_tta: bool = True,
        verbose: bool = True
    ):
        self.classifier_threshold = classifier_threshold
        self.seg_threshold = seg_threshold
        self.min_area = min_area
        self.use_tta = use_tta
        self.verbose = verbose
        
        self.classifier = None
        self.seg_models = []
        
        self._load_models()
    
    def _log(self, msg: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(msg)
    
    def _load_models(self):
        """Load classifier and segmentation models."""
        self._log("Loading models...")
        
        # Load classifier
        classifier_path = MODELS_DIR / 'binary_classifier_best.pth'
        if classifier_path.exists():
            self.classifier = ForgeryClassifier(pretrained=False)
            checkpoint = torch.load(classifier_path, map_location=DEVICE, weights_only=False)
            self.classifier.load_state_dict(checkpoint['model_state_dict'])
            self.classifier = self.classifier.to(DEVICE)
            self.classifier.eval()
            self._log(f"  ✓ Loaded classifier: {classifier_path.name}")
        else:
            self._log(f"  ⚠ Classifier not found at {classifier_path}")
            self._log("    Pipeline will skip Pass 1 and run segmentation on all images")
        
        # Load segmentation ensemble
        seg_model_names = [
            'highres_no_ela_v4_best.pth',
            'hard_negative_v4_best.pth',
            'high_recall_v4_best.pth',
            'enhanced_aug_v4_best.pth'
        ]
        
        for name in seg_model_names:
            path = MODELS_DIR / name
            if path.exists():
                base = smp.FPN(
                    encoder_name='timm-efficientnet-b2',
                    encoder_weights=None,
                    in_channels=3,
                    classes=1
                )
                model = AttentionFPN(base).to(DEVICE)
                checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                self.seg_models.append(model)
                self._log(f"  ✓ Loaded segmentation: {name}")
        
        if len(self.seg_models) == 0:
            raise RuntimeError("No segmentation models found. Please train models first.")
        
        self._log(f"  Total: 1 classifier + {len(self.seg_models)} segmentation models")
        self._log("")
    
    def _preprocess_classifier(self, img_path: Union[str, Path]) -> torch.Tensor:
        """Preprocess image for classifier."""
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Could not read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (CLASSIFIER_SIZE, CLASSIFIER_SIZE))
        img = img.astype(np.float32) / 255.0
        # ImageNet normalization
        img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        img = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()
        return img.to(DEVICE)
    
    def _preprocess_segmentation(self, img_path: Union[str, Path]) -> Tuple[torch.Tensor, np.ndarray, Tuple[int, int]]:
        """Preprocess image for segmentation."""
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Could not read image: {img_path}")
        original_size = (img.shape[1], img.shape[0])  # (width, height)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (SEG_SIZE, SEG_SIZE))
        img_norm = img_resized.astype(np.float32) / 255.0
        # ImageNet normalization
        img_norm = (img_norm - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).float()
        return tensor.to(DEVICE), img, original_size
    
    def _apply_tta(self, model: nn.Module, img_tensor: torch.Tensor) -> torch.Tensor:
        """Apply 4x test-time augmentation (flips) with MAX aggregation."""
        preds = []
        
        with torch.no_grad():
            # Original
            pred = torch.sigmoid(model(img_tensor))
            preds.append(pred)
            
            # Horizontal flip
            pred = torch.sigmoid(model(torch.flip(img_tensor, dims=[3])))
            pred = torch.flip(pred, dims=[3])
            preds.append(pred)
            
            # Vertical flip
            pred = torch.sigmoid(model(torch.flip(img_tensor, dims=[2])))
            pred = torch.flip(pred, dims=[2])
            preds.append(pred)
            
            # Both flips
            pred = torch.sigmoid(model(torch.flip(img_tensor, dims=[2, 3])))
            pred = torch.flip(pred, dims=[2, 3])
            preds.append(pred)
        
        return torch.stack(preds).max(dim=0)[0]
    
    def _get_adaptive_threshold(self, img_bgr: np.ndarray, base_threshold: float) -> float:
        """Compute adaptive threshold based on image brightness."""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        
        if brightness < 50:
            return max(0.20, base_threshold - 0.10)
        elif brightness < 80:
            return max(0.25, base_threshold - 0.05)
        elif brightness > 200:
            return min(0.50, base_threshold + 0.05)
        return base_threshold
    
    def _remove_small_components(self, mask: np.ndarray) -> np.ndarray:
        """Remove connected components smaller than min_area."""
        if self.min_area <= 0:
            return mask
        
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        cleaned = mask.copy()
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < self.min_area:
                cleaned[labels == i] = 0
        return cleaned
    
    def run_pass1(self, img_path: Union[str, Path]) -> Tuple[bool, float]:
        """
        Pass 1: Binary classification.
        
        Returns:
            (passed, probability): Whether to proceed to Pass 2, and the forgery probability
        """
        if self.classifier is None:
            # No classifier available, always proceed to Pass 2
            return True, 1.0
        
        with torch.no_grad():
            img = self._preprocess_classifier(img_path)
            logit = self.classifier(img)
            prob = torch.sigmoid(logit).item()
        
        passed = prob >= self.classifier_threshold
        return passed, prob
    
    def run_pass2(self, img_path: Union[str, Path]) -> Tuple[bool, np.ndarray, float, int]:
        """
        Pass 2: Ensemble segmentation with TTA.
        
        Returns:
            (is_forged, mask, max_prob, area): Detection result, mask, max probability, forgery area
        """
        img_tensor, img_bgr, original_size = self._preprocess_segmentation(img_path)
        
        # Ensemble prediction with TTA
        all_preds = []
        with torch.no_grad():
            for model in self.seg_models:
                if self.use_tta:
                    pred = self._apply_tta(model, img_tensor)
                else:
                    pred = torch.sigmoid(model(img_tensor))
                all_preds.append(pred)
        
        # Average ensemble predictions
        mask = torch.stack(all_preds).mean(dim=0).squeeze().cpu().numpy()
        max_prob = float(mask.max())
        
        # Adaptive threshold
        threshold = self._get_adaptive_threshold(img_bgr, self.seg_threshold)
        
        # Apply threshold and create binary mask
        binary_mask = (mask > threshold).astype(np.uint8)
        
        # Remove small components
        binary_mask = self._remove_small_components(binary_mask)
        
        # Compute stats
        is_forged = binary_mask.max() > 0
        area = int(binary_mask.sum())
        
        return is_forged, binary_mask, max_prob, area
    
    def predict(self, img_path: Union[str, Path]) -> Dict:
        """
        Run full two-pass pipeline on an image.
        
        Returns dict with:
            - is_forged: bool
            - classifier_prob: float (Pass 1 probability)
            - passed_to_segmentation: bool (whether Pass 2 was run)
            - seg_max_prob: float (maximum segmentation probability)
            - forgery_area: int (number of forged pixels)
            - mask: np.ndarray or None (segmentation mask if Pass 2 was run)
        """
        img_path = Path(img_path)
        
        result = {
            'image': str(img_path),
            'filename': img_path.name,
            'is_forged': False,
            'classifier_prob': 0.0,
            'passed_to_segmentation': False,
            'seg_max_prob': 0.0,
            'forgery_area': 0,
            'mask': None
        }
        
        # Pass 1: Classifier
        passed, prob = self.run_pass1(img_path)
        result['classifier_prob'] = prob
        
        if not passed:
            # Classified as authentic, skip segmentation
            result['is_forged'] = False
            return result
        
        # Pass 2: Segmentation
        result['passed_to_segmentation'] = True
        is_forged, mask, max_prob, area = self.run_pass2(img_path)
        
        result['is_forged'] = is_forged
        result['seg_max_prob'] = max_prob
        result['forgery_area'] = area
        result['mask'] = mask
        
        return result
    
    def process_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        save_masks: bool = True,
        save_overlays: bool = True
    ) -> List[Dict]:
        """
        Process all images in a directory.
        
        Args:
            input_dir: Directory containing images
            output_dir: Directory to save results (optional)
            save_masks: Whether to save binary masks
            save_overlays: Whether to save overlay visualizations
            
        Returns:
            List of prediction results
        """
        input_dir = Path(input_dir)
        
        # Find all images
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp']
        image_files = []
        for ext in extensions:
            image_files.extend(input_dir.glob(ext))
            image_files.extend(input_dir.glob(ext.upper()))
        image_files = sorted(set(image_files))
        
        if not image_files:
            self._log(f"No images found in {input_dir}")
            return []
        
        self._log(f"Processing {len(image_files)} images...")
        
        # Setup output directory
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            if save_masks:
                (output_dir / 'masks').mkdir(exist_ok=True)
            if save_overlays:
                (output_dir / 'overlays').mkdir(exist_ok=True)
        
        results = []
        stats = {'total': 0, 'forged': 0, 'passed_to_seg': 0}
        
        for i, img_path in enumerate(image_files):
            try:
                result = self.predict(img_path)
                results.append(result)
                
                stats['total'] += 1
                if result['is_forged']:
                    stats['forged'] += 1
                if result['passed_to_segmentation']:
                    stats['passed_to_seg'] += 1
                
                # Save outputs
                if output_dir and result['mask'] is not None:
                    stem = img_path.stem
                    
                    if save_masks:
                        mask_path = output_dir / 'masks' / f'{stem}_mask.png'
                        cv2.imwrite(str(mask_path), result['mask'] * 255)
                    
                    if save_overlays and result['is_forged']:
                        overlay = self._create_overlay(img_path, result['mask'])
                        overlay_path = output_dir / 'overlays' / f'{stem}_overlay.png'
                        cv2.imwrite(str(overlay_path), overlay)
                
                # Progress - show every 100 images only
                if (i + 1) % 100 == 0 or i + 1 == len(image_files):
                    self._log(f"  Processed {i+1}/{len(image_files)} images...")
                
            except Exception as e:
                self._log(f"  [{i+1}/{len(image_files)}] {img_path.name}: ERROR - {e}")
                results.append({
                    'image': str(img_path),
                    'filename': img_path.name,
                    'error': str(e)
                })
        
        # Summary
        self._log("")
        self._log("=" * 60)
        self._log("SUMMARY")
        self._log("=" * 60)
        self._log(f"Total images processed: {stats['total']}")
        self._log(f"Passed to segmentation: {stats['passed_to_seg']} ({stats['passed_to_seg']/max(1,stats['total'])*100:.1f}%)")
        self._log(f"Detected as forged: {stats['forged']} ({stats['forged']/max(1,stats['total'])*100:.1f}%)")
        self._log(f"Detected as authentic: {stats['total'] - stats['forged']} ({(stats['total']-stats['forged'])/max(1,stats['total'])*100:.1f}%)")
        
        return results
    
    def _create_overlay(self, img_path: Union[str, Path], mask: np.ndarray) -> np.ndarray:
        """Create visualization overlay with forgery regions highlighted."""
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        
        # Resize mask to original image size
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Create colored overlay
        overlay = img.copy()
        overlay[mask_resized > 0] = [0, 0, 255]  # Red for forged regions
        
        # Blend with original
        result = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
        
        # Draw contours
        contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)  # Green contours
        
        return result


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Two-Pass Forgery Detection Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single image
  python two_pass_pipeline.py --input image.png --output results/
  
  # Process directory
  python two_pass_pipeline.py --input images/ --output results/
  
  # Export results to CSV
  python two_pass_pipeline.py --input images/ --output results/ --csv results.csv
  
  # Adjust thresholds for higher precision
  python two_pass_pipeline.py --input images/ --classifier-threshold 0.5 --seg-threshold 0.5
        """
    )
    
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input image or directory')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output directory for results')
    parser.add_argument('--csv', type=str, default=None,
                        help='Path to save CSV results')
    parser.add_argument('--json', type=str, default=None,
                        help='Path to save JSON results')
    
    # Threshold options
    parser.add_argument('--classifier-threshold', type=float, default=DEFAULT_CLASSIFIER_THRESHOLD,
                        help=f'Classifier threshold (default: {DEFAULT_CLASSIFIER_THRESHOLD})')
    parser.add_argument('--seg-threshold', type=float, default=DEFAULT_SEG_THRESHOLD,
                        help=f'Segmentation threshold (default: {DEFAULT_SEG_THRESHOLD})')
    parser.add_argument('--min-area', type=int, default=DEFAULT_MIN_AREA,
                        help=f'Minimum forgery area in pixels (default: {DEFAULT_MIN_AREA})')
    
    # Processing options
    parser.add_argument('--no-tta', action='store_true',
                        help='Disable test-time augmentation (faster but less accurate)')
    parser.add_argument('--no-masks', action='store_true',
                        help='Do not save binary masks')
    parser.add_argument('--no-overlays', action='store_true',
                        help='Do not save overlay visualizations')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Minimal output')
    
    args = parser.parse_args()
    
    # Header
    if not args.quiet:
        print("=" * 70)
        print("TWO-PASS FORGERY DETECTION PIPELINE")
        print("=" * 70)
        print(f"Device: {DEVICE}")
        print(f"Input: {args.input}")
        print(f"Output: {args.output or '(none)'}")
        print(f"Classifier threshold: {args.classifier_threshold}")
        print(f"Segmentation threshold: {args.seg_threshold}")
        print(f"Min area: {args.min_area}")
        print(f"TTA: {'enabled' if not args.no_tta else 'disabled'}")
        print()
    
    # Initialize pipeline
    pipeline = TwoPassPipeline(
        classifier_threshold=args.classifier_threshold,
        seg_threshold=args.seg_threshold,
        min_area=args.min_area,
        use_tta=not args.no_tta,
        verbose=not args.quiet
    )
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single image
        result = pipeline.predict(input_path)
        
        status = "FORGED" if result['is_forged'] else "AUTHENTIC"
        print(f"\nResult: {status}")
        print(f"  Classifier probability: {result['classifier_prob']:.4f}")
        if result['passed_to_segmentation']:
            print(f"  Segmentation max prob: {result['seg_max_prob']:.4f}")
            print(f"  Forgery area: {result['forgery_area']} pixels")
        
        # Save outputs
        if args.output:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            if result['mask'] is not None:
                if not args.no_masks:
                    mask_path = output_dir / f'{input_path.stem}_mask.png'
                    cv2.imwrite(str(mask_path), result['mask'] * 255)
                    print(f"  Saved mask: {mask_path}")
                
                if not args.no_overlays and result['is_forged']:
                    overlay = pipeline._create_overlay(input_path, result['mask'])
                    overlay_path = output_dir / f'{input_path.stem}_overlay.png'
                    cv2.imwrite(str(overlay_path), overlay)
                    print(f"  Saved overlay: {overlay_path}")
        
        results = [result]
        
    elif input_path.is_dir():
        # Directory of images
        results = pipeline.process_directory(
            input_path,
            output_dir=args.output,
            save_masks=not args.no_masks,
            save_overlays=not args.no_overlays
        )
    else:
        print(f"Error: Input path does not exist: {args.input}")
        sys.exit(1)
    
    # Save submission CSV with RLE annotation
    if args.csv and results:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        submission_rows = []
        for r in results:
            if 'error' in r:
                annotation = 'authentic'
            elif r.get('is_forged') and r.get('mask') is not None:
                annotation = mask_to_rle(r['mask'])
            else:
                annotation = 'authentic'
            submission_rows.append({'filename': r.get('filename', ''), 'annotation': annotation})
        fieldnames = ['filename', 'annotation']
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in submission_rows:
                writer.writerow(row)
        if not args.quiet:
            print(f"\nSaved submission CSV: {csv_path}")
    
    # Save JSON
    if args.json and results:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Remove mask arrays from JSON output and convert numpy types to native Python types
        def convert_types(obj):
            """Convert numpy types to native Python types for JSON serialization."""
            if isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_types(item) for item in obj]
            return obj
        
        json_results = []
        for r in results:
            r_copy = {k: v for k, v in r.items() if k != 'mask'}
            r_copy = convert_types(r_copy)
            json_results.append(r_copy)
        
        with open(json_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'config': {
                    'classifier_threshold': args.classifier_threshold,
                    'seg_threshold': args.seg_threshold,
                    'min_area': args.min_area,
                    'tta': not args.no_tta
                },
                'results': json_results
            }, f, indent=2)
        
        if not args.quiet:
            print(f"Saved JSON: {json_path}")
    
    if not args.quiet:
        print("\nDone!")


if __name__ == '__main__':
    main()
