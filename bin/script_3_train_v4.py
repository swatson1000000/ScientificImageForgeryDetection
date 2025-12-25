#!/usr/bin/env python3
"""
Hard Negative Mining V4 - Train with ALL FP-producing authentic images.

Uses 677 authentic images that produce false positives at t=0.75, min-area=500
to significantly improve model training and reduce FP rate.

This includes:
- 173 dark images (mean brightness < 30)
- 504 normal images that still produce FPs

Fine-tunes v3 models to create v4 models.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from pathlib import Path

import time
import argparse

# ============================================================================
# CONFIGURATION
# ============================================================================

IMG_SIZE = 512
BATCH_SIZE = 4
EPOCHS = 25
LR = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
TRAIN_IMAGES_PATH = PROJECT_DIR / 'train_images'
TRAIN_MASKS_PATH = PROJECT_DIR / 'train_masks'
MODELS_DIR = PROJECT_DIR / 'models'

# Hard negative images - ALL FP-producing authentic images at t=0.75, min-area=500
# This list is generated from scanning all 2377 authentic images
HARD_NEGATIVE_FILE = Path('/tmp/all_fp_images.txt')

# Models to retrain (starting from v3)
MODELS_CONFIG = {
    'highres_no_ela': {
        'base_model': 'highres_no_ela_v3_best.pth',
        'output': 'highres_no_ela_v4_best.pth',
        'in_channels': 3,
    },
    'hard_negative': {
        'base_model': 'hard_negative_v3_best.pth', 
        'output': 'hard_negative_v4_best.pth',
        'in_channels': 3,
    },
    'high_recall': {
        'base_model': 'high_recall_v3_best.pth',
        'output': 'high_recall_v4_best.pth',
        'in_channels': 3,
    },
    'enhanced_aug': {
        'base_model': 'enhanced_aug_v3_best.pth',
        'output': 'enhanced_aug_v4_best.pth',
        'in_channels': 3,
    },
}

# ============================================================================
# MODEL ARCHITECTURE
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


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


class CombinedLossV4(nn.Module):
    """Combined Focal + Dice loss with STRONGER FP penalty for v4."""
    def __init__(self, focal_weight=0.5, dice_weight=0.5, fp_penalty=4.0):
        super().__init__()
        self.focal = FocalLoss(alpha=0.25, gamma=2.0)
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.fp_penalty = fp_penalty  # Increased from 2.0 to 4.0
    
    def forward(self, pred, target):
        # Focal loss
        focal_loss = self.focal(pred, target)
        
        # Dice loss
        pred_sigmoid = torch.sigmoid(pred)
        intersection = (pred_sigmoid * target).sum()
        dice = (2 * intersection + 1) / (pred_sigmoid.sum() + target.sum() + 1)
        dice_loss = 1 - dice
        
        # Extra FP penalty for authentic images (target is all zeros)
        is_authentic = (target.sum(dim=(1,2,3)) == 0).float()
        fp_loss = (pred_sigmoid * is_authentic.view(-1, 1, 1, 1)).mean() * self.fp_penalty
        
        return self.focal_weight * focal_loss + self.dice_weight * dice_loss + fp_loss


# ============================================================================
# DATASET
# ============================================================================

class HardNegativeDatasetV4(Dataset):
    """Dataset with forged images + ALL hard negative authentic images."""
    
    def __init__(self, forged_paths, mask_paths, hard_negative_paths, 
                 img_size=512, hard_neg_weight=5.0):
        self.img_size = img_size
        
        # Forged samples
        self.samples = []
        self.weights = []
        
        for img_path, mask_path in zip(forged_paths, mask_paths):
            self.samples.append({
                'image': img_path,
                'mask': mask_path,
                'is_forged': True
            })
            # Weight by forgery size
            mask = np.load(mask_path)
            if mask.ndim == 3:
                mask = mask[0]
            forgery_ratio = (mask > 0.5).sum() / mask.size * 100
            if forgery_ratio < 1.0:
                self.weights.append(3.0)
            elif forgery_ratio < 5.0:
                self.weights.append(2.0)
            else:
                self.weights.append(1.0)
        
        # Hard negative authentic samples (higher weight)
        for img_path in hard_negative_paths:
            self.samples.append({
                'image': img_path,
                'mask': None,  # All zeros
                'is_forged': False
            })
            self.weights.append(hard_neg_weight)  # High weight for hard negatives
        
        print(f"Dataset: {len(forged_paths)} forged + {len(hard_negative_paths)} hard negatives")
        
        # Augmentations
        self.transform = A.Compose([
            A.RandomResizedCrop(size=(img_size, img_size), scale=(0.7, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.OneOf([
                A.GaussNoise(std_range=(0.01, 0.05)),
                A.GaussianBlur(blur_limit=3),
                A.MedianBlur(blur_limit=3),
            ], p=0.3),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ], p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = cv2.imread(str(sample['image']))
        if image is None:
            raise ValueError(f"Could not load {sample['image']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.img_size, self.img_size))
        
        # Load or create mask
        if sample['is_forged'] and sample['mask'] is not None:
            mask = np.load(sample['mask'])
            if mask.ndim == 3:
                mask = mask[0]
            mask = cv2.resize(mask.astype(np.float32), (self.img_size, self.img_size))
            mask = (mask > 0.5).astype(np.float32)
        else:
            mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        
        # Apply augmentations
        transformed = self.transform(image=image, mask=mask)
        image_tensor = transformed['image']
        mask = transformed['mask']
        # ToTensorV2 converts mask to tensor, handle both cases
        if isinstance(mask, np.ndarray):
            mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
        else:
            mask_tensor = mask.unsqueeze(0).float()
        
        return image_tensor, mask_tensor


def load_hard_negatives():
    """Load list of hard negative image IDs from file."""
    if not HARD_NEGATIVE_FILE.exists():
        raise FileNotFoundError(f"Hard negative file not found: {HARD_NEGATIVE_FILE}")
    
    image_ids = []
    with open(HARD_NEGATIVE_FILE, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if parts:
                image_ids.append(parts[0])
    
    return image_ids


def load_model(model_path, in_channels=3):
    """Load model from checkpoint."""
    base_model = smp.FPN(
        encoder_name='timm-efficientnet-b2',
        encoder_weights=None,
        in_channels=in_channels,
        classes=1
    )
    model = AttentionFPN(base_model).to(DEVICE)
    
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model


def create_fresh_model(in_channels=3):
    """Create a new model with ImageNet pretrained encoder."""
    base_model = smp.FPN(
        encoder_name='timm-efficientnet-b2',
        encoder_weights='imagenet',
        in_channels=in_channels,
        classes=1
    )
    model = AttentionFPN(base_model).to(DEVICE)
    return model


def train_model(model_name, config, forged_paths, mask_paths, hard_neg_paths):
    """Train one model with hard negatives."""
    print(f"\n{'='*60}")
    print(f"Training {model_name} v4")
    print(f"{'='*60}")
    
    base_model_path = MODELS_DIR / config['base_model']
    if not base_model_path.exists():
        print(f"  Base model not found: {base_model_path}")
        print(f"  Training from scratch with ImageNet pretrained encoder...")
        model = create_fresh_model(config['in_channels'])
    else:
        # Load pre-trained v3 model
        model = load_model(base_model_path, config['in_channels'])
        print(f"  Loaded base model: {config['base_model']}")
    
    # Create dataset
    dataset = HardNegativeDatasetV4(
        forged_paths, mask_paths, hard_neg_paths,
        img_size=IMG_SIZE, hard_neg_weight=5.0
    )
    
    # Weighted sampler
    sampler = WeightedRandomSampler(
        dataset.weights,
        num_samples=len(dataset),
        replacement=True
    )
    
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4)
    
    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = CombinedLossV4(fp_penalty=4.0)
    
    best_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        
        for batch_idx, (images, masks) in enumerate(loader):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        scheduler.step()
        avg_loss = epoch_loss / len(loader)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch + 1
            
            # Save best model
            output_path = MODELS_DIR / config['output']
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, output_path)
            print(f"  Saved best model (epoch {epoch+1}, loss={avg_loss:.4f})")
    
    print(f"  Training complete. Best: epoch {best_epoch}, loss={best_loss:.4f}")
    return best_loss


def main():
    parser = argparse.ArgumentParser(description='Train v4 models with hard negatives')
    parser.add_argument('--models', nargs='+', default=list(MODELS_CONFIG.keys()),
                       help='Models to train (default: all)')
    args = parser.parse_args()
    
    print("="*60)
    print("Hard Negative Mining V4")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Models to train: {args.models}")
    
    # Load hard negative image IDs
    hard_neg_ids = load_hard_negatives()
    print(f"Hard negative images: {len(hard_neg_ids)}")
    
    # Find hard negative image paths
    hard_neg_paths = []
    auth_dir = TRAIN_IMAGES_PATH / 'authentic'
    for img_id in hard_neg_ids:
        for ext in ['.png', '.jpg', '.jpeg', '.tif']:
            path = auth_dir / f"{img_id}{ext}"
            if path.exists():
                hard_neg_paths.append(path)
                break
    print(f"Found {len(hard_neg_paths)} hard negative images")
    
    # Load forged images
    forged_dir = TRAIN_IMAGES_PATH / 'forged'
    forged_paths = sorted(list(forged_dir.glob('*.png')))
    mask_paths = [TRAIN_MASKS_PATH / f"{p.stem}.npy" for p in forged_paths]
    
    # Filter to existing masks
    valid = [(fp, mp) for fp, mp in zip(forged_paths, mask_paths) if mp.exists()]
    forged_paths, mask_paths = zip(*valid) if valid else ([], [])
    print(f"Forged images: {len(forged_paths)}")
    
    # Train each model
    start_time = time.time()
    
    for model_name in args.models:
        if model_name not in MODELS_CONFIG:
            print(f"Unknown model: {model_name}")
            continue
        
        config = MODELS_CONFIG[model_name]
        train_model(model_name, config, forged_paths, mask_paths, hard_neg_paths)
    
    elapsed = time.time() - start_time
    print(f"\nTotal training time: {elapsed/3600:.1f} hours")


if __name__ == '__main__':
    main()
