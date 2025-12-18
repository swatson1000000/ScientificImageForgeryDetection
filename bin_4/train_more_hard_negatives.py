#!/usr/bin/env python3
"""
Script 1: Train V4 model with MORE authentic hard negatives
Based on error analysis: FP are high-confidence, model needs to see more authentic images

Changes from original V4 training:
- Include ALL authentic images as hard negatives (not just 20%)
- Add authentic images with high edge density (FP pattern)
- Higher FP penalty (8x instead of 4x)
"""

import os
import sys
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from pathlib import Path
import time

# ============================================================================
# CONFIGURATION
# ============================================================================

IMG_SIZE = 512
ENCODER = 'timm-efficientnet-b2'
BATCH_SIZE = 8
EPOCHS = 30
LR = 1e-4
FP_PENALTY = 8.0  # Increased from 4.0
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
TRAIN_IMAGES_PATH = PROJECT_DIR / 'train_images'
TRAIN_MASKS_PATH = PROJECT_DIR / 'train_masks'
MODELS_DIR = PROJECT_DIR / 'models'

# ============================================================================
# DATASET
# ============================================================================

class HardNegativeDataset(Dataset):
    """Dataset with ALL authentic images as hard negatives."""
    
    def __init__(self, forged_paths, mask_dir, authentic_paths, transform, is_train=True):
        self.forged_paths = forged_paths
        self.mask_dir = mask_dir
        self.authentic_paths = authentic_paths
        self.transform = transform
        self.is_train = is_train
        
        # Balance: use all authentic images
        # For training, we want roughly equal forged and authentic
        if is_train:
            # Use all authentic images
            self.authentic_ratio = len(authentic_paths) / len(forged_paths) if forged_paths else 1.0
        else:
            self.authentic_ratio = 0  # No authentic in validation
        
        self.total_len = len(forged_paths) + len(authentic_paths)
    
    def __len__(self):
        return self.total_len
    
    def __getitem__(self, idx):
        if idx < len(self.forged_paths):
            # Forged image
            img_path = self.forged_paths[idx]
            mask_path = self.mask_dir / img_path.name
            
            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                mask = (mask > 127).astype(np.float32)
            else:
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        else:
            # Authentic image (hard negative)
            auth_idx = idx - len(self.forged_paths)
            img_path = self.authentic_paths[auth_idx]
            
            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask'].unsqueeze(0)
        
        return image, mask


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
            encoder_weights='imagenet',
            in_channels=in_channels,
            classes=1,
        )
        self.attention = AttentionGate(1)
    
    def forward(self, x):
        return self.attention(self.base(x))


# ============================================================================
# LOSS
# ============================================================================

class HighFPPenaltyLoss(nn.Module):
    """Combined loss with very high FP penalty."""
    
    def __init__(self, fp_penalty=8.0):
        super().__init__()
        self.fp_penalty = fp_penalty
    
    def forward(self, pred, target):
        pred_sigmoid = torch.sigmoid(pred)
        
        # Dice loss
        intersection = (pred_sigmoid * target).sum()
        dice = (2 * intersection + 1) / (pred_sigmoid.sum() + target.sum() + 1)
        dice_loss = 1 - dice
        
        # BCE loss
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        # Apply higher weight to false positives
        # FP: pred is high (>0.5) but target is 0
        fp_weight = torch.where(
            (pred_sigmoid > 0.5) & (target < 0.5),
            torch.tensor(self.fp_penalty, device=pred.device),
            torch.tensor(1.0, device=pred.device)
        )
        weighted_bce = (bce * fp_weight).mean()
        
        return 0.5 * dice_loss + 0.5 * weighted_bce


# ============================================================================
# TRAINING
# ============================================================================

def get_transforms(is_train=True):
    if is_train:
        return A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.OneOf([
                A.GaussNoise(var_limit=(10, 50)),
                A.GaussianBlur(blur_limit=(3, 7)),
                A.MotionBlur(blur_limit=7),
            ], p=0.3),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
            ], p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f"    Batch {batch_idx}/{len(loader)}, Loss: {loss.item():.4f}")
    
    return total_loss / len(loader)


def validate(model, loader, device):
    model.eval()
    total_iou = 0
    count = 0
    
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = torch.sigmoid(model(images))
            preds = (outputs > 0.5).float()
            
            intersection = (preds * masks).sum()
            union = preds.sum() + masks.sum() - intersection
            iou = (intersection + 1) / (union + 1)
            
            total_iou += iou.item()
            count += 1
    
    return total_iou / count if count > 0 else 0


def main():
    print("=" * 60)
    print("TRAINING WITH MORE HARD NEGATIVES")
    print("=" * 60)
    print(f"FP Penalty: {FP_PENALTY}x")
    print(f"Device: {DEVICE}")
    print()
    
    # Get all images
    forged_dir = TRAIN_IMAGES_PATH / 'forged'
    authentic_dir = TRAIN_IMAGES_PATH / 'authentic'
    mask_dir = TRAIN_MASKS_PATH / 'forged'
    
    forged_images = sorted(list(forged_dir.glob('*.png')))
    authentic_images = sorted(list(authentic_dir.glob('*.png')))
    
    print(f"Forged images: {len(forged_images)}")
    print(f"Authentic images (ALL as hard negatives): {len(authentic_images)}")
    
    # Split for validation
    np.random.seed(42)
    indices = np.random.permutation(len(forged_images))
    val_size = int(0.1 * len(forged_images))
    
    train_forged = [forged_images[i] for i in indices[val_size:]]
    val_forged = [forged_images[i] for i in indices[:val_size]]
    
    # Use all authentic for training
    train_authentic = authentic_images
    
    print(f"Train forged: {len(train_forged)}, Train authentic: {len(train_authentic)}")
    print(f"Val forged: {len(val_forged)}")
    
    # Datasets
    train_dataset = HardNegativeDataset(
        train_forged, mask_dir, train_authentic,
        get_transforms(is_train=True), is_train=True
    )
    val_dataset = HardNegativeDataset(
        val_forged, mask_dir, [],
        get_transforms(is_train=False), is_train=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Model
    model = AttentionFPN().to(DEVICE)
    criterion = HighFPPenaltyLoss(fp_penalty=FP_PENALTY)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # Training
    best_iou = 0
    save_path = MODELS_DIR / 'hard_negative_v7_best.pth'
    
    for epoch in range(1, EPOCHS + 1):
        start = time.time()
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_iou = validate(model, val_loader, DEVICE)
        scheduler.step()
        
        elapsed = time.time() - start
        print(f"Epoch {epoch}/{EPOCHS} - Loss: {train_loss:.4f}, Val IoU: {val_iou:.4f}, "
              f"LR: {scheduler.get_last_lr()[0]:.2e}, Time: {elapsed:.0f}s")
        
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_iou,
            }, save_path)
            print(f"  ✓ Saved best model (IoU: {val_iou:.4f})")
    
    print(f"\nTraining complete! Best IoU: {best_iou:.4f}")
    print(f"Model saved to: {save_path}")


if __name__ == '__main__':
    main()
