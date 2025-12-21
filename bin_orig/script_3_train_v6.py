#!/usr/bin/env python3
"""
V6 Training - Three Major Improvements:
1. Larger encoder: EfficientNet-B4 (instead of B2)
2. Higher resolution: 768px (instead of 512px)
3. Different architecture: UNet++ (instead of FPN)

Expected training time: ~12-24 hours per model
"""

import os
import sys
sys.stdout.reconfigure(line_buffering=True)
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

# ============================================================================
# CONFIGURATION - V6 IMPROVEMENTS
# ============================================================================

IMG_SIZE = 768          # IMPROVEMENT 2: Higher resolution (was 512)
ENCODER = 'timm-efficientnet-b4'  # IMPROVEMENT 1: Larger encoder (was b2)
BATCH_SIZE = 2          # Reduced due to larger model + resolution
EPOCHS = 40             # More epochs for larger model
LR = 5e-5               # Lower LR for larger model
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
TRAIN_IMAGES_PATH = PROJECT_DIR / 'train_images'
TRAIN_MASKS_PATH = PROJECT_DIR / 'train_masks'
MODELS_DIR = PROJECT_DIR / 'models'

# ============================================================================
# MODEL ARCHITECTURE - V6 with UNet++
# ============================================================================

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


class AttentionUNetPP(nn.Module):
    """UNet++ with attention gate - IMPROVEMENT 3."""
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        self.attention = AttentionGate(1)
    
    def forward(self, x):
        return self.attention(self.base(x))


def create_model_v6(in_channels=3):
    """Create UNet++ with EfficientNet-B4 encoder at 768px."""
    # IMPROVEMENT 3: UNet++ architecture (better feature aggregation)
    base = smp.UnetPlusPlus(
        encoder_name=ENCODER,          # IMPROVEMENT 1: EfficientNet-B4
        encoder_weights='imagenet',
        in_channels=in_channels,
        classes=1,
        decoder_attention_type='scse',  # Squeeze-and-Excitation attention
    )
    return AttentionUNetPP(base)


# ============================================================================
# LOSS FUNCTION
# ============================================================================

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


class CombinedLossV6(nn.Module):
    """Combined Focal + Dice + Boundary loss for V6."""
    def __init__(self, focal_weight=0.4, dice_weight=0.4, boundary_weight=0.2, fp_penalty=4.0):
        super().__init__()
        self.focal = FocalLoss(alpha=0.25, gamma=2.0)
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.fp_penalty = fp_penalty
    
    def forward(self, pred, target):
        # Focal loss
        focal_loss = self.focal(pred, target)
        
        # Dice loss
        pred_sigmoid = torch.sigmoid(pred)
        intersection = (pred_sigmoid * target).sum()
        dice = (2 * intersection + 1) / (pred_sigmoid.sum() + target.sum() + 1)
        dice_loss = 1 - dice
        
        # Boundary loss (encourage sharp edges)
        # Simple gradient-based boundary detection
        if target.sum() > 0:
            pred_grad_x = torch.abs(pred_sigmoid[:, :, :, 1:] - pred_sigmoid[:, :, :, :-1])
            pred_grad_y = torch.abs(pred_sigmoid[:, :, 1:, :] - pred_sigmoid[:, :, :-1, :])
            target_grad_x = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
            target_grad_y = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
            boundary_loss = F.mse_loss(pred_grad_x, target_grad_x) + F.mse_loss(pred_grad_y, target_grad_y)
        else:
            boundary_loss = torch.tensor(0.0, device=pred.device)
        
        # FP penalty for authentic images
        is_authentic = (target.sum(dim=(1,2,3)) == 0).float()
        fp_loss = (pred_sigmoid * is_authentic.view(-1, 1, 1, 1)).mean() * self.fp_penalty
        
        total = (self.focal_weight * focal_loss + 
                 self.dice_weight * dice_loss + 
                 self.boundary_weight * boundary_loss + 
                 fp_loss)
        return total


# ============================================================================
# DATASET
# ============================================================================

class ForgeryDatasetV6(Dataset):
    """Dataset for V6 training at 768px resolution."""
    
    def __init__(self, forged_paths, mask_paths, authentic_paths=None, img_size=768):
        self.img_size = img_size
        self.samples = []
        self.weights = []
        
        # Forged samples
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
                self.weights.append(3.0)  # Small forgeries get more weight
            elif forgery_ratio < 5.0:
                self.weights.append(2.0)
            else:
                self.weights.append(1.0)
        
        # Authentic samples (hard negatives)
        if authentic_paths:
            for img_path in authentic_paths:
                self.samples.append({
                    'image': img_path,
                    'mask': None,
                    'is_forged': False
                })
                self.weights.append(5.0)  # High weight for hard negatives
        
        print(f"Dataset: {len(forged_paths)} forged + {len(authentic_paths) if authentic_paths else 0} authentic")
        
        # Strong augmentations for 768px
        self.transform = A.Compose([
            A.RandomResizedCrop(size=(img_size, img_size), scale=(0.6, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.OneOf([
                A.GaussNoise(std_range=(0.01, 0.05)),
                A.GaussianBlur(blur_limit=3),
                A.MotionBlur(blur_limit=3),
            ], p=0.3),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            ], p=0.5),
            A.OneOf([
                A.ElasticTransform(alpha=30, sigma=5),
                A.GridDistortion(num_steps=5, distort_limit=0.1),
                A.OpticalDistortion(distort_limit=0.1),
            ], p=0.2),
            A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(20, 60), 
                           hole_width_range=(20, 60), fill="random", p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        self.val_transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = cv2.imread(str(sample['image']))
        if image is None:
            # Return dummy on error
            return torch.zeros(3, self.img_size, self.img_size), torch.zeros(1, self.img_size, self.img_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load or create mask
        if sample['mask'] is not None:
            mask = np.load(sample['mask'])
            if mask.ndim == 3:
                mask = mask[0]
            mask = (mask > 0.5).astype(np.float32)
        else:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        
        # Apply augmentation
        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask'].unsqueeze(0)
        
        return image, mask


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0
    
    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f"    Batch {batch_idx}/{len(loader)}, Loss: {loss.item():.4f}")
    
    return total_loss / len(loader)


def validate(model, loader, device, threshold=0.35):
    model.eval()
    total_iou = 0
    count = 0
    
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = torch.sigmoid(model(images))
            preds = (outputs > threshold).float()
            
            # IoU
            intersection = (preds * masks).sum()
            union = preds.sum() + masks.sum() - intersection
            if union > 0:
                iou = intersection / union
                total_iou += iou.item()
                count += 1
    
    return total_iou / max(count, 1)


def main():
    print("=" * 60)
    print("V6 TRAINING - THREE MAJOR IMPROVEMENTS")
    print("=" * 60)
    print(f"1. Encoder: {ENCODER} (larger)")
    print(f"2. Resolution: {IMG_SIZE}px (higher)")
    print(f"3. Architecture: UNet++ (different)")
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print()
    
    # Collect forged images and masks
    forged_dir = TRAIN_IMAGES_PATH / 'forged'
    forged_images = sorted(list(forged_dir.glob('*.png')))
    
    forged_paths = []
    mask_paths = []
    for img_path in forged_images:
        mask_path = TRAIN_MASKS_PATH / f"{img_path.stem}.npy"
        if mask_path.exists():
            forged_paths.append(img_path)
            mask_paths.append(mask_path)
    
    print(f"Found {len(forged_paths)} forged images with masks")
    
    # Collect authentic images for hard negative mining
    authentic_dir = TRAIN_IMAGES_PATH / 'authentic'
    authentic_images = sorted(list(authentic_dir.glob('*.png')))
    
    # Use a subset of authentic images as hard negatives (every 5th image)
    hard_negatives = authentic_images[::5]  # ~475 images
    print(f"Using {len(hard_negatives)} authentic images as hard negatives")
    
    # Split for validation
    val_split = int(len(forged_paths) * 0.1)
    train_forged = forged_paths[val_split:]
    train_masks = mask_paths[val_split:]
    val_forged = forged_paths[:val_split]
    val_masks = mask_paths[:val_split]
    
    print(f"Train: {len(train_forged)} forged, Val: {len(val_forged)} forged")
    
    # Create datasets
    train_dataset = ForgeryDatasetV6(train_forged, train_masks, hard_negatives, img_size=IMG_SIZE)
    val_dataset = ForgeryDatasetV6(val_forged, val_masks, img_size=IMG_SIZE)
    
    # Weighted sampler
    sampler = WeightedRandomSampler(
        weights=train_dataset.weights,
        num_samples=len(train_dataset),
        replacement=True
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, 
                             num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=4, pin_memory=True)
    
    # Create model
    print("\nCreating V6 model...")
    model = create_model_v6(in_channels=3)
    model = model.to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # Loss
    criterion = CombinedLossV6(focal_weight=0.4, dice_weight=0.4, boundary_weight=0.2, fp_penalty=4.0)
    
    # Mixed precision
    scaler = torch.amp.GradScaler('cuda')
    
    # Training loop
    best_iou = 0
    MODELS_DIR.mkdir(exist_ok=True)
    
    print("\nStarting training...")
    start_time = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE, scaler)
        val_iou = validate(model, val_loader, DEVICE)
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch}/{EPOCHS} - Loss: {train_loss:.4f}, Val IoU: {val_iou:.4f}, "
              f"LR: {scheduler.get_last_lr()[0]:.2e}, Time: {epoch_time:.0f}s")
        
        # Save best model
        if val_iou > best_iou:
            best_iou = val_iou
            save_path = MODELS_DIR / 'unetpp_b4_768_v6_best.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_iou,
            }, save_path)
            print(f"  ✓ Saved best model (IoU: {val_iou:.4f})")
    
    total_time = time.time() - start_time
    print()
    print("=" * 60)
    print(f"Training complete!")
    print(f"Best validation IoU: {best_iou:.4f}")
    print(f"Total time: {total_time/3600:.1f} hours")
    print(f"Model saved to: {MODELS_DIR / 'unetpp_b4_768_v6_best.pth'}")
    print("=" * 60)


if __name__ == '__main__':
    main()
