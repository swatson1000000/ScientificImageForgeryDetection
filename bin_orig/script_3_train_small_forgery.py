#!/usr/bin/env python3
"""
Small Forgery Specialist - Train model focused on small/missed forgeries.

Strategy:
1. Use higher resolution (768x768) for better small object detection
2. Train on images that current model misses
3. Focus on small forgery regions with weighted sampling
4. Use attention mechanisms for better localization
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

# ============================================================================
# CONFIGURATION
# ============================================================================

IMG_SIZE = 768  # Higher resolution for small forgeries
BATCH_SIZE = 2  # Smaller batch due to higher res
EPOCHS = 30
LR = 5e-5  # Lower LR for fine-tuning
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
TRAIN_IMAGES_PATH = PROJECT_DIR / 'train_images'
TRAIN_MASKS_PATH = PROJECT_DIR / 'train_masks'
MODELS_DIR = PROJECT_DIR / 'models'

# Will be populated with missed forgeries
MISSED_FORGERIES_FILE = Path('/tmp/missed_forgeries.txt')

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class AttentionGate(nn.Module):
    """Attention gate for better small object focus."""
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


class SmallObjectLoss(nn.Module):
    """Loss function weighted towards small objects."""
    def __init__(self, focal_weight=0.4, dice_weight=0.4, boundary_weight=0.2):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
    
    def focal_loss(self, pred, target, alpha=0.25, gamma=2.0):
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce)
        focal = alpha * (1 - pt) ** gamma * bce
        return focal.mean()
    
    def dice_loss(self, pred, target):
        pred_sigmoid = torch.sigmoid(pred)
        intersection = (pred_sigmoid * target).sum()
        dice = (2 * intersection + 1) / (pred_sigmoid.sum() + target.sum() + 1)
        return 1 - dice
    
    def boundary_loss(self, pred, target):
        """Extra penalty for boundary errors (important for small objects)."""
        pred_sigmoid = torch.sigmoid(pred)
        
        # Simple Sobel-like edge detection
        kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], 
                              dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)
        
        # Compute edges
        if target.dim() == 4:
            target_edges = F.conv2d(target, kernel, padding=1)
        else:
            target_edges = F.conv2d(target.unsqueeze(1), kernel, padding=1)
        
        pred_edges = F.conv2d(pred_sigmoid, kernel, padding=1)
        
        # Weight errors at boundaries more heavily
        boundary_weight = (target_edges.abs() > 0.1).float() * 5 + 1
        boundary_error = ((pred_sigmoid - target) ** 2 * boundary_weight).mean()
        
        return boundary_error
    
    def forward(self, pred, target):
        focal = self.focal_loss(pred, target)
        dice = self.dice_loss(pred, target)
        boundary = self.boundary_loss(pred, target)
        
        return self.focal_weight * focal + self.dice_weight * dice + self.boundary_weight * boundary


# ============================================================================
# DATASET
# ============================================================================

class SmallForgeryDataset(Dataset):
    """Dataset focused on small forgeries."""
    
    def __init__(self, image_dir, mask_dir, missed_images=None, include_all=True, augment=True):
        """
        Args:
            image_dir: Path to forged images
            mask_dir: Path to ground truth masks
            missed_images: List of image names that current model misses
            include_all: If True, include all images but weight missed ones higher
            augment: Apply data augmentation
        """
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.augment = augment
        
        # Get all forged images
        all_images = sorted(self.image_dir.glob('*.png'))
        
        if missed_images:
            missed_set = set(missed_images)
            if include_all:
                # Include all but track which are missed
                self.images = all_images
                self.is_missed = [img.name in missed_set for img in all_images]
            else:
                # Only include missed images
                self.images = [img for img in all_images if img.name in missed_set]
                self.is_missed = [True] * len(self.images)
        else:
            self.images = all_images
            self.is_missed = [False] * len(self.images)
        
        # Calculate sample weights (missed images weighted 3x higher)
        self.weights = [3.0 if missed else 1.0 for missed in self.is_missed]
        
        # Augmentation pipeline
        if augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.5),
                A.OneOf([
                    A.GaussNoise(var_limit=(10, 50), p=1),
                    A.GaussianBlur(blur_limit=3, p=1),
                    A.MedianBlur(blur_limit=3, p=1),
                ], p=0.3),
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1),
                ], p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        # Masks are .npy files with same stem as image
        mask_path = self.mask_dir / f"{img_path.stem}.npy"
        
        # Load image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        
        # Load mask (.npy format)
        if mask_path.exists():
            mask = np.load(str(mask_path))
            if mask.ndim == 3:
                mask = mask[0]
            mask = cv2.resize(mask.astype(np.float32), (IMG_SIZE, IMG_SIZE))
            mask = (mask > 0.5).astype(np.float32)
        else:
            mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
        
        # Apply augmentation
        augmented = self.transform(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask'].unsqueeze(0)
        
        return image, mask, self.is_missed[idx]


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for images, masks, _ in dataloader:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate(model, dataloader, device, threshold=0.3):
    model.eval()
    total_iou = 0
    total_dice = 0
    count = 0
    
    with torch.no_grad():
        for images, masks, _ in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = torch.sigmoid(model(images))
            preds = (outputs > threshold).float()
            
            # Calculate IoU and Dice
            for i in range(preds.shape[0]):
                pred = preds[i].cpu().numpy().flatten()
                target = masks[i].cpu().numpy().flatten()
                
                intersection = (pred * target).sum()
                union = pred.sum() + target.sum() - intersection
                
                if union > 0:
                    iou = intersection / union
                    dice = 2 * intersection / (pred.sum() + target.sum() + 1e-8)
                    total_iou += iou
                    total_dice += dice
                    count += 1
    
    if count == 0:
        return 0, 0
    
    return total_iou / count, total_dice / count


def main():
    print("=" * 60)
    print("SMALL FORGERY SPECIALIST TRAINING")
    print("=" * 60)
    print(f"Resolution: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Device: {DEVICE}")
    print()
    
    # Load missed forgeries list
    missed_images = []
    if MISSED_FORGERIES_FILE.exists():
        with open(MISSED_FORGERIES_FILE) as f:
            missed_images = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(missed_images)} missed forgery images")
    else:
        print("No missed forgeries file found - training on all images")
    
    # Create dataset with weighted sampling
    train_dataset = SmallForgeryDataset(
        TRAIN_IMAGES_PATH / 'forged',
        TRAIN_MASKS_PATH,
        missed_images=missed_images,
        include_all=True,  # Include all but weight missed ones higher
        augment=True
    )
    
    # Use weighted sampler to oversample missed images
    sampler = WeightedRandomSampler(
        weights=train_dataset.weights,
        num_samples=len(train_dataset),
        replacement=True
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )
    
    # Create validation dataset (no augmentation, no sampling)
    val_dataset = SmallForgeryDataset(
        TRAIN_IMAGES_PATH / 'forged',
        TRAIN_MASKS_PATH,
        missed_images=missed_images,
        include_all=False,  # Only validate on missed images
        augment=False
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples (missed only): {len(val_dataset)}")
    print()
    
    # Create model (start from best v4 model)
    base_model_path = MODELS_DIR / 'highres_no_ela_v4_best.pth'
    
    base_model = smp.FPN(
        encoder_name='timm-efficientnet-b2',
        encoder_weights=None,
        in_channels=3,
        classes=1
    )
    model = AttentionFPN(base_model).to(DEVICE)
    
    if base_model_path.exists():
        print(f"Loading base model: {base_model_path.name}")
        checkpoint = torch.load(base_model_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("No base model found, training from scratch with ImageNet weights")
        base_model = smp.FPN(
            encoder_name='timm-efficientnet-b2',
            encoder_weights='imagenet',
            in_channels=3,
            classes=1
        )
        model = AttentionFPN(base_model).to(DEVICE)
    
    # Loss and optimizer
    criterion = SmallObjectLoss(focal_weight=0.4, dice_weight=0.4, boundary_weight=0.2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    
    # Training loop
    best_iou = 0
    output_path = MODELS_DIR / 'small_forgery_v5_best.pth'
    
    print("Starting training...")
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_iou, val_dice = validate(model, val_loader, DEVICE)
        scheduler.step()
        
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {train_loss:.4f}, Val IoU: {val_iou:.4f}, Val Dice: {val_dice:.4f}, LR: {lr:.2e}")
        
        # Save best model
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_iou,
                'val_dice': val_dice,
            }, output_path)
            print(f"  ✓ Saved new best model (IoU: {val_iou:.4f})")
    
    elapsed = time.time() - start_time
    print()
    print("=" * 60)
    print(f"Training complete in {elapsed/60:.1f} minutes")
    print(f"Best validation IoU: {best_iou:.4f}")
    print(f"Model saved to: {output_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
