#!/usr/bin/env python3
"""
Training with Enhanced Augmentation for Scientific Image Forgery Detection
Includes copy-paste augmentation for small forgeries.
"""

import os
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

import random
import time

# Configuration
IMG_SIZE = 512
BATCH_SIZE = 4
EPOCHS = 30
LR = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATASET_PATH = Path(__file__).parent.parent


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


class CopyPasteAugmentation:
    """Copy-paste augmentation for forgery detection."""
    
    def __init__(self, forgery_patches, prob=0.5):
        """
        forgery_patches: list of (image_patch, mask_patch) tuples
        """
        self.forgery_patches = forgery_patches
        self.prob = prob
    
    def __call__(self, image, mask):
        if random.random() > self.prob or len(self.forgery_patches) == 0:
            return image, mask
        
        # Select random patch
        patch_img, patch_mask = random.choice(self.forgery_patches)
        
        # Random scale
        scale = random.uniform(0.5, 1.5)
        new_h = int(patch_img.shape[0] * scale)
        new_w = int(patch_img.shape[1] * scale)
        new_h = min(new_h, image.shape[0] - 10)
        new_w = min(new_w, image.shape[1] - 10)
        
        if new_h < 10 or new_w < 10:
            return image, mask
        
        patch_img = cv2.resize(patch_img, (new_w, new_h))
        patch_mask = cv2.resize(patch_mask, (new_w, new_h))
        
        # Random position
        max_y = image.shape[0] - new_h
        max_x = image.shape[1] - new_w
        if max_y <= 0 or max_x <= 0:
            return image, mask
        
        y = random.randint(0, max_y)
        x = random.randint(0, max_x)
        
        # Blend patch into image
        patch_mask_3ch = np.stack([patch_mask] * 3, axis=-1)
        image_copy = image.copy()
        mask_copy = mask.copy()
        
        # Apply patch where mask is positive
        blend_mask = (patch_mask > 0.5).astype(np.float32)
        blend_mask_3ch = np.stack([blend_mask] * 3, axis=-1)
        
        image_copy[y:y+new_h, x:x+new_w] = (
            image_copy[y:y+new_h, x:x+new_w] * (1 - blend_mask_3ch) +
            patch_img * blend_mask_3ch
        ).astype(np.uint8)
        
        # Update mask
        mask_copy[y:y+new_h, x:x+new_w] = np.maximum(
            mask_copy[y:y+new_h, x:x+new_w],
            patch_mask
        )
        
        return image_copy, mask_copy


class EnhancedForgeryDataset(Dataset):
    def __init__(self, image_paths, mask_paths, augment=True, copy_paste_patches=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.augment = augment
        
        # Copy-paste augmentation
        self.copy_paste = None
        if copy_paste_patches:
            self.copy_paste = CopyPasteAugmentation(copy_paste_patches, prob=0.3)
        
        # Strong augmentation pipeline
        if augment:
            self.transform = A.Compose([
                A.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE), scale=(0.7, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Affine(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
                
                # Color augmentations (stronger)
                A.OneOf([
                    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
                    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
                ], p=0.7),
                
                # Noise and blur
                A.OneOf([
                    A.GaussNoise(),  # Use defaults
                    A.GaussianBlur(blur_limit=(3, 7)),
                    A.MotionBlur(blur_limit=7),
                ], p=0.3),
                
                # Compression artifacts
                A.ImageCompression(quality_range=(70, 100), p=0.3),
                
                # Cutout/GridMask
                A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(16, 32), hole_width_range=(16, 32), p=0.2),
                
                # Normalize
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(IMG_SIZE, IMG_SIZE),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = cv2.imread(str(self.image_paths[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask = np.load(str(self.mask_paths[idx]))
        if mask.ndim == 3:
            mask = mask[0]
        
        # Resize to consistent size first
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        mask = cv2.resize(mask.astype(np.float32), (IMG_SIZE, IMG_SIZE))
        
        # Apply copy-paste augmentation
        if self.copy_paste and self.augment:
            image, mask = self.copy_paste(image, mask)
        
        # Apply transforms
        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']
        
        # Ensure mask is tensor
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        mask = mask.float().unsqueeze(0)
        
        return image, mask


def extract_forgery_patches(image_paths, mask_paths, min_size=20, max_patches=500):
    """Extract forgery patches for copy-paste augmentation."""
    patches = []
    
    for img_path, mask_path in zip(image_paths[:100], mask_paths[:100]):  # Sample from first 100
        image = cv2.imread(str(img_path))
        mask = np.load(str(mask_path))
        if mask.ndim == 3:
            mask = mask[0]
        
        # Find forgery regions
        binary_mask = (mask > 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w >= min_size and h >= min_size:
                # Extract patch with some padding
                pad = 5
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(image.shape[1], x + w + pad)
                y2 = min(image.shape[0], y + h + pad)
                
                patch_img = image[y1:y2, x1:x2].copy()
                patch_mask = mask[y1:y2, x1:x2].copy()
                
                patches.append((patch_img, patch_mask))
                
                if len(patches) >= max_patches:
                    break
        
        if len(patches) >= max_patches:
            break
    
    print(f"Extracted {len(patches)} forgery patches for copy-paste augmentation")
    return patches


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for images, masks in loader:  # Training
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def validate(model, loader, device):
    model.eval()
    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0
    total_loss = 0
    criterion = FocalLoss(gamma=2.0)
    
    with torch.no_grad():
        for images, masks in loader:  # Validating
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            
            preds = torch.sigmoid(outputs) > 0.5
            targets = masks > 0.5
            
            total_tp += ((preds == 1) & (targets == 1)).sum().item()
            total_fp += ((preds == 1) & (targets == 0)).sum().item()
            total_fn += ((preds == 0) & (targets == 1)).sum().item()
            total_tn += ((preds == 0) & (targets == 0)).sum().item()
    
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    num = total_tp * total_tn - total_fp * total_fn
    den = (total_tp + total_fp) * (total_tp + total_fn) * (total_tn + total_fp) * (total_tn + total_fn)
    mcc = num / (den ** 0.5) if den > 0 else 0
    
    return {
        'loss': total_loss / len(loader),
        'f1': f1,
        'mcc': mcc,
        'recall': recall,
        'precision': precision
    }


def main():
    print("=" * 80)
    print("ENHANCED AUGMENTATION TRAINING")
    print("=" * 80)
    
    # Load data - correct directory structure
    forged_dir = DATASET_PATH / 'train_images' / 'forged'
    mask_dir = DATASET_PATH / 'train_masks'
    
    image_paths = []
    mask_paths = []
    
    for f in sorted(forged_dir.glob('*.png')):
        mask_path = mask_dir / f"{f.stem}.npy"
        if mask_path.exists():
            image_paths.append(f)
            mask_paths.append(mask_path)
    
    print(f"Training images: {len(image_paths)}")
    
    # Split train/val
    val_size = int(len(image_paths) * 0.1)
    val_images = image_paths[:val_size]
    val_masks = mask_paths[:val_size]
    train_images = image_paths[val_size:]
    train_masks = mask_paths[val_size:]
    
    print(f"Train: {len(train_images)}, Val: {len(val_images)}")
    
    # Extract forgery patches for copy-paste
    print("Extracting forgery patches...")
    forgery_patches = extract_forgery_patches(train_images, train_masks)
    
    # Create datasets
    train_dataset = EnhancedForgeryDataset(
        train_images, train_masks, 
        augment=True, 
        copy_paste_patches=forgery_patches
    )
    val_dataset = EnhancedForgeryDataset(val_images, val_masks, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    # Create model
    base_model = smp.FPN(
        encoder_name='timm-efficientnet-b2',
        encoder_weights='imagenet',
        in_channels=3,
        classes=1
    )
    model = AttentionFPN(base_model).to(DEVICE)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    criterion = FocalLoss(gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    
    # Training loop
    best_mcc = 0
    save_path = DATASET_PATH / 'models' / 'enhanced_aug_best.pth'
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_metrics = validate(model, val_loader, DEVICE)
        scheduler.step()
        
        elapsed = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{EPOCHS} ({elapsed:.1f}s) - "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_metrics['loss']:.4f}, "
              f"Val F1: {val_metrics['f1']:.4f}, Val MCC: {val_metrics['mcc']:.4f}, "
              f"Val Recall: {val_metrics['recall']:.4f}")
        
        if val_metrics['mcc'] > best_mcc:
            best_mcc = val_metrics['mcc']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_mcc': best_mcc,
            }, save_path)
            print(f"  ✓ Saved best model (MCC: {best_mcc:.4f})")
    
    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best Val MCC: {best_mcc:.4f}")
    print(f"Model saved to: {save_path}")


if __name__ == '__main__':
    main()
