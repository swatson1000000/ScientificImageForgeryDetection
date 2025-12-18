#!/usr/bin/env python3
"""
Hard Negative Mining V5 - Reduce False Positives

Train model with the 198 current FP images as hard negatives.
These are authentic images that V4 incorrectly flags as forged.

Strategy:
- Fine-tune best V4 model (highres_no_ela_v4)
- Add 198 FP images with all-zero masks (weight 10x)
- Keep all forged images for balance
- Stronger FP penalty in loss
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

# ============================================================================
# CONFIGURATION
# ============================================================================

IMG_SIZE = 512
BATCH_SIZE = 4
EPOCHS = 20
LR = 2e-5  # Lower LR for fine-tuning
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
TRAIN_IMAGES_PATH = PROJECT_DIR / 'train_images'
TRAIN_MASKS_PATH = PROJECT_DIR / 'train_masks'
MODELS_DIR = PROJECT_DIR / 'models'

# Base model and output
BASE_MODEL = MODELS_DIR / 'highres_no_ela_v4_best.pth'
OUTPUT_MODEL = MODELS_DIR / 'highres_no_ela_v5_best.pth'

# FP images file
FP_IMAGES_FILE = SCRIPT_DIR / 'fp_images_v5.txt'

# ============================================================================
# MODEL ARCHITECTURE (same as V4)
# ============================================================================

class AttentionGate(nn.Module):
    def __init__(self, in_channels, hidden1=8, hidden2=4):
        super().__init__()
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


# ============================================================================
# LOSS WITH STRONG FP PENALTY
# ============================================================================

class HardNegativeLoss(nn.Module):
    """Loss with very strong penalty for false positives on authentic images."""
    def __init__(self, focal_weight=0.4, dice_weight=0.3, fp_penalty=10.0):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.fp_penalty = fp_penalty
    
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
    
    def forward(self, pred, target):
        focal = self.focal_loss(pred, target)
        dice = self.dice_loss(pred, target)
        
        # Strong FP penalty for authentic images (all-zero masks)
        is_authentic = (target.sum(dim=(1, 2, 3)) == 0).float()
        pred_sigmoid = torch.sigmoid(pred)
        # Penalize any positive prediction on authentic images
        fp_loss = (pred_sigmoid * is_authentic.view(-1, 1, 1, 1)).mean() * self.fp_penalty
        
        return self.focal_weight * focal + self.dice_weight * dice + fp_loss


# ============================================================================
# DATASET
# ============================================================================

class HardNegativeDatasetV5(Dataset):
    """Dataset with forged images + hard negative FP images."""
    
    def __init__(self, forged_paths, mask_paths, fp_paths, img_size=512, fp_weight=10.0):
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
            self.weights.append(1.0)
        
        # Hard negative FP samples (high weight)
        for img_path in fp_paths:
            self.samples.append({
                'image': img_path,
                'mask': None,
                'is_forged': False
            })
            self.weights.append(fp_weight)
        
        print(f"Dataset: {len(forged_paths)} forged + {len(fp_paths)} hard negatives (FPs)")
        
        # Augmentations
        self.transform = A.Compose([
            A.RandomResizedCrop(size=(img_size, img_size), scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.OneOf([
                A.GaussNoise(std_range=(0.01, 0.05)),
                A.GaussianBlur(blur_limit=3),
            ], p=0.3),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20),
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
        
        if isinstance(mask, np.ndarray):
            mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
        else:
            mask_tensor = mask.unsqueeze(0).float()
        
        return image_tensor, mask_tensor


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate_fp_rate(model, fp_loader, device, threshold=0.35):
    """Validate FP rate on hard negative images."""
    model.eval()
    fp_count = 0
    total = 0
    
    with torch.no_grad():
        for images, masks in fp_loader:
            images = images.to(device)
            
            outputs = torch.sigmoid(model(images))
            preds = (outputs > threshold).float()
            
            for i in range(preds.shape[0]):
                pred = preds[i].cpu().numpy()
                if pred.sum() > 300:  # min_area threshold
                    fp_count += 1
                total += 1
    
    return fp_count, total


def create_model(in_channels=3):
    """Create AttentionFPN model."""
    base = smp.FPN(
        encoder_name='timm-efficientnet-b2',
        encoder_weights='imagenet',
        in_channels=in_channels,
        classes=1
    )
    return AttentionFPN(base)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("HARD NEGATIVE MINING V5 - FP REDUCTION")
    print("=" * 60)
    print(f"Resolution: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Device: {DEVICE}")
    print()
    
    # Load FP images
    if not FP_IMAGES_FILE.exists():
        raise FileNotFoundError(f"FP images file not found: {FP_IMAGES_FILE}")
    
    with open(FP_IMAGES_FILE) as f:
        fp_names = [l.strip() for l in f.readlines()]
    
    print(f"Loaded {len(fp_names)} FP image names")
    
    # Convert to paths
    auth_dir = TRAIN_IMAGES_PATH / 'authentic'
    fp_paths = []
    for name in fp_names:
        # Try with and without extension
        for ext in ['.png', '.jpg', '']:
            path = auth_dir / f"{name}{ext}"
            if path.exists():
                fp_paths.append(path)
                break
    
    print(f"Found {len(fp_paths)} FP image paths")
    
    # Load forged images
    forged_dir = TRAIN_IMAGES_PATH / 'forged'
    forged_paths = sorted(list(forged_dir.glob('*.png')))
    mask_paths = [TRAIN_MASKS_PATH / f"{p.stem}.npy" for p in forged_paths]
    
    # Filter to existing masks
    valid = [(fp, mp) for fp, mp in zip(forged_paths, mask_paths) if mp.exists()]
    forged_paths, mask_paths = zip(*valid) if valid else ([], [])
    print(f"Forged images: {len(forged_paths)}")
    
    # Create dataset
    dataset = HardNegativeDatasetV5(
        forged_paths, mask_paths, fp_paths,
        img_size=IMG_SIZE, fp_weight=10.0
    )
    
    # Create separate FP validation set
    class FPOnlyDataset(Dataset):
        def __init__(self, paths, img_size):
            self.paths = paths
            self.img_size = img_size
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
        
        def __len__(self):
            return len(self.paths)
        
        def __getitem__(self, idx):
            image = cv2.imread(str(self.paths[idx]))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            transformed = self.transform(image=image)
            return transformed['image'], torch.zeros(1, self.img_size, self.img_size)
    
    fp_val_dataset = FPOnlyDataset(fp_paths, IMG_SIZE)
    fp_val_loader = DataLoader(fp_val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Weighted sampler
    sampler = WeightedRandomSampler(
        weights=dataset.weights,
        num_samples=len(dataset),
        replacement=True
    )
    
    train_loader = DataLoader(
        dataset, batch_size=BATCH_SIZE,
        sampler=sampler, num_workers=4, pin_memory=True
    )
    
    # Create model and load base weights
    model = create_model(in_channels=3)
    
    if BASE_MODEL.exists():
        print(f"Loading base model: {BASE_MODEL.name}")
        checkpoint = torch.load(BASE_MODEL, map_location='cpu', weights_only=True)
        # Handle both formats: direct state_dict or wrapped checkpoint
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)
    else:
        print("Warning: Base model not found, training from scratch")
    
    model = model.to(DEVICE)
    
    # Check initial FP rate
    initial_fp, total_fp = validate_fp_rate(model, fp_val_loader, DEVICE)
    print(f"Initial FP rate on hard negatives: {initial_fp}/{total_fp} ({initial_fp/total_fp*100:.1f}%)")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR/100)
    
    # Loss
    criterion = HardNegativeLoss(focal_weight=0.4, dice_weight=0.3, fp_penalty=10.0)
    
    # Training loop
    best_fp_rate = initial_fp / total_fp
    print("\nStarting training...")
    
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        fp_count, total = validate_fp_rate(model, fp_val_loader, DEVICE)
        fp_rate = fp_count / total
        scheduler.step()
        
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}/{EPOCHS} - Loss: {train_loss:.4f}, FP: {fp_count}/{total} ({fp_rate*100:.1f}%), LR: {lr:.2e}")
        
        if fp_rate < best_fp_rate:
            best_fp_rate = fp_rate
            torch.save(model.state_dict(), OUTPUT_MODEL)
            print(f"  ✓ Saved new best model (FP rate: {fp_rate*100:.1f}%)")
    
    print()
    print("=" * 60)
    print(f"Training complete!")
    print(f"Best FP rate on hard negatives: {best_fp_rate*100:.1f}%")
    print(f"Model saved to: {OUTPUT_MODEL}")
    print("=" * 60)


if __name__ == '__main__':
    main()
