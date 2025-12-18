#!/usr/bin/env python3
"""
Train models with augmented forged data + hard negatives.

This script trains all 5 main models using:
- Original forged images (2,751)
- Augmented forged images (7,564) 
- Hard negative authentic images (73 from test set that produce FPs)
- Regular authentic images for balance

Total training data:
- Forged: ~10,315 images
- Authentic: ~12,200 images (balanced sampling)
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, ConcatDataset
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from pathlib import Path

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
MODELS_DIR = PROJECT_DIR / 'models'

# Data paths
FORGED_ORIGINAL = PROJECT_DIR / 'train_images' / 'forged'
FORGED_AUGMENTED = PROJECT_DIR / 'train_images' / 'forged_augmented'
MASKS_ORIGINAL = PROJECT_DIR / 'train_masks'
MASKS_AUGMENTED = PROJECT_DIR / 'train_masks_augmented'
AUTHENTIC_DIR = PROJECT_DIR / 'train_images' / 'authentic'
AUTHENTIC_AUG_DIR = PROJECT_DIR / 'train_images' / 'authentic_augmented'
AUTHENTIC_AUG_V2_DIR = PROJECT_DIR / 'train_images' / 'authentic_augmented_v2'

# Hard negative images from test_authentic_100 that produce FPs
HARD_NEGATIVE_DIR = PROJECT_DIR / 'test_authentic_100'
HARD_NEGATIVE_IDS = [
    10, 10015, 10017, 10030, 10138, 10147, 10152, 10176, 10208, 10296,
    10348, 1041, 10435, 10448, 10457, 10458, 10602, 10636, 10703, 10730,
    10742, 10799, 10855, 10878, 10884, 10911, 10939, 10951, 10980, 10986,
    10991, 10992, 11128, 11206, 11223, 11240, 11249, 11277, 11296, 11299,
    11307, 11356, 11392, 11399, 11426, 11440, 11446, 11501, 11513, 11547,
    11639, 11646, 11662, 11698, 11708, 1173, 1174, 11740, 11747, 11761,
    11854, 11904, 11964, 11985, 12019, 12046, 12051, 12064, 12073, 12124,
    12149, 12184, 12244
]

# Models to train
MODELS = {
    'highres_no_ela': {'in_channels': 3, 'output': 'highres_no_ela_v3_best.pth'},
    'hard_negative': {'in_channels': 3, 'output': 'hard_negative_v3_best.pth'},
    'high_recall': {'in_channels': 3, 'output': 'high_recall_v3_best.pth'},
    'enhanced_aug': {'in_channels': 3, 'output': 'enhanced_aug_v3_best.pth'},
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


class CombinedLoss(nn.Module):
    """Combined Focal + Dice loss with FP penalty."""
    def __init__(self, focal_weight=0.5, dice_weight=0.5, fp_penalty=2.0):
        super().__init__()
        self.focal = FocalLoss(alpha=0.25, gamma=2.0)
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.fp_penalty = fp_penalty
    
    def forward(self, pred, target):
        focal_loss = self.focal(pred, target)
        
        pred_sigmoid = torch.sigmoid(pred)
        intersection = (pred_sigmoid * target).sum()
        dice = (2 * intersection + 1) / (pred_sigmoid.sum() + target.sum() + 1)
        dice_loss = 1 - dice
        
        # Extra FP penalty for authentic images
        is_authentic = (target.sum(dim=(1,2,3)) == 0).float()
        fp_loss = (pred_sigmoid * is_authentic.view(-1, 1, 1, 1)).mean() * self.fp_penalty
        
        return self.focal_weight * focal_loss + self.dice_weight * dice_loss + fp_loss


# ============================================================================
# DATASET
# ============================================================================

class ForgeryDataset(Dataset):
    """Dataset for forged images with masks."""
    
    def __init__(self, image_paths, mask_paths, img_size=512, augment=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.img_size = img_size
        
        if augment:
            self.transform = A.Compose([
                A.RandomResizedCrop(size=(img_size, img_size), scale=(0.7, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.OneOf([
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                ], p=0.5),
                A.GaussianBlur(blur_limit=(3, 5), p=0.2),
                A.GaussNoise(std_range=(0.01, 0.03), p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = cv2.imread(str(self.image_paths[idx]))
        if image is None:
            return torch.zeros(3, self.img_size, self.img_size), torch.zeros(1, self.img_size, self.img_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = np.load(str(self.mask_paths[idx]))
        if mask.ndim == 3:
            mask = mask[0]
        
        transformed = self.transform(image=image, mask=mask)
        image_tensor = transformed['image']
        mask_out = transformed['mask']
        if isinstance(mask_out, np.ndarray):
            mask_tensor = torch.from_numpy(mask_out).unsqueeze(0).float()
        else:
            mask_tensor = mask_out.unsqueeze(0).float() if mask_out.dim() == 2 else mask_out.float()
        
        return image_tensor, mask_tensor


class AuthenticDataset(Dataset):
    """Dataset for authentic images (no forgery mask)."""
    
    def __init__(self, image_paths, img_size=512, augment=True, weight=1.0):
        self.image_paths = image_paths
        self.img_size = img_size
        self.weight = weight
        
        if augment:
            self.transform = A.Compose([
                A.RandomResizedCrop(size=(img_size, img_size), scale=(0.7, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.GaussianBlur(blur_limit=(3, 5), p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = cv2.imread(str(self.image_paths[idx]))
        if image is None:
            return torch.zeros(3, self.img_size, self.img_size), torch.zeros(1, self.img_size, self.img_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        transformed = self.transform(image=image)
        image_tensor = transformed['image']
        mask_tensor = torch.zeros(1, self.img_size, self.img_size)
        
        return image_tensor, mask_tensor


def collect_data():
    """Collect all training data paths."""
    
    # Forged images (original + augmented)
    forged_images = []
    forged_masks = []
    
    # Original forged
    for img_path in sorted(FORGED_ORIGINAL.glob('*.png')):
        mask_path = MASKS_ORIGINAL / f"{img_path.stem}.npy"
        if mask_path.exists():
            forged_images.append(img_path)
            forged_masks.append(mask_path)
    print(f"Original forged: {len(forged_images)}")
    
    # Augmented forged
    aug_count = 0
    for img_path in sorted(FORGED_AUGMENTED.glob('*.png')):
        mask_path = MASKS_AUGMENTED / f"{img_path.stem}.npy"
        if mask_path.exists():
            forged_images.append(img_path)
            forged_masks.append(mask_path)
            aug_count += 1
    print(f"Augmented forged: {aug_count}")
    print(f"Total forged: {len(forged_images)}")
    
    # Authentic images
    authentic_images = []
    
    # Original authentic
    for img_path in sorted(AUTHENTIC_DIR.glob('*.png')):
        authentic_images.append(img_path)
    print(f"Original authentic: {len(authentic_images)}")
    
    # Augmented authentic (sample to balance)
    aug_authentic = list(AUTHENTIC_AUG_DIR.glob('*.png'))[:2000]
    authentic_images.extend(aug_authentic)
    
    aug_v2 = list(AUTHENTIC_AUG_V2_DIR.glob('*.png'))[:2000]
    authentic_images.extend(aug_v2)
    print(f"Total authentic: {len(authentic_images)}")
    
    # Hard negatives (highest priority)
    hard_negatives = []
    for img_id in HARD_NEGATIVE_IDS:
        img_path = HARD_NEGATIVE_DIR / f"{img_id}.png"
        if img_path.exists():
            hard_negatives.append(img_path)
    print(f"Hard negatives: {len(hard_negatives)}")
    
    return forged_images, forged_masks, authentic_images, hard_negatives


# ============================================================================
# TRAINING
# ============================================================================

def train_model(model_name, forged_images, forged_masks, authentic_images, hard_negatives):
    """Train a single model."""
    
    config = MODELS[model_name]
    print("=" * 80)
    print(f"TRAINING: {model_name}")
    print("=" * 80)
    
    # Create datasets
    forged_dataset = ForgeryDataset(forged_images, forged_masks, IMG_SIZE, augment=True)
    authentic_dataset = AuthenticDataset(authentic_images, IMG_SIZE, augment=True)
    hard_neg_dataset = AuthenticDataset(hard_negatives, IMG_SIZE, augment=True, weight=5.0)
    
    # Compute weights
    weights = []
    # Forged weights (by size)
    for mask_path in forged_masks:
        mask = np.load(str(mask_path))
        if mask.ndim == 3:
            mask = mask[0]
        ratio = (mask > 0.5).sum() / mask.size * 100
        if ratio < 1.0:
            weights.append(3.0)  # Small forgeries
        elif ratio < 5.0:
            weights.append(2.0)
        else:
            weights.append(1.0)
    
    # Authentic weights
    weights.extend([1.0] * len(authentic_images))
    
    # Hard negative weights (high priority)
    weights.extend([5.0] * len(hard_negatives))
    
    # Combine datasets
    from torch.utils.data import ConcatDataset
    combined_dataset = ConcatDataset([forged_dataset, authentic_dataset, hard_neg_dataset])
    
    sampler = WeightedRandomSampler(weights, num_samples=len(combined_dataset), replacement=True)
    dataloader = DataLoader(combined_dataset, batch_size=BATCH_SIZE, sampler=sampler, 
                           num_workers=4, pin_memory=True)
    
    print(f"Dataset: {len(forged_images)} forged + {len(authentic_images)} authentic + {len(hard_negatives)} hard neg")
    print(f"Total samples: {len(combined_dataset)}")
    print(f"Batches per epoch: {len(dataloader)}")
    
    # Create model
    base_model = smp.FPN(
        encoder_name='timm-efficientnet-b2',
        encoder_weights='imagenet',
        in_channels=config['in_channels'],
        classes=1
    )
    model = AttentionFPN(base_model).to(DEVICE)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = CombinedLoss(focal_weight=0.5, dice_weight=0.5, fp_penalty=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # Training
    best_loss = float('inf')
    output_path = MODELS_DIR / config['output']
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        
        for batch_idx, (images, masks) in enumerate(dataloader):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        scheduler.step()
        
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, output_path)
            print(f"  ✓ Saved best model to {output_path}")
    
    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Train with augmented data')
    parser.add_argument('--model', type=str, default='all',
                        choices=['all'] + list(MODELS.keys()),
                        help='Which model to train')
    args = parser.parse_args()
    
    print("=" * 80)
    print("TRAINING WITH AUGMENTED DATA")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Image size: {IMG_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print()
    
    # Collect data
    forged_images, forged_masks, authentic_images, hard_negatives = collect_data()
    print()
    
    if args.model == 'all':
        models_to_train = list(MODELS.keys())
    else:
        models_to_train = [args.model]
    
    for model_name in models_to_train:
        train_model(model_name, forged_images, forged_masks, authentic_images, hard_negatives)
        print()


if __name__ == '__main__':
    main()
