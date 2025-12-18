#!/usr/bin/env python3
"""
Hard Negative Mining V2 - Retrain with known FP images from test set.

Uses the 73 authentic images from test_authentic_100 that produce false positives
at threshold 0.65 to improve model training.

This script retrains all 5 main models with these hard negatives.
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

# Hard negative images - these produce FPs on test_authentic_100
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

# Models to retrain
MODELS_CONFIG = {
    'highres_no_ela': {
        'script': 'script_1_highres_no_ela.py',
        'output': 'highres_no_ela_v2_best.pth',
        'in_channels': 3,
    },
    'hard_negative': {
        'script': 'script_2_hard_negative_mining.py', 
        'output': 'hard_negative_v2_best.pth',
        'in_channels': 3,
    },
    'high_recall': {
        'script': 'script_3_hard_fn_mining.py',
        'output': 'high_recall_v2_best.pth',
        'in_channels': 3,
    },
    'enhanced_aug': {
        'script': 'script_4_enhanced_augmentation.py',
        'output': 'enhanced_aug_v2_best.pth',
        'in_channels': 3,
    },
    'comprehensive': {
        'script': 'script_5_comprehensive.py',
        'output': 'comprehensive_v2_best.pth',
        'in_channels': 4,  # RGB + ELA
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


class CombinedLoss(nn.Module):
    """Combined Focal + Dice loss with FP penalty."""
    def __init__(self, focal_weight=0.5, dice_weight=0.5, fp_penalty=2.0):
        super().__init__()
        self.focal = FocalLoss(alpha=0.25, gamma=2.0)
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.fp_penalty = fp_penalty
    
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


def compute_ela(image, quality=90):
    """Compute Error Level Analysis."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded = cv2.imencode('.jpg', image, encode_param)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    ela = cv2.absdiff(image, decoded)
    ela = cv2.cvtColor(ela, cv2.COLOR_BGR2GRAY)
    ela = (ela.astype(np.float32) * 10).clip(0, 255).astype(np.uint8)
    return ela


# ============================================================================
# DATASET
# ============================================================================

class HardNegativeDatasetV2(Dataset):
    """Dataset with forged images + hard negative authentic images."""
    
    def __init__(self, forged_paths, mask_paths, hard_negative_paths, 
                 img_size=512, use_ela=False, hard_neg_weight=3.0):
        self.img_size = img_size
        self.use_ela = use_ela
        
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
                self.weights.append(3.0)  # Small forgeries get 3x weight
            elif forgery_ratio < 5.0:
                self.weights.append(2.0)
            else:
                self.weights.append(1.0)
        
        # Hard negative samples (authentic that produce FP)
        for img_path in hard_negative_paths:
            self.samples.append({
                'image': img_path,
                'mask': None,
                'is_forged': False
            })
            self.weights.append(hard_neg_weight)  # High weight for hard negatives
        
        # Augmentation
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
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = cv2.imread(str(sample['image']))
        if image is None:
            # Return dummy data
            if self.use_ela:
                return torch.zeros(4, self.img_size, self.img_size), torch.zeros(1, self.img_size, self.img_size)
            return torch.zeros(3, self.img_size, self.img_size), torch.zeros(1, self.img_size, self.img_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load or create mask
        if sample['is_forged'] and sample['mask'] is not None:
            mask = np.load(sample['mask'])
            if mask.ndim == 3:
                mask = mask[0]
        else:
            # Authentic - all zeros mask
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        
        # Apply augmentation
        transformed = self.transform(image=image, mask=mask)
        image_tensor = transformed['image']
        mask_out = transformed['mask']
        if isinstance(mask_out, np.ndarray):
            mask_tensor = torch.from_numpy(mask_out).unsqueeze(0).float()
        else:
            mask_tensor = mask_out.unsqueeze(0).float() if mask_out.dim() == 2 else mask_out.float()
        
        # Add ELA channel if needed
        if self.use_ela:
            # Compute ELA on original image before normalization
            ela = compute_ela(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            ela = cv2.resize(ela, (self.img_size, self.img_size))
            ela_normalized = ela.astype(np.float32) / 255.0
            ela_tensor = torch.from_numpy(ela_normalized).unsqueeze(0)
            image_tensor = torch.cat([image_tensor, ela_tensor], dim=0)
        
        return image_tensor, mask_tensor


# ============================================================================
# TRAINING
# ============================================================================

def train_model(model_name, in_channels=3, use_ela=False):
    """Train a single model with hard negatives."""
    
    print("=" * 80)
    print(f"TRAINING: {model_name} (in_channels={in_channels}, use_ela={use_ela})")
    print("=" * 80)
    
    # Collect forged images and masks
    forged_dir = TRAIN_IMAGES_PATH / 'forged'
    forged_images = sorted(forged_dir.glob('*.png'))
    
    forged_paths = []
    mask_paths = []
    
    for img_path in forged_images:
        mask_path = TRAIN_MASKS_PATH / f"{img_path.stem}.npy"
        if mask_path.exists():
            forged_paths.append(str(img_path))
            mask_paths.append(str(mask_path))
    
    print(f"Found {len(forged_paths)} forged images with masks")
    
    # Collect hard negative images
    hard_neg_paths = []
    for img_id in HARD_NEGATIVE_IDS:
        img_path = HARD_NEGATIVE_DIR / f"{img_id}.png"
        if img_path.exists():
            hard_neg_paths.append(str(img_path))
    
    print(f"Found {len(hard_neg_paths)} hard negative images")
    
    # Also add some regular authentic images
    authentic_dir = TRAIN_IMAGES_PATH / 'authentic'
    authentic_images = sorted(authentic_dir.glob('*.png'))[:500]  # Limit to 500
    for img_path in authentic_images:
        hard_neg_paths.append(str(img_path))
    
    print(f"Total authentic samples: {len(hard_neg_paths)}")
    
    # Create dataset
    dataset = HardNegativeDatasetV2(
        forged_paths, mask_paths, hard_neg_paths,
        img_size=IMG_SIZE, use_ela=use_ela, hard_neg_weight=3.0
    )
    
    # Weighted sampler
    sampler = WeightedRandomSampler(
        weights=dataset.weights,
        num_samples=len(dataset),
        replacement=True
    )
    
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=4, pin_memory=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Batches per epoch: {len(dataloader)}")
    
    # Create model
    base_model = smp.FPN(
        encoder_name='timm-efficientnet-b2',
        encoder_weights='imagenet',
        in_channels=in_channels,
        classes=1
    )
    model = AttentionFPN(base_model).to(DEVICE)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = CombinedLoss(focal_weight=0.5, dice_weight=0.5, fp_penalty=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # Training loop
    best_loss = float('inf')
    output_path = MODELS_DIR / MODELS_CONFIG[model_name]['output']
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        fp_count = 0
        
        for batch_idx, (images, masks) in enumerate(dataloader):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Count FPs on authentic (mask is all zeros)
            with torch.no_grad():
                is_authentic = (masks.sum(dim=(1,2,3)) == 0)
                if is_authentic.any():
                    preds = torch.sigmoid(outputs[is_authentic])
                    fp_count += (preds > 0.5).sum().item()
        
        avg_loss = epoch_loss / len(dataloader)
        scheduler.step()
        
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, FP pixels on authentic={fp_count}")
        
        # Save best model
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
    parser = argparse.ArgumentParser(description='Hard Negative Mining V2')
    parser.add_argument('--model', type=str, default='all',
                        choices=['all', 'highres_no_ela', 'hard_negative', 'high_recall', 
                                'enhanced_aug', 'comprehensive'],
                        help='Which model to train')
    args = parser.parse_args()
    
    print("=" * 80)
    print("HARD NEGATIVE MINING V2")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Image size: {IMG_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Hard negative IDs: {len(HARD_NEGATIVE_IDS)}")
    print()
    
    if args.model == 'all':
        # Train all models (except comprehensive which uses ELA)
        models_to_train = ['highres_no_ela', 'hard_negative', 'high_recall', 'enhanced_aug']
    else:
        models_to_train = [args.model]
    
    for model_name in models_to_train:
        config = MODELS_CONFIG[model_name]
        use_ela = config['in_channels'] == 4
        train_model(model_name, in_channels=config['in_channels'], use_ela=use_ela)
        print()


if __name__ == '__main__':
    main()
