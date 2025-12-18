#!/usr/bin/env python3
"""
Hard Negative Mining Training for Scientific Image Forgery Detection
Identifies false positive cases and uses them to improve model training.
"""

import os
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

# Configuration
IMG_SIZE = 512
BATCH_SIZE = 4
EPOCHS = 30
LR = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
TRAIN_IMAGES_PATH = PROJECT_DIR / 'train_images'
TRAIN_MASKS_PATH = PROJECT_DIR / 'train_masks'
AUTHENTIC_PATH = PROJECT_DIR / 'train_images' / 'authentic'
MODEL_SAVE_PATH = PROJECT_DIR / 'models' / 'hard_negative_best.pth'
BEST_MODEL_PATH = PROJECT_DIR / 'models' / 'highres_no_ela_best.pth'


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


def load_model(model_path, device):
    """Load pretrained model."""
    base_model = smp.FPN(
        encoder_name='timm-efficientnet-b2',
        encoder_weights=None,
        in_channels=3,
        classes=1
    )
    model = AttentionFPN(base_model).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def find_hard_negatives(model, authentic_dir, threshold=0.5, max_samples=500):
    """
    Find authentic images that produce false positives.
    These are 'hard negatives' that confuse the model.
    """
    model.eval()
    hard_negatives = []
    
    authentic_images = list(Path(authentic_dir).glob('*.png'))
    print(f"Scanning {len(authentic_images)} authentic images for hard negatives...")
    
    transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    for img_path in authentic_images:  # Finding hard negatives
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Prepare input
        transformed = transform(image=image)
        input_tensor = transformed['image'].unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            pred = torch.sigmoid(model(input_tensor))
            pred = pred.cpu().numpy()[0, 0]
        
        # Check if this produces false positives
        fp_pixels = (pred > threshold).sum()
        fp_ratio = fp_pixels / pred.size
        
        if fp_ratio > 0.001:  # More than 0.1% false positive pixels
            hard_negatives.append({
                'path': str(img_path),
                'fp_ratio': fp_ratio,
                'max_conf': pred.max()
            })
    
    # Sort by FP ratio (worst first)
    hard_negatives.sort(key=lambda x: x['fp_ratio'], reverse=True)
    
    print(f"Found {len(hard_negatives)} hard negative images")
    if hard_negatives:
        print(f"Worst FP ratio: {hard_negatives[0]['fp_ratio']*100:.2f}%")
    
    return hard_negatives[:max_samples]


class HardNegativeDataset(Dataset):
    """Dataset that includes both forged images and hard negative authentic images."""
    
    def __init__(self, forged_paths, mask_paths, hard_negative_paths, img_size=512, augment=True):
        self.forged_paths = forged_paths
        self.mask_paths = mask_paths
        self.hard_negative_paths = hard_negative_paths
        self.img_size = img_size
        self.augment = augment
        
        # Combine forged and hard negatives
        # Hard negatives get empty masks (all zeros)
        self.all_paths = list(zip(forged_paths, mask_paths, [True]*len(forged_paths)))
        self.all_paths += [(hn, None, False) for hn in hard_negative_paths]
        
        # Compute weights - hard negatives get higher weight
        self.weights = []
        for img_path, mask_path, is_forged in self.all_paths:
            if not is_forged:
                # Hard negatives get 2x weight
                self.weights.append(2.0)
            else:
                # Forged images weighted by forgery size
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
        return len(self.all_paths)
    
    def __getitem__(self, idx):
        img_path, mask_path, is_forged = self.all_paths[idx]
        
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.img_size, self.img_size))
        
        if is_forged and mask_path:
            mask = np.load(str(mask_path))
            if mask.ndim == 3:
                mask = mask[0]
            mask = cv2.resize(mask.astype(np.float32), (self.img_size, self.img_size))
        else:
            # Hard negative - empty mask
            mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        
        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']
        
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        mask = mask.float().unsqueeze(0)
        
        return image, mask


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
    
    # FP rate on negatives (authentic)
    fp_rate = total_fp / (total_fp + total_tn) if (total_fp + total_tn) > 0 else 0
    
    num = total_tp * total_tn - total_fp * total_fn
    den = (total_tp + total_fp) * (total_tp + total_fn) * (total_tn + total_fp) * (total_tn + total_fn)
    mcc = num / (den ** 0.5) if den > 0 else 0
    
    return {
        'loss': total_loss / len(loader),
        'f1': f1,
        'mcc': mcc,
        'recall': recall,
        'precision': precision,
        'fp_rate': fp_rate
    }


def main():
    print("=" * 80)
    print("HARD NEGATIVE MINING TRAINING")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
    print()
    
    # Step 1: Load best model and find hard negatives
    print("Step 1: Loading best model to find hard negatives...")
    best_model = load_model(BEST_MODEL_PATH, DEVICE)
    
    hard_negatives = find_hard_negatives(
        best_model, 
        AUTHENTIC_PATH, 
        threshold=0.5,
        max_samples=300
    )
    hard_negative_paths = [hn['path'] for hn in hard_negatives]
    
    # Free memory
    del best_model
    torch.cuda.empty_cache()
    
    # Step 2: Load forged training data
    print("\nStep 2: Loading forged training data...")
    forged_dir = TRAIN_IMAGES_PATH / 'forged'
    
    forged_paths = []
    mask_paths = []
    
    for f in sorted(forged_dir.glob('*.png')):
        mask_path = TRAIN_MASKS_PATH / f"{f.stem}.npy"
        if mask_path.exists():
            forged_paths.append(str(f))
            mask_paths.append(str(mask_path))
    
    print(f"Forged images: {len(forged_paths)}")
    print(f"Hard negatives: {len(hard_negative_paths)}")
    print(f"Total training samples: {len(forged_paths) + len(hard_negative_paths)}")
    
    # Split train/val
    split_idx = int(len(forged_paths) * 0.9)
    train_forged = forged_paths[:split_idx]
    train_masks = mask_paths[:split_idx]
    val_forged = forged_paths[split_idx:]
    val_masks = mask_paths[split_idx:]
    
    # Add some hard negatives to val set
    hn_split = int(len(hard_negative_paths) * 0.9)
    train_hn = hard_negative_paths[:hn_split]
    val_hn = hard_negative_paths[hn_split:]
    
    print(f"Train: {len(train_forged)} forged + {len(train_hn)} hard negatives")
    print(f"Val: {len(val_forged)} forged + {len(val_hn)} hard negatives")
    
    # Create datasets
    train_dataset = HardNegativeDataset(train_forged, train_masks, train_hn, IMG_SIZE, augment=True)
    val_dataset = HardNegativeDataset(val_forged, val_masks, val_hn, IMG_SIZE, augment=False)
    
    # Weighted sampler
    sampler = WeightedRandomSampler(
        weights=train_dataset.weights,
        num_samples=len(train_dataset),
        replacement=True
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    # Create fresh model (don't fine-tune, train from scratch with hard negatives)
    base_model = smp.FPN(
        encoder_name='timm-efficientnet-b2',
        encoder_weights='imagenet',
        in_channels=3,
        classes=1
    )
    model = AttentionFPN(base_model).to(DEVICE)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {num_params:,}")
    
    # Training setup
    criterion = FocalLoss(gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    
    # Training loop
    best_mcc = 0
    
    print("\nStep 3: Training with hard negatives...")
    for epoch in range(EPOCHS):
        start_time = time.time()
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_metrics = validate(model, val_loader, DEVICE)
        scheduler.step()
        
        elapsed = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{EPOCHS} ({elapsed:.1f}s) - "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_metrics['loss']:.4f}, "
              f"Val F1: {val_metrics['f1']:.4f}, Val MCC: {val_metrics['mcc']:.4f}, "
              f"Val Recall: {val_metrics['recall']:.4f}, Val FP Rate: {val_metrics['fp_rate']:.4f}")
        
        if val_metrics['mcc'] > best_mcc:
            best_mcc = val_metrics['mcc']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_mcc': best_mcc,
                'hard_negatives_count': len(hard_negative_paths),
            }, MODEL_SAVE_PATH)
            print(f"  ✓ Saved best model (MCC: {best_mcc:.4f})")
    
    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best Val MCC: {best_mcc:.4f}")
    print(f"Model saved to: {MODEL_SAVE_PATH}")


if __name__ == '__main__':
    main()
