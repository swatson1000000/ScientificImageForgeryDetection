#!/usr/bin/env python3
"""
Hard False-Negative Mining Training for Scientific Image Forgery Detection
Identifies forgeries that the model misses and oversamples them in training.
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
MODEL_SAVE_PATH = PROJECT_DIR / 'models' / 'high_recall_best.pth'
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


def find_hard_false_negatives(model, image_paths, mask_paths, threshold=0.5):
    """
    Find forged images where the model fails to detect the forgery.
    These are 'hard false negatives' - forgeries the model misses.
    """
    model.eval()
    hard_fn = []
    
    transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    print(f"Scanning {len(image_paths)} forged images for missed forgeries...")
    
    for img_path, mask_path in zip(image_paths, mask_paths):  # Finding hard FN
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load ground truth mask
        gt_mask = np.load(str(mask_path))
        if gt_mask.ndim == 3:
            gt_mask = gt_mask[0]
        
        # Prepare input
        transformed = transform(image=image_rgb)
        input_tensor = transformed['image'].unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            pred = torch.sigmoid(model(input_tensor))
            pred = pred.cpu().numpy()[0, 0]
        
        # Resize GT to match prediction
        gt_resized = cv2.resize(gt_mask.astype(np.float32), (IMG_SIZE, IMG_SIZE))
        
        # Calculate recall on this image
        gt_binary = (gt_resized > 0.5)
        pred_binary = (pred > threshold)
        
        gt_pixels = gt_binary.sum()
        if gt_pixels == 0:
            continue
            
        detected_pixels = (pred_binary & gt_binary).sum()
        recall = detected_pixels / gt_pixels
        
        # Also calculate forgery size
        forgery_ratio = gt_pixels / gt_binary.size * 100
        
        # If recall < 50%, this is a hard false negative
        if recall < 0.5:
            hard_fn.append({
                'image_path': str(img_path),
                'mask_path': str(mask_path),
                'recall': recall,
                'forgery_ratio': forgery_ratio,
                'max_pred': pred.max()
            })
    
    # Sort by recall (worst first)
    hard_fn.sort(key=lambda x: x['recall'])
    
    print(f"\nFound {len(hard_fn)} hard false negatives (recall < 50%)")
    if hard_fn:
        print(f"Worst recall: {hard_fn[0]['recall']*100:.1f}%")
        print(f"Average forgery size of missed cases: {np.mean([x['forgery_ratio'] for x in hard_fn]):.2f}%")
        
        # Analyze by size
        small = [x for x in hard_fn if x['forgery_ratio'] < 2]
        medium = [x for x in hard_fn if 2 <= x['forgery_ratio'] < 10]
        large = [x for x in hard_fn if x['forgery_ratio'] >= 10]
        print(f"By size - Small (<2%): {len(small)}, Medium (2-10%): {len(medium)}, Large (>10%): {len(large)}")
    
    return hard_fn


class HardFNDataset(Dataset):
    """Dataset that oversamples hard false negatives."""
    
    def __init__(self, image_paths, mask_paths, hard_fn_indices, img_size=512, augment=True, hard_fn_weight=5.0):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.hard_fn_indices = set(hard_fn_indices)
        self.img_size = img_size
        self.augment = augment
        
        # Compute weights - hard FN get much higher weight
        self.weights = []
        for i, mask_path in enumerate(mask_paths):
            if i in self.hard_fn_indices:
                # Hard false negative - high weight
                self.weights.append(hard_fn_weight)
            else:
                # Normal sample - weight by forgery size
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
                A.RandomResizedCrop(size=(img_size, img_size), scale=(0.6, 1.0)),  # More aggressive crop
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Affine(shift_limit=0.15, scale_limit=0.25, rotate_limit=45, p=0.5),
                A.OneOf([
                    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
                    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
                    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=30),
                ], p=0.7),
                A.OneOf([
                    A.GaussNoise(),
                    A.GaussianBlur(blur_limit=(3, 7)),
                    A.MotionBlur(blur_limit=7),
                ], p=0.3),
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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = np.load(str(self.mask_paths[idx]))
        if mask.ndim == 3:
            mask = mask[0]
        
        # Resize to consistent size
        image = cv2.resize(image, (self.img_size, self.img_size))
        mask = cv2.resize(mask.astype(np.float32), (self.img_size, self.img_size))
        
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
    print("HARD FALSE-NEGATIVE MINING TRAINING")
    print("Goal: Improve recall by targeting missed forgeries")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
    print()
    
    # Step 1: Load all training data
    print("Step 1: Loading training data...")
    forged_dir = TRAIN_IMAGES_PATH / 'forged'
    
    image_paths = []
    mask_paths = []
    
    for f in sorted(forged_dir.glob('*.png')):
        mask_path = TRAIN_MASKS_PATH / f"{f.stem}.npy"
        if mask_path.exists():
            image_paths.append(str(f))
            mask_paths.append(str(mask_path))
    
    print(f"Total forged images: {len(image_paths)}")
    
    # Step 2: Load best model and find hard false negatives
    print("\nStep 2: Loading best model to find hard false negatives...")
    best_model = load_model(BEST_MODEL_PATH, DEVICE)
    
    hard_fn = find_hard_false_negatives(best_model, image_paths, mask_paths, threshold=0.5)
    
    # Get indices of hard FN
    hard_fn_paths = set(x['image_path'] for x in hard_fn)
    hard_fn_indices = [i for i, p in enumerate(image_paths) if p in hard_fn_paths]
    
    print(f"\nHard false negatives: {len(hard_fn_indices)} images")
    
    # Free memory
    del best_model
    torch.cuda.empty_cache()
    
    # Step 3: Split train/val
    split_idx = int(len(image_paths) * 0.9)
    train_images = image_paths[:split_idx]
    train_masks = mask_paths[:split_idx]
    val_images = image_paths[split_idx:]
    val_masks = mask_paths[split_idx:]
    
    # Adjust hard_fn_indices for train set only
    train_hard_fn_indices = [i for i in hard_fn_indices if i < split_idx]
    
    print(f"\nTrain: {len(train_images)} images ({len(train_hard_fn_indices)} hard FN)")
    print(f"Val: {len(val_images)} images")
    
    # Create datasets
    train_dataset = HardFNDataset(
        train_images, train_masks, 
        train_hard_fn_indices,
        IMG_SIZE, 
        augment=True,
        hard_fn_weight=5.0  # 5x oversampling for hard cases
    )
    val_dataset = HardFNDataset(val_images, val_masks, [], IMG_SIZE, augment=False)
    
    # Weighted sampler
    sampler = WeightedRandomSampler(
        weights=train_dataset.weights,
        num_samples=len(train_dataset),
        replacement=True
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    # Create model (fresh, not fine-tuning)
    base_model = smp.FPN(
        encoder_name='timm-efficientnet-b2',
        encoder_weights='imagenet',
        in_channels=3,
        classes=1
    )
    model = AttentionFPN(base_model).to(DEVICE)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {num_params:,}")
    
    # Training setup - use higher alpha for focal loss to focus on hard cases
    criterion = FocalLoss(alpha=0.5, gamma=2.5)  # Higher alpha, higher gamma
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    
    # Training loop
    best_recall = 0
    best_mcc = 0
    
    print("\nStep 3: Training with hard false-negative oversampling...")
    for epoch in range(EPOCHS):
        start_time = time.time()
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_metrics = validate(model, val_loader, DEVICE)
        scheduler.step()
        
        elapsed = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{EPOCHS} ({elapsed:.1f}s) - "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_metrics['loss']:.4f}, "
              f"Val F1: {val_metrics['f1']:.4f}, Val MCC: {val_metrics['mcc']:.4f}, "
              f"Val Recall: {val_metrics['recall']:.4f}, Val Prec: {val_metrics['precision']:.4f}")
        
        # Save best by recall (our goal)
        if val_metrics['recall'] > best_recall:
            best_recall = val_metrics['recall']
            best_mcc = val_metrics['mcc']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_recall': best_recall,
                'best_mcc': best_mcc,
                'hard_fn_count': len(hard_fn_indices),
            }, MODEL_SAVE_PATH)
            print(f"  ✓ Saved best model (Recall: {best_recall:.4f}, MCC: {best_mcc:.4f})")
    
    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best Val Recall: {best_recall:.4f}")
    print(f"Best Val MCC: {best_mcc:.4f}")
    print(f"Model saved to: {MODEL_SAVE_PATH}")


if __name__ == '__main__':
    main()
