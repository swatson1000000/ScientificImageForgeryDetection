#!/usr/bin/env python3
"""
Small Object Augmentation Training V5

Train a model specifically optimized for detecting small forgeries using:
1. Higher resolution (768x768) to preserve small details
2. Random crops that force model to see small objects at larger scale
3. Mosaic augmentation - 4 images combined
4. Copy-paste augmentation - paste forgeries onto authentic backgrounds
5. Heavier weighting for small forgeries
6. Scale augmentation with zoom-in bias

Target: Improve detection of the 553 missed forgeries (mostly small, low-contrast)
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
import random

# ============================================================================
# CONFIGURATION
# ============================================================================

IMG_SIZE = 768  # Higher resolution for small objects
BATCH_SIZE = 2  # Reduced due to higher resolution
EPOCHS = 30
LR = 5e-5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
TRAIN_IMAGES_PATH = PROJECT_DIR / 'train_images'
TRAIN_MASKS_PATH = PROJECT_DIR / 'train_masks'
MODELS_DIR = PROJECT_DIR / 'models'

# Base model to fine-tune
BASE_MODEL = MODELS_DIR / 'highres_no_ela_v4_best.pth'
OUTPUT_MODEL = MODELS_DIR / 'small_object_v5_best.pth'

# Missed forgeries file
MISSED_FORGERIES_FILE = SCRIPT_DIR / 'missed_forgeries.txt'

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
# SMALL OBJECT LOSS
# ============================================================================

class SmallObjectLoss(nn.Module):
    """Loss weighted towards small objects with boundary emphasis."""
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
        """Emphasize boundary accuracy for small objects."""
        pred_sigmoid = torch.sigmoid(pred)
        
        # Sobel-like edge detection
        kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], 
                              dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)
        
        if target.dim() == 4:
            target_edges = F.conv2d(target, kernel, padding=1)
        else:
            target_edges = F.conv2d(target.unsqueeze(1), kernel, padding=1)
        
        pred_edges = F.conv2d(pred_sigmoid, kernel, padding=1)
        
        # Weight errors at boundaries
        boundary_weight = (target_edges.abs() > 0.1).float() * 5 + 1
        boundary_error = ((pred_sigmoid - target) ** 2 * boundary_weight).mean()
        
        return boundary_error
    
    def forward(self, pred, target):
        focal = self.focal_loss(pred, target)
        dice = self.dice_loss(pred, target)
        boundary = self.boundary_loss(pred, target)
        return self.focal_weight * focal + self.dice_weight * dice + self.boundary_weight * boundary


# ============================================================================
# DATASET WITH AGGRESSIVE AUGMENTATION
# ============================================================================

class SmallObjectDataset(Dataset):
    """Dataset with aggressive augmentation for small object detection."""
    
    def __init__(self, forged_paths, mask_paths, missed_set, img_size=768, 
                 is_train=True, copy_paste_prob=0.3, mosaic_prob=0.2):
        self.img_size = img_size
        self.is_train = is_train
        self.missed_set = missed_set
        self.copy_paste_prob = copy_paste_prob
        self.mosaic_prob = mosaic_prob
        
        # Store all samples
        self.forged_paths = forged_paths
        self.mask_paths = mask_paths
        
        # Build sample list with weights
        self.samples = []
        self.weights = []
        
        for img_path, mask_path in zip(forged_paths, mask_paths):
            self.samples.append({
                'image': img_path,
                'mask': mask_path,
            })
            
            # Heavy weight for missed/small forgeries
            img_name = img_path.name
            mask = np.load(mask_path)
            if mask.ndim == 3:
                mask = mask[0]
            forgery_ratio = (mask > 0.5).sum() / mask.size * 100
            
            # Weight by: missed status + small size
            weight = 1.0
            if img_name in missed_set:
                weight *= 4.0  # 4x weight for missed
            if forgery_ratio < 1.0:
                weight *= 3.0  # 3x for very small
            elif forgery_ratio < 3.0:
                weight *= 2.0  # 2x for small
            
            self.weights.append(weight)
        
        print(f"Dataset: {len(self.samples)} samples")
        print(f"  Missed forgeries: {sum(1 for s in self.samples if s['image'].name in missed_set)}")
        print(f"  Max weight: {max(self.weights):.1f}")
        
        # Build transforms
        if is_train:
            self.transform = A.Compose([
                # Aggressive scale augmentation with zoom-in bias for small objects
                A.RandomResizedCrop(
                    size=(img_size, img_size),
                    scale=(0.5, 1.0),  # Zoom in up to 2x
                    ratio=(0.8, 1.2),
                    p=1.0
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                # Rotation for more variety
                A.Rotate(limit=15, p=0.3, border_mode=cv2.BORDER_REFLECT_101),
                # Aggressive noise/blur
                A.OneOf([
                    A.GaussNoise(std_range=(0.01, 0.08)),
                    A.GaussianBlur(blur_limit=(3, 5)),
                    A.MedianBlur(blur_limit=5),
                    A.MotionBlur(blur_limit=5),
                ], p=0.4),
                # Color augmentation
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
                    A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=30),
                    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
                    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8)),  # Good for low contrast
                ], p=0.6),
                # Cutout to force partial detection
                A.CoarseDropout(
                    num_holes_range=(1, 4),
                    hole_height_range=(20, 60),
                    hole_width_range=(20, 60),
                    fill="random",
                    p=0.3
                ),
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
        return len(self.samples)
    
    def load_sample(self, idx):
        """Load image and mask for a given index."""
        sample = self.samples[idx]
        
        image = cv2.imread(str(sample['image']))
        if image is None:
            raise ValueError(f"Could not load {sample['image']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = np.load(str(sample['mask']))
        if mask.ndim == 3:
            mask = mask[0]
        mask = (mask > 0.5).astype(np.float32)
        
        return image, mask
    
    def apply_copy_paste(self, image, mask, donor_idx):
        """Copy forgery region from donor and paste onto current image."""
        donor_image, donor_mask = self.load_sample(donor_idx)
        
        # Find forgery region in donor
        if donor_mask.max() == 0:
            return image, mask
        
        # Resize donor to match current image
        h, w = image.shape[:2]
        donor_image = cv2.resize(donor_image, (w, h))
        donor_mask = cv2.resize(donor_mask, (w, h))
        
        # Find bounding box of donor forgery
        coords = np.where(donor_mask > 0.5)
        if len(coords[0]) == 0:
            return image, mask
        
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        
        # Extract forgery patch
        patch_h = y_max - y_min
        patch_w = x_max - x_min
        
        if patch_h < 10 or patch_w < 10:
            return image, mask
        
        # Random placement in target
        max_y = max(0, h - patch_h)
        max_x = max(0, w - patch_w)
        new_y = random.randint(0, max_y) if max_y > 0 else 0
        new_x = random.randint(0, max_x) if max_x > 0 else 0
        
        # Create paste mask
        paste_mask = donor_mask[y_min:y_max, x_min:x_max]
        paste_region = donor_image[y_min:y_max, x_min:x_max]
        
        # Blend into target
        end_y = min(new_y + patch_h, h)
        end_x = min(new_x + patch_w, w)
        actual_h = end_y - new_y
        actual_w = end_x - new_x
        
        paste_mask_crop = paste_mask[:actual_h, :actual_w]
        paste_region_crop = paste_region[:actual_h, :actual_w]
        
        # Alpha blend
        alpha = paste_mask_crop[:, :, np.newaxis]
        image[new_y:end_y, new_x:end_x] = (
            alpha * paste_region_crop + 
            (1 - alpha) * image[new_y:end_y, new_x:end_x]
        ).astype(np.uint8)
        
        # Update mask
        mask[new_y:end_y, new_x:end_x] = np.maximum(
            mask[new_y:end_y, new_x:end_x],
            paste_mask_crop
        )
        
        return image, mask
    
    def apply_mosaic(self, idx):
        """Create mosaic from 4 random images."""
        indices = [idx] + random.sample(range(len(self.samples)), 3)
        
        # Target size for each quadrant
        quad_size = self.img_size // 2
        
        # Create mosaic
        mosaic_img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        mosaic_mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        
        positions = [(0, 0), (0, quad_size), (quad_size, 0), (quad_size, quad_size)]
        
        for i, pos in enumerate(positions):
            img, mask = self.load_sample(indices[i])
            
            # Resize to quadrant size
            img = cv2.resize(img, (quad_size, quad_size))
            mask = cv2.resize(mask, (quad_size, quad_size))
            
            y, x = pos
            mosaic_img[y:y+quad_size, x:x+quad_size] = img
            mosaic_mask[y:y+quad_size, x:x+quad_size] = mask
        
        return mosaic_img, mosaic_mask
    
    def __getitem__(self, idx):
        # Decide augmentation strategy
        if self.is_train:
            use_mosaic = random.random() < self.mosaic_prob
            use_copy_paste = random.random() < self.copy_paste_prob
        else:
            use_mosaic = False
            use_copy_paste = False
        
        if use_mosaic:
            image, mask = self.apply_mosaic(idx)
        else:
            image, mask = self.load_sample(idx)
            
            # Resize to target size
            image = cv2.resize(image, (self.img_size, self.img_size))
            mask = cv2.resize(mask, (self.img_size, self.img_size))
            
            # Apply copy-paste
            if use_copy_paste:
                donor_idx = random.randint(0, len(self.samples) - 1)
                image, mask = self.apply_copy_paste(image, mask, donor_idx)
        
        # Apply standard augmentations
        transformed = self.transform(image=image, mask=mask)
        image_tensor = transformed['image']
        mask = transformed['mask']
        
        if isinstance(mask, np.ndarray):
            mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
        else:
            mask_tensor = mask.unsqueeze(0).float()
        
        return image_tensor, mask_tensor


# ============================================================================
# TRAINING FUNCTIONS
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
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate(model, dataloader, device, threshold=0.3):
    model.eval()
    total_iou = 0
    total_dice = 0
    count = 0
    
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = torch.sigmoid(model(images))
            preds = (outputs > threshold).float()
            
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
    print("SMALL OBJECT AUGMENTATION TRAINING V5")
    print("=" * 60)
    print(f"Resolution: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Device: {DEVICE}")
    print()
    
    # Load missed forgeries
    missed_set = set()
    if MISSED_FORGERIES_FILE.exists():
        with open(MISSED_FORGERIES_FILE) as f:
            missed_set = set(l.strip() for l in f.readlines())
        print(f"Loaded {len(missed_set)} missed forgery names")
    else:
        print("Warning: missed_forgeries.txt not found")
    
    # Load forged images
    forged_dir = TRAIN_IMAGES_PATH / 'forged'
    forged_paths = sorted(list(forged_dir.glob('*.png')))
    mask_paths = [TRAIN_MASKS_PATH / f"{p.stem}.npy" for p in forged_paths]
    
    # Filter to existing masks
    valid = [(fp, mp) for fp, mp in zip(forged_paths, mask_paths) if mp.exists()]
    forged_paths, mask_paths = zip(*valid) if valid else ([], [])
    print(f"Forged images with masks: {len(forged_paths)}")
    
    # Split: use missed images as validation
    train_forged = []
    train_masks = []
    val_forged = []
    val_masks = []
    
    for fp, mp in zip(forged_paths, mask_paths):
        if fp.name in missed_set:
            val_forged.append(fp)
            val_masks.append(mp)
        train_forged.append(fp)
        train_masks.append(mp)
    
    print(f"Training: {len(train_forged)}, Validation (missed): {len(val_forged)}")
    
    # Create datasets
    train_dataset = SmallObjectDataset(
        train_forged, train_masks, missed_set,
        img_size=IMG_SIZE, is_train=True,
        copy_paste_prob=0.3, mosaic_prob=0.2
    )
    
    val_dataset = SmallObjectDataset(
        val_forged, val_masks, missed_set,
        img_size=IMG_SIZE, is_train=False
    )
    
    # Weighted sampler
    sampler = WeightedRandomSampler(
        weights=train_dataset.weights,
        num_samples=len(train_dataset),
        replacement=True
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        sampler=sampler, num_workers=4, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=4, pin_memory=True
    )
    
    # Create model
    model = create_model(in_channels=3)
    
    # Load base weights if available
    if BASE_MODEL.exists():
        print(f"Loading base model: {BASE_MODEL.name}")
        state_dict = torch.load(BASE_MODEL, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict, strict=False)
    else:
        print("Training from scratch (no base model)")
    
    model = model.to(DEVICE)
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR/100)
    
    # Loss
    criterion = SmallObjectLoss(focal_weight=0.4, dice_weight=0.4, boundary_weight=0.2)
    
    # Training loop
    best_iou = 0
    print("\nStarting training...")
    
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_iou, val_dice = validate(model, val_loader, DEVICE)
        scheduler.step()
        
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}/{EPOCHS} - Loss: {train_loss:.4f}, Val IoU: {val_iou:.4f}, Val Dice: {val_dice:.4f}, LR: {lr:.2e}")
        
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), OUTPUT_MODEL)
            print(f"  ✓ Saved new best model (IoU: {val_iou:.4f})")
    
    print()
    print("=" * 60)
    print(f"Training complete!")
    print(f"Best validation IoU: {best_iou:.4f}")
    print(f"Model saved to: {OUTPUT_MODEL}")
    print("=" * 60)


if __name__ == '__main__':
    main()
