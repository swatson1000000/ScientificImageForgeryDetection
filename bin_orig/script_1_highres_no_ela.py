"""
High Resolution Training (No ELA)
=================================

Training with improvements but WITHOUT ELA channel:
1. Higher resolution (512x512)
2. Focal loss with gamma=3.0
3. Small forgery oversampling (3x)
4. Attention gates after decoder
5. NO ELA (standard 3-channel RGB)

This tests if ELA was causing the overfitting issue.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn.functional as F
import cv2
import math
from datetime import datetime
import time

# Import segmentation_models_pytorch
try:
    import segmentation_models_pytorch as smp
except ImportError:
    os.system("pip install segmentation-models-pytorch -q")
    import segmentation_models_pytorch as smp

# ============================================================================
# CONFIGURATION
# ============================================================================

DATASET_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_IMAGES_PATH = os.path.join(DATASET_PATH, "train_images")
TRAIN_MASKS_PATH = os.path.join(DATASET_PATH, "train_masks")
MODELS_PATH = os.path.join(DATASET_PATH, "models")

os.makedirs(MODELS_PATH, exist_ok=True)

# Training configuration
IMG_SIZE = 512  # Higher resolution for small forgeries
BATCH_SIZE = 4
EPOCHS = 30
LEARNING_RATE = 0.0003
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Focal loss parameters
FOCAL_GAMMA = 3.0
FOCAL_ALPHA = 0.75

# Small forgery oversampling
SMALL_FORGERY_THRESHOLD = 3.0  # % of image
OVERSAMPLE_FACTOR = 3


# ============================================================================
# ATTENTION GATE MODULE
# ============================================================================

class AttentionGate(nn.Module):
    """Attention gate that learns to focus on suspicious regions."""
    
    def __init__(self, in_channels):
        super(AttentionGate, self).__init__()
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
        attention = self.conv(x)
        return x * attention


class AttentionFPN(nn.Module):
    """FPN model with attention gate applied to decoder output."""
    
    def __init__(self, base_model):
        super(AttentionFPN, self).__init__()
        self.base = base_model
        self.attention = AttentionGate(1)
    
    def forward(self, x):
        out = self.base(x)
        out = self.attention(out)
        return out


# ============================================================================
# FOCAL LOSS
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss with higher gamma to focus on hard-to-detect small forgeries"""
    
    def __init__(self, alpha=0.75, gamma=3.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        p = torch.sigmoid(inputs)
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * focal_weight * bce
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""
    
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice


class CombinedLoss(nn.Module):
    """Combined Focal + Dice Loss"""
    
    def __init__(self, focal_weight=0.7, dice_weight=0.3, gamma=3.0):
        super(CombinedLoss, self).__init__()
        self.focal = FocalLoss(gamma=gamma)
        self.dice = DiceLoss()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
    
    def forward(self, inputs, targets):
        return self.focal_weight * self.focal(inputs, targets) + \
               self.dice_weight * self.dice(inputs, targets)


# ============================================================================
# DATASET (NO ELA - Standard 3-channel RGB)
# ============================================================================

class HighResDataset(Dataset):
    """Dataset with higher resolution and oversampling (NO ELA)"""
    
    def __init__(self, image_paths, mask_paths, img_size=512, augment=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.img_size = img_size
        self.augment = augment
        
        # Analyze forgery sizes for oversampling
        self.forgery_sizes = []
        print("Analyzing forgery sizes...")
        for mask_path in mask_paths:
            try:
                mask = np.load(mask_path)
                if mask.ndim == 3:
                    mask = np.max(mask, axis=0)
                forgery_pct = (mask > 0.5).sum() / mask.size * 100
                self.forgery_sizes.append(forgery_pct)
            except:
                self.forgery_sizes.append(0)
        
        # Calculate sample weights (oversample small forgeries)
        self.weights = []
        small_count = 0
        for size in self.forgery_sizes:
            if size < SMALL_FORGERY_THRESHOLD and size > 0:
                self.weights.append(OVERSAMPLE_FACTOR)
                small_count += 1
            else:
                self.weights.append(1.0)
        
        print(f"  Total samples: {len(self.forgery_sizes)}")
        print(f"  Small forgeries (<{SMALL_FORGERY_THRESHOLD}%): {small_count}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = cv2.imread(self.image_paths[idx])
        if image is None:
            image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.img_size, self.img_size))
        
        # Load mask
        try:
            mask = np.load(self.mask_paths[idx])
            if mask.ndim == 3:
                mask = np.max(mask, axis=0)
            mask = cv2.resize(mask.astype(np.float32), (self.img_size, self.img_size))
        except:
            mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        
        # Augmentation
        if self.augment:
            if np.random.random() > 0.5:
                image = cv2.flip(image, 1)
                mask = cv2.flip(mask, 1)
            if np.random.random() > 0.5:
                image = cv2.flip(image, 0)
                mask = cv2.flip(mask, 0)
            if np.random.random() > 0.5:
                k = np.random.randint(1, 4)
                image = np.rot90(image, k)
                mask = np.rot90(mask, k)
            if np.random.random() > 0.5:
                image = image.astype(np.float32)
                image *= np.random.uniform(0.8, 1.2)
                image = np.clip(image, 0, 255).astype(np.uint8)
            
            # Make contiguous after augmentation
            image = np.ascontiguousarray(image)
            mask = np.ascontiguousarray(mask)
        
        # Normalize RGB channels (ImageNet stats)
        image = image.astype(np.float32) / 255.0
        image[..., 0] = (image[..., 0] - 0.485) / 0.229
        image[..., 1] = (image[..., 1] - 0.456) / 0.224
        image[..., 2] = (image[..., 2] - 0.406) / 0.225
        
        # To tensor [C, H, W]
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)
        
        return image_tensor, mask_tensor


# ============================================================================
# METRICS
# ============================================================================

def calculate_metrics(pred, target, threshold=0.5):
    """Calculate segmentation metrics"""
    pred_binary = (pred > threshold).float()
    target_binary = (target > 0.5).float()
    
    tp = (pred_binary * target_binary).sum().item()
    fp = (pred_binary * (1 - target_binary)).sum().item()
    fn = ((1 - pred_binary) * target_binary).sum().item()
    tn = ((1 - pred_binary) * (1 - target_binary)).sum().item()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    
    mcc_num = tp * tn - fp * fn
    mcc_den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + 1e-8)
    mcc = mcc_num / mcc_den
    
    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'iou': iou,
        'mcc': mcc
    }


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch_idx, (images, masks) in enumerate(dataloader):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_metrics = {'f1': [], 'precision': [], 'recall': [], 'iou': [], 'mcc': []}
    
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            
            preds = torch.sigmoid(outputs)
            for i in range(preds.shape[0]):
                metrics = calculate_metrics(preds[i], masks[i])
                for k, v in metrics.items():
                    all_metrics[k].append(v)
    
    avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
    return total_loss / len(dataloader), avg_metrics


def main():
    print("="*80)
    print("HIGH RESOLUTION TRAINING (NO ELA)")
    print("="*80)
    print(f"Device: {DEVICE}")
    print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Input channels: 3 (RGB only - NO ELA)")
    print(f"Focal gamma: {FOCAL_GAMMA}")
    print(f"Small forgery threshold: {SMALL_FORGERY_THRESHOLD}%")
    print(f"Oversample factor: {OVERSAMPLE_FACTOR}x")
    print(f"Epochs: {EPOCHS}")
    print()
    
    # Collect training data
    print("Loading training data...")
    forged_dir = os.path.join(TRAIN_IMAGES_PATH, 'forged')
    mask_dir = TRAIN_MASKS_PATH
    
    image_paths = []
    mask_paths = []
    
    for f in sorted(os.listdir(forged_dir)):
        if f.endswith('.png'):
            img_path = os.path.join(forged_dir, f)
            mask_path = os.path.join(mask_dir, f.replace('.png', '.npy'))
            if os.path.exists(mask_path):
                image_paths.append(img_path)
                mask_paths.append(mask_path)
    
    print(f"Found {len(image_paths)} training images with masks")
    
    # Split train/val
    split_idx = int(len(image_paths) * 0.9)
    train_images = image_paths[:split_idx]
    train_masks = mask_paths[:split_idx]
    val_images = image_paths[split_idx:]
    val_masks = mask_paths[split_idx:]
    
    print(f"Train: {len(train_images)}, Val: {len(val_images)}")
    
    # Create datasets
    train_dataset = HighResDataset(train_images, train_masks, img_size=IMG_SIZE, augment=True)
    val_dataset = HighResDataset(val_images, val_masks, img_size=IMG_SIZE, augment=False)
    
    # Weighted sampler for oversampling small forgeries
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
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    # Create model with 3 input channels (RGB only)
    print("\nCreating model...")
    base_model = smp.FPN(
        encoder_name='timm-efficientnet-b2',
        encoder_weights='imagenet',
        in_channels=3,  # RGB only
        classes=1,
    )
    
    # Wrap with attention gate
    model = AttentionFPN(base_model).to(DEVICE)
    
    print(f"Model created with attention gate")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss
    criterion = CombinedLoss(focal_weight=0.7, dice_weight=0.3, gamma=FOCAL_GAMMA)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # Training loop
    print("\nStarting training...")
    best_val_mcc = -1.0
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_metrics = validate(model, val_loader, criterion, DEVICE)
        
        scheduler.step()
        
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{EPOCHS} ({epoch_time:.1f}s) - "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Val F1: {val_metrics['f1']:.4f}, Val MCC: {val_metrics['mcc']:.4f}, "
              f"Val Recall: {val_metrics['recall']:.4f}")
        
        # Save best model based on MCC
        if val_metrics['mcc'] > best_val_mcc:
            best_val_mcc = val_metrics['mcc']
            save_path = os.path.join(MODELS_PATH, 'highres_no_ela_best.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_mcc': best_val_mcc,
                'val_f1': val_metrics['f1'],
                'val_recall': val_metrics['recall'],
                'config': {
                    'img_size': IMG_SIZE,
                    'focal_gamma': FOCAL_GAMMA,
                    'in_channels': 3,
                    'has_ela': False,
                    'has_attention': True,
                }
            }, save_path)
            print(f"  ✓ Saved best model (MCC: {best_val_mcc:.4f})")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Best Val MCC: {best_val_mcc:.4f}")
    print(f"Model saved to: {os.path.join(MODELS_PATH, 'highres_no_ela_best.pth')}")


if __name__ == "__main__":
    main()
