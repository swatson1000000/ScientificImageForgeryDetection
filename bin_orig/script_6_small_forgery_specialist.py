"""
Small Forgery Specialist Model
==============================

Specialized model trained exclusively on small forgeries (<2% of image area).
Key strategies:
1. Train ONLY on tiny (<0.5%) and small (0.5-2%) forgeries
2. Higher resolution inference with crops
3. More aggressive augmentation for small regions
4. Very high focal loss gamma to focus on hard examples
5. Lower min-area threshold during training
6. Heavy oversampling of tiny forgeries

This model is designed to be used in ensemble to catch small forgeries
that the main models miss.
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
import random

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
IMG_SIZE = 512
BATCH_SIZE = 4
EPOCHS = 40  # More epochs for small dataset
LEARNING_RATE = 0.0001  # Lower LR for fine-grained learning
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Small forgery thresholds
TINY_THRESHOLD = 0.5   # % of image
SMALL_THRESHOLD = 2.0  # % of image - only train on forgeries smaller than this

# Focal loss - very high gamma for small objects
FOCAL_GAMMA = 4.0
FOCAL_ALPHA = 0.85  # Higher weight for positive class

# Oversampling - heavily oversample tiny forgeries
TINY_OVERSAMPLE = 5
SMALL_OVERSAMPLE = 2


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
# FOCAL LOSS - HIGH GAMMA FOR SMALL OBJECTS
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss with very high gamma to focus on hard-to-detect small forgeries"""
    
    def __init__(self, alpha=0.85, gamma=4.0, reduction='mean'):
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
    """Combined Focal + Dice Loss - more focal weight for small forgeries"""
    
    def __init__(self, focal_weight=0.8, dice_weight=0.2, gamma=4.0, alpha=0.85):
        super(CombinedLoss, self).__init__()
        self.focal = FocalLoss(gamma=gamma, alpha=alpha)
        self.dice = DiceLoss()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
    
    def forward(self, inputs, targets):
        return self.focal_weight * self.focal(inputs, targets) + \
               self.dice_weight * self.dice(inputs, targets)


# ============================================================================
# DATASET - SMALL FORGERIES ONLY WITH AGGRESSIVE AUGMENTATION
# ============================================================================

class SmallForgeryDataset(Dataset):
    """
    Dataset focused on small forgeries with aggressive augmentation.
    Uses crop-around-forgery to zoom in on small regions.
    """
    
    def __init__(self, image_paths, mask_paths, forgery_sizes, img_size=512, augment=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.forgery_sizes = forgery_sizes
        self.img_size = img_size
        self.augment = augment
        
        # Calculate sample weights
        self.weights = []
        for size in self.forgery_sizes:
            if size < TINY_THRESHOLD:
                self.weights.append(TINY_OVERSAMPLE)
            else:
                self.weights.append(SMALL_OVERSAMPLE)
    
    def __len__(self):
        return len(self.image_paths)
    
    def crop_around_forgery(self, image, mask):
        """
        Crop around the forged region to get better resolution.
        Returns a crop that contains the forgery but zooms in.
        """
        # Find forgery location
        y_indices, x_indices = np.where(mask > 0.5)
        
        if len(y_indices) == 0:
            return image, mask
        
        # Get bounding box
        y_min, y_max = y_indices.min(), y_indices.max()
        x_min, x_max = x_indices.min(), x_indices.max()
        
        # Add padding (random amount for augmentation)
        h, w = mask.shape
        forgery_h = y_max - y_min
        forgery_w = x_max - x_min
        
        # Pad by 2-4x the forgery size
        pad_factor = random.uniform(1.5, 3.0) if self.augment else 2.0
        pad_h = int(forgery_h * pad_factor)
        pad_w = int(forgery_w * pad_factor)
        
        # Ensure minimum crop size
        min_crop = min(h, w) // 2
        pad_h = max(pad_h, min_crop)
        pad_w = max(pad_w, min_crop)
        
        # Calculate crop bounds with random offset for augmentation
        if self.augment:
            offset_y = random.randint(-pad_h//4, pad_h//4)
            offset_x = random.randint(-pad_w//4, pad_w//4)
        else:
            offset_y, offset_x = 0, 0
        
        center_y = (y_min + y_max) // 2 + offset_y
        center_x = (x_min + x_max) // 2 + offset_x
        
        crop_y1 = max(0, center_y - pad_h)
        crop_y2 = min(h, center_y + pad_h)
        crop_x1 = max(0, center_x - pad_w)
        crop_x2 = min(w, center_x + pad_w)
        
        # Ensure we have a valid crop
        if crop_y2 <= crop_y1 or crop_x2 <= crop_x1:
            return image, mask
        
        cropped_img = image[crop_y1:crop_y2, crop_x1:crop_x2]
        cropped_mask = mask[crop_y1:crop_y2, crop_x1:crop_x2]
        
        return cropped_img, cropped_mask
    
    def __getitem__(self, idx):
        # Load image
        image = cv2.imread(self.image_paths[idx])
        if image is None:
            image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        try:
            mask = np.load(self.mask_paths[idx])
            if mask.ndim == 3:
                mask = np.max(mask, axis=0)
            mask = mask.astype(np.float32)
        except:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        
        # Crop around forgery (50% chance in training)
        if self.augment and random.random() > 0.5 and self.forgery_sizes[idx] < TINY_THRESHOLD:
            # Always crop for tiny forgeries
            image, mask = self.crop_around_forgery(image, mask)
        elif self.augment and random.random() > 0.7:
            # Sometimes crop for small forgeries
            image, mask = self.crop_around_forgery(image, mask)
        
        # Resize to training size
        image = cv2.resize(image, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size))
        
        # Aggressive augmentation
        if self.augment:
            # Flips
            if random.random() > 0.5:
                image = cv2.flip(image, 1)
                mask = cv2.flip(mask, 1)
            if random.random() > 0.5:
                image = cv2.flip(image, 0)
                mask = cv2.flip(mask, 0)
            
            # Rotation
            if random.random() > 0.5:
                k = random.randint(1, 3)
                image = np.rot90(image, k)
                mask = np.rot90(mask, k)
            
            # Brightness/contrast
            if random.random() > 0.5:
                image = image.astype(np.float32)
                alpha = random.uniform(0.7, 1.3)  # Contrast
                beta = random.uniform(-20, 20)    # Brightness
                image = alpha * image + beta
                image = np.clip(image, 0, 255).astype(np.uint8)
            
            # Color jitter
            if random.random() > 0.7:
                hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
                hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-10, 10)) % 180
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.8, 1.2), 0, 255)
                hsv[:, :, 2] = np.clip(hsv[:, :, 2] * random.uniform(0.8, 1.2), 0, 255)
                image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            
            # Gaussian blur (slight)
            if random.random() > 0.8:
                ksize = random.choice([3, 5])
                image = cv2.GaussianBlur(image, (ksize, ksize), 0)
            
            # JPEG compression artifact simulation
            if random.random() > 0.7:
                quality = random.randint(70, 95)
                _, encoded = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 
                                         [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                image = cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            
            # Make contiguous
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
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
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
    print("SMALL FORGERY SPECIALIST MODEL")
    print("="*80)
    print(f"Device: {DEVICE}")
    print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Only training on forgeries < {SMALL_THRESHOLD}% of image")
    print(f"Focal gamma: {FOCAL_GAMMA}")
    print(f"Focal alpha: {FOCAL_ALPHA}")
    print(f"Tiny oversample: {TINY_OVERSAMPLE}x, Small oversample: {SMALL_OVERSAMPLE}x")
    print(f"Epochs: {EPOCHS}")
    print()
    
    # Collect training data - ONLY SMALL FORGERIES
    print("Loading training data (small forgeries only)...")
    forged_dir = os.path.join(TRAIN_IMAGES_PATH, 'forged')
    mask_dir = TRAIN_MASKS_PATH
    
    image_paths = []
    mask_paths = []
    forgery_sizes = []
    
    tiny_count = 0
    small_count = 0
    
    for f in sorted(os.listdir(forged_dir)):
        if f.endswith('.png'):
            img_path = os.path.join(forged_dir, f)
            mask_path = os.path.join(mask_dir, f.replace('.png', '.npy'))
            
            if os.path.exists(mask_path):
                # Calculate forgery size
                mask = np.load(mask_path)
                if mask.ndim == 3:
                    mask = np.max(mask, axis=0)
                forgery_pct = (mask > 0.5).sum() / mask.size * 100
                
                # Only include small forgeries
                if forgery_pct < SMALL_THRESHOLD and forgery_pct > 0:
                    image_paths.append(img_path)
                    mask_paths.append(mask_path)
                    forgery_sizes.append(forgery_pct)
                    
                    if forgery_pct < TINY_THRESHOLD:
                        tiny_count += 1
                    else:
                        small_count += 1
    
    print(f"Found {len(image_paths)} small forgery images")
    print(f"  Tiny (<{TINY_THRESHOLD}%): {tiny_count}")
    print(f"  Small ({TINY_THRESHOLD}-{SMALL_THRESHOLD}%): {small_count}")
    
    if len(image_paths) < 100:
        print("ERROR: Not enough small forgery samples!")
        return
    
    # Shuffle and split
    combined = list(zip(image_paths, mask_paths, forgery_sizes))
    random.seed(42)
    random.shuffle(combined)
    image_paths, mask_paths, forgery_sizes = zip(*combined)
    image_paths = list(image_paths)
    mask_paths = list(mask_paths)
    forgery_sizes = list(forgery_sizes)
    
    # Split train/val (90/10)
    split_idx = int(len(image_paths) * 0.9)
    train_images = image_paths[:split_idx]
    train_masks = mask_paths[:split_idx]
    train_sizes = forgery_sizes[:split_idx]
    val_images = image_paths[split_idx:]
    val_masks = mask_paths[split_idx:]
    val_sizes = forgery_sizes[split_idx:]
    
    print(f"Train: {len(train_images)}, Val: {len(val_images)}")
    
    # Create datasets
    train_dataset = SmallForgeryDataset(
        train_images, train_masks, train_sizes, 
        img_size=IMG_SIZE, augment=True
    )
    val_dataset = SmallForgeryDataset(
        val_images, val_masks, val_sizes,
        img_size=IMG_SIZE, augment=False
    )
    
    # Weighted sampler for oversampling tiny forgeries
    sampler = WeightedRandomSampler(
        weights=train_dataset.weights,
        num_samples=len(train_dataset) * 2,  # 2x effective epoch size
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
    
    # Create model (3 input channels - RGB only for this specialist)
    print("\nCreating model...")
    base_model = smp.FPN(
        encoder_name='timm-efficientnet-b2',
        encoder_weights='imagenet',
        in_channels=3,
        classes=1,
    )
    
    # Wrap with attention gate
    model = AttentionFPN(base_model).to(DEVICE)
    
    print(f"Model created with attention gate")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss - higher focal weight for small objects
    criterion = CombinedLoss(
        focal_weight=0.8, 
        dice_weight=0.2, 
        gamma=FOCAL_GAMMA,
        alpha=FOCAL_ALPHA
    )
    
    # Optimizer with lower learning rate
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # Training loop
    print("\nStarting training...")
    best_val_recall = 0.0
    patience = 10
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_metrics = validate(model, val_loader, criterion, DEVICE)
        
        scheduler.step()
        
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{EPOCHS} ({epoch_time:.1f}s) - "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Val Recall: {val_metrics['recall']:.4f}, Val F1: {val_metrics['f1']:.4f}, "
              f"Val IoU: {val_metrics['iou']:.4f}")
        
        # Save best model based on RECALL (most important for small forgeries)
        if val_metrics['recall'] > best_val_recall:
            best_val_recall = val_metrics['recall']
            patience_counter = 0
            save_path = os.path.join(MODELS_PATH, 'small_forgery_specialist_best.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_recall': best_val_recall,
                'val_f1': val_metrics['f1'],
                'val_iou': val_metrics['iou'],
                'config': {
                    'img_size': IMG_SIZE,
                    'focal_gamma': FOCAL_GAMMA,
                    'focal_alpha': FOCAL_ALPHA,
                    'in_channels': 3,
                    'small_threshold': SMALL_THRESHOLD,
                    'tiny_threshold': TINY_THRESHOLD,
                }
            }, save_path)
            print(f"  ✓ Saved best model (Recall: {best_val_recall:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping after {patience} epochs without improvement")
                break
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Best Val Recall: {best_val_recall:.4f}")
    print(f"Model saved to: {os.path.join(MODELS_PATH, 'small_forgery_specialist_best.pth')}")


if __name__ == "__main__":
    main()
