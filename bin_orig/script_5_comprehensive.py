"""
Comprehensive Forgery Detection Training
=========================================

Phase 2 Implementation combining all improvements:
1. Higher resolution (512x512)
2. Focal loss with gamma=3.0
3. Small forgery oversampling (3x)
4. ELA (Error Level Analysis) as 4th input channel
5. Attention gates after decoder
6. Hard example mining (patch-based)

Target: 85-90% recall (up from 66%)
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
VAL_IMAGES_PATH = os.path.join(DATASET_PATH, "validation_images")
VAL_MASKS_PATH = os.path.join(DATASET_PATH, "validation_masks")
MODELS_PATH = os.path.join(DATASET_PATH, "models")

os.makedirs(MODELS_PATH, exist_ok=True)

# Training configuration
IMG_SIZE = 512  # Higher resolution for small forgeries
BATCH_SIZE = 4  # Reduced due to larger images and 4 channels
EPOCHS = 30
LEARNING_RATE = 0.0003
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Focal loss parameters
FOCAL_GAMMA = 3.0  # Higher gamma for hard examples
FOCAL_ALPHA = 0.75

# Small forgery oversampling
SMALL_FORGERY_THRESHOLD = 3.0  # % of image
OVERSAMPLE_FACTOR = 3

# ELA configuration
ELA_QUALITY = 90  # JPEG quality for ELA computation

# Hard example mining
HARD_EXAMPLE_RATIO = 0.3  # 30% of batch are hard patches
PATCH_SIZE = 128


# ============================================================================
# ERROR LEVEL ANALYSIS (ELA)
# ============================================================================

def compute_ela(image_bgr, quality=ELA_QUALITY):
    """
    Compute Error Level Analysis for an image.
    
    ELA highlights compression inconsistencies that occur when regions 
    are copy-pasted from different sources or manipulated.
    
    Args:
        image_bgr: Input image in BGR format (numpy array)
        quality: JPEG quality for recompression (default 90)
    
    Returns:
        ELA image as single-channel numpy array (0-255)
    """
    # Encode image as JPEG at specified quality
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded = cv2.imencode('.jpg', image_bgr, encode_param)
    
    # Decode back
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    
    # Compute absolute difference
    ela = cv2.absdiff(image_bgr, decoded)
    
    # Convert to grayscale and amplify
    ela_gray = cv2.cvtColor(ela, cv2.COLOR_BGR2GRAY)
    
    # Amplify differences (scale by 10x for visibility)
    ela_amplified = np.clip(ela_gray.astype(np.float32) * 10, 0, 255).astype(np.uint8)
    
    return ela_amplified


# ============================================================================
# ATTENTION GATE MODULE
# ============================================================================

class AttentionGate(nn.Module):
    """
    Attention gate that learns to focus on suspicious regions.
    
    The gate produces a spatial attention map that highlights
    areas likely to contain forgeries.
    """
    
    def __init__(self, in_channels):
        super(AttentionGate, self).__init__()
        # Use minimum of 8 channels to avoid zero-size tensors
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
    """
    FPN model with attention gate applied to decoder output.
    """
    
    def __init__(self, base_model, decoder_channels=256):
        super(AttentionFPN, self).__init__()
        self.base = base_model
        self.attention = AttentionGate(1)  # Applied to final 1-channel output
    
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
# DATASET WITH ELA AND HARD EXAMPLE MINING
# ============================================================================

class ComprehensiveDataset(Dataset):
    """
    Dataset with:
    - ELA as 4th input channel
    - Small forgery size tracking for oversampling
    - Hard example patch extraction
    """
    
    def __init__(self, image_paths, mask_paths, img_size=512, augment=True, 
                 include_patches=True, patch_size=128):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.img_size = img_size
        self.augment = augment
        self.include_patches = include_patches
        self.patch_size = patch_size
        
        # Analyze forgery sizes for oversampling
        self.forgery_sizes = []
        self.forgery_centers = []  # For hard example mining
        
        print("Analyzing forgery sizes and locations...")
        for i, mask_path in enumerate(mask_paths):
            try:
                mask = np.load(mask_path)
                if mask.ndim == 3:
                    mask = np.max(mask, axis=0)
                
                forgery_pct = (mask > 0.5).sum() / mask.size * 100
                self.forgery_sizes.append(forgery_pct)
                
                # Find forgery centroid for hard example mining
                if forgery_pct > 0:
                    y_coords, x_coords = np.where(mask > 0.5)
                    if len(y_coords) > 0:
                        cy = int(y_coords.mean())
                        cx = int(x_coords.mean())
                        self.forgery_centers.append((cy, cx, mask.shape))
                    else:
                        self.forgery_centers.append(None)
                else:
                    self.forgery_centers.append(None)
            except:
                self.forgery_sizes.append(0)
                self.forgery_centers.append(None)
        
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
        print(f"  Images with forgery centers: {sum(1 for c in self.forgery_centers if c is not None)}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = cv2.imread(self.image_paths[idx])
        if image is None:
            image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        
        # Compute ELA before any resizing (on original image)
        ela = compute_ela(image)
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image and ELA
        image = cv2.resize(image, (self.img_size, self.img_size))
        ela = cv2.resize(ela, (self.img_size, self.img_size))
        
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
            # Horizontal flip
            if np.random.random() > 0.5:
                image = cv2.flip(image, 1)
                ela = cv2.flip(ela, 1)
                mask = cv2.flip(mask, 1)
            
            # Vertical flip
            if np.random.random() > 0.5:
                image = cv2.flip(image, 0)
                ela = cv2.flip(ela, 0)
                mask = cv2.flip(mask, 0)
            
            # Rotation
            if np.random.random() > 0.5:
                k = np.random.randint(1, 4)
                image = np.rot90(image, k)
                ela = np.rot90(ela, k)
                mask = np.rot90(mask, k)
            
            # Color jitter (only on RGB, not ELA)
            if np.random.random() > 0.5:
                image = image.astype(np.float32)
                image *= np.random.uniform(0.8, 1.2)
                image = np.clip(image, 0, 255).astype(np.uint8)
            
            # Make contiguous after augmentation
            image = np.ascontiguousarray(image)
            ela = np.ascontiguousarray(ela)
            mask = np.ascontiguousarray(mask)
        
        # Normalize RGB channels (ImageNet stats)
        image = image.astype(np.float32) / 255.0
        image[..., 0] = (image[..., 0] - 0.485) / 0.229
        image[..., 1] = (image[..., 1] - 0.456) / 0.224
        image[..., 2] = (image[..., 2] - 0.406) / 0.225
        
        # Normalize ELA channel (0-1 range)
        ela = ela.astype(np.float32) / 255.0
        
        # Combine RGB + ELA into 4-channel input
        image_4ch = np.zeros((self.img_size, self.img_size, 4), dtype=np.float32)
        image_4ch[..., :3] = image
        image_4ch[..., 3] = ela
        
        # To tensor [C, H, W]
        image_tensor = torch.from_numpy(image_4ch.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)
        
        return image_tensor, mask_tensor


class HardExampleDataset(Dataset):
    """
    Dataset that extracts patches centered on forgery regions.
    Used for hard example mining during training.
    """
    
    def __init__(self, image_paths, mask_paths, patch_size=128, num_patches=500):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.patch_size = patch_size
        
        # Pre-compute patch locations
        self.patches = []
        print(f"Extracting {num_patches} hard example patches...")
        
        for i, (img_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
            try:
                mask = np.load(mask_path)
                if mask.ndim == 3:
                    mask = np.max(mask, axis=0)
                
                if (mask > 0.5).any():
                    y_coords, x_coords = np.where(mask > 0.5)
                    if len(y_coords) > 0:
                        # Sample random points from forgery region
                        for _ in range(min(3, len(y_coords))):
                            rand_idx = np.random.randint(len(y_coords))
                            cy, cx = y_coords[rand_idx], x_coords[rand_idx]
                            self.patches.append({
                                'image_path': img_path,
                                'mask_path': mask_path,
                                'center': (cy, cx),
                                'mask_shape': mask.shape
                            })
                            if len(self.patches) >= num_patches:
                                break
            except:
                continue
            
            if len(self.patches) >= num_patches:
                break
        
        print(f"  Collected {len(self.patches)} hard example patches")
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        patch_info = self.patches[idx]
        
        # Load image and mask
        image = cv2.imread(patch_info['image_path'])
        if image is None:
            return self._get_dummy()
        
        # Compute ELA
        ela = compute_ela(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        try:
            mask = np.load(patch_info['mask_path'])
            if mask.ndim == 3:
                mask = np.max(mask, axis=0)
        except:
            return self._get_dummy()
        
        # Extract patch around forgery center
        cy, cx = patch_info['center']
        h, w = image.shape[:2]
        half = self.patch_size // 2
        
        # Compute patch bounds with boundary handling
        y1 = max(0, cy - half)
        y2 = min(h, cy + half)
        x1 = max(0, cx - half)
        x2 = min(w, cx + half)
        
        # Extract patches
        img_patch = image[y1:y2, x1:x2]
        ela_patch = ela[y1:y2, x1:x2]
        mask_patch = mask[y1:y2, x1:x2]
        
        # Resize to consistent size
        img_patch = cv2.resize(img_patch, (self.patch_size, self.patch_size))
        ela_patch = cv2.resize(ela_patch, (self.patch_size, self.patch_size))
        mask_patch = cv2.resize(mask_patch.astype(np.float32), (self.patch_size, self.patch_size))
        
        # Normalize
        img_patch = img_patch.astype(np.float32) / 255.0
        img_patch[..., 0] = (img_patch[..., 0] - 0.485) / 0.229
        img_patch[..., 1] = (img_patch[..., 1] - 0.456) / 0.224
        img_patch[..., 2] = (img_patch[..., 2] - 0.406) / 0.225
        ela_patch = ela_patch.astype(np.float32) / 255.0
        
        # Combine channels
        patch_4ch = np.zeros((self.patch_size, self.patch_size, 4), dtype=np.float32)
        patch_4ch[..., :3] = img_patch
        patch_4ch[..., 3] = ela_patch
        
        image_tensor = torch.from_numpy(patch_4ch.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(mask_patch).float().unsqueeze(0)
        
        return image_tensor, mask_tensor
    
    def _get_dummy(self):
        """Return dummy tensors for failed loads"""
        image_tensor = torch.zeros(4, self.patch_size, self.patch_size)
        mask_tensor = torch.zeros(1, self.patch_size, self.patch_size)
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
    print("COMPREHENSIVE FORGERY DETECTION TRAINING")
    print("="*80)
    print(f"Device: {DEVICE}")
    print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Input channels: 4 (RGB + ELA)")
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
    train_dataset = ComprehensiveDataset(
        train_images, train_masks, 
        img_size=IMG_SIZE, 
        augment=True
    )
    val_dataset = ComprehensiveDataset(
        val_images, val_masks, 
        img_size=IMG_SIZE, 
        augment=False
    )
    
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
    
    # Create model with 4 input channels (RGB + ELA)
    print("\nCreating model...")
    base_model = smp.FPN(
        encoder_name='timm-efficientnet-b2',
        encoder_weights='imagenet',
        in_channels=4,  # RGB + ELA
        classes=1,
    )
    
    # Wrap with attention gate
    model = AttentionFPN(base_model).to(DEVICE)
    
    print(f"Model created with attention gate")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss with higher focal gamma
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
            save_path = os.path.join(MODELS_PATH, 'comprehensive_best.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_mcc': best_val_mcc,
                'val_f1': val_metrics['f1'],
                'val_recall': val_metrics['recall'],
                'config': {
                    'img_size': IMG_SIZE,
                    'focal_gamma': FOCAL_GAMMA,
                    'in_channels': 4,
                    'has_ela': True,
                    'has_attention': True,
                }
            }, save_path)
            print(f"  ✓ Saved best model (MCC: {best_val_mcc:.4f})")
    
    # Save final model
    final_path = os.path.join(MODELS_PATH, 'comprehensive_final.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': EPOCHS,
        'val_mcc': val_metrics['mcc'],
        'config': {
            'img_size': IMG_SIZE,
            'focal_gamma': FOCAL_GAMMA,
            'in_channels': 4,
            'has_ela': True,
            'has_attention': True,
        }
    }, final_path)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Best Val MCC: {best_val_mcc:.4f}")
    print(f"Best model saved to: {os.path.join(MODELS_PATH, 'comprehensive_best.pth')}")
    print(f"Final model saved to: {final_path}")


if __name__ == "__main__":
    main()
