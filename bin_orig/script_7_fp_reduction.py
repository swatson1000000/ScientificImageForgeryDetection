"""
False Positive Reduction Model
==============================

Trains a model specifically designed to reduce false positives by:
1. Using ALL authentic data (12,208 images) as hard negatives
2. Weighted loss that penalizes false positives 3x more
3. Learning to distinguish authentic textures from forgery artifacts
4. Output used as a VETO model (subtracts from ensemble predictions)

The model learns: "This region looks AUTHENTIC, do not flag it"
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.nn.functional as F
import cv2
import math
from datetime import datetime
import time
from pathlib import Path

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
IMG_SIZE = 512
BATCH_SIZE = 4
EPOCHS = 20  # Fewer epochs since we have more data
LEARNING_RATE = 0.0002
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# False positive reduction parameters
# v1: FP_WEIGHT=3.0 resulted in 99.82% specificity but only 7% recall (too aggressive)
# v2: FP_WEIGHT=1.5 to balance better between FP reduction and forgery detection
FP_WEIGHT = 1.5  # Penalize false positives 1.5x more than false negatives
AUTHENTIC_RATIO = 0.6  # 60% authentic, 40% forged per batch


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
# FP-WEIGHTED LOSS
# ============================================================================

class FPWeightedFocalLoss(nn.Module):
    """
    Focal Loss with extra penalty for false positives.
    
    This makes the model MORE CONSERVATIVE - it will prefer to miss
    some real forgeries rather than flag authentic regions.
    """
    
    def __init__(self, alpha=0.75, gamma=2.0, fp_weight=3.0, reduction='mean'):
        super(FPWeightedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.fp_weight = fp_weight  # Extra penalty for FP
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        p = torch.sigmoid(inputs)
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Extra weight for false positives (predicting 1 when target is 0)
        # FP occurs when: prediction > 0.5 AND target == 0
        fp_mask = ((p > 0.5) & (targets < 0.5)).float()
        fp_penalty = 1.0 + (self.fp_weight - 1.0) * fp_mask
        
        focal_loss = alpha_t * focal_weight * bce * fp_penalty
        
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


class FPWeightedCombinedLoss(nn.Module):
    """Combined FP-Weighted Focal + Dice Loss"""
    
    def __init__(self, focal_weight=0.7, dice_weight=0.3, fp_weight=3.0):
        super(FPWeightedCombinedLoss, self).__init__()
        self.focal = FPWeightedFocalLoss(fp_weight=fp_weight)
        self.dice = DiceLoss()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
    
    def forward(self, inputs, targets):
        return self.focal_weight * self.focal(inputs, targets) + \
               self.dice_weight * self.dice(inputs, targets)


# ============================================================================
# DATASET
# ============================================================================

class AuthenticDataset(Dataset):
    """Dataset for authentic images (all have empty masks)"""
    
    def __init__(self, image_paths, img_size=512, augment=True):
        self.image_paths = image_paths
        self.img_size = img_size
        self.augment = augment
    
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
        
        # Empty mask (authentic = no forgery)
        mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        
        # Augmentation
        if self.augment:
            if np.random.random() > 0.5:
                image = cv2.flip(image, 1)
            if np.random.random() > 0.5:
                image = cv2.flip(image, 0)
            if np.random.random() > 0.5:
                k = np.random.randint(1, 4)
                image = np.rot90(image, k)
            if np.random.random() > 0.5:
                image = image.astype(np.float32)
                image *= np.random.uniform(0.8, 1.2)
                image = np.clip(image, 0, 255).astype(np.uint8)
            
            image = np.ascontiguousarray(image)
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        image[..., 0] = (image[..., 0] - 0.485) / 0.229
        image[..., 1] = (image[..., 1] - 0.456) / 0.224
        image[..., 2] = (image[..., 2] - 0.406) / 0.225
        
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)
        
        return image_tensor, mask_tensor


class ForgedDataset(Dataset):
    """Dataset for forged images with masks"""
    
    def __init__(self, image_paths, mask_paths, img_size=512, augment=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.img_size = img_size
        self.augment = augment
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = cv2.imread(self.image_paths[idx])
        if image is None:
            image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        
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
            
            image = np.ascontiguousarray(image)
            mask = np.ascontiguousarray(mask)
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        image[..., 0] = (image[..., 0] - 0.485) / 0.229
        image[..., 1] = (image[..., 1] - 0.456) / 0.224
        image[..., 2] = (image[..., 2] - 0.406) / 0.225
        
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)
        
        return image_tensor, mask_tensor


# ============================================================================
# METRICS
# ============================================================================

def calculate_metrics(pred, target, threshold=0.5):
    """Calculate segmentation metrics with FP rate"""
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
    
    # False positive rate (what we want to minimize!)
    fpr = fp / (fp + tn + 1e-8)
    
    # Specificity (true negative rate) - what we want to maximize
    specificity = tn / (tn + fp + 1e-8)
    
    mcc_num = tp * tn - fp * fn
    mcc_den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + 1e-8)
    mcc = mcc_num / mcc_den
    
    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'iou': iou,
        'mcc': mcc,
        'fpr': fpr,
        'specificity': specificity,
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
    all_metrics = {'f1': [], 'precision': [], 'recall': [], 'iou': [], 
                   'mcc': [], 'fpr': [], 'specificity': []}
    
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
    print("FALSE POSITIVE REDUCTION MODEL")
    print("="*80)
    print(f"Device: {DEVICE}")
    print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"FP penalty weight: {FP_WEIGHT}x")
    print(f"Authentic ratio: {AUTHENTIC_RATIO:.0%}")
    print(f"Epochs: {EPOCHS}")
    print()
    
    # Collect ALL authentic images
    print("Loading authentic images...")
    authentic_dirs = [
        os.path.join(TRAIN_IMAGES_PATH, 'authentic'),
        os.path.join(TRAIN_IMAGES_PATH, 'authentic_augmented'),
        os.path.join(TRAIN_IMAGES_PATH, 'authentic_augmented_v2'),
    ]
    
    authentic_paths = []
    for auth_dir in authentic_dirs:
        if os.path.exists(auth_dir):
            for f in os.listdir(auth_dir):
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                    authentic_paths.append(os.path.join(auth_dir, f))
    
    print(f"  Found {len(authentic_paths)} authentic images")
    
    # Collect forged images
    print("Loading forged images...")
    forged_dir = os.path.join(TRAIN_IMAGES_PATH, 'forged')
    mask_dir = TRAIN_MASKS_PATH
    
    forged_paths = []
    mask_paths = []
    
    for f in sorted(os.listdir(forged_dir)):
        if f.endswith('.png'):
            img_path = os.path.join(forged_dir, f)
            mask_path = os.path.join(mask_dir, f.replace('.png', '.npy'))
            if os.path.exists(mask_path):
                forged_paths.append(img_path)
                mask_paths.append(mask_path)
    
    print(f"  Found {len(forged_paths)} forged images with masks")
    
    # Split train/val (90/10)
    np.random.seed(42)
    
    auth_indices = np.random.permutation(len(authentic_paths))
    auth_split = int(len(authentic_paths) * 0.9)
    train_auth = [authentic_paths[i] for i in auth_indices[:auth_split]]
    val_auth = [authentic_paths[i] for i in auth_indices[auth_split:]]
    
    forge_indices = np.random.permutation(len(forged_paths))
    forge_split = int(len(forged_paths) * 0.9)
    train_forge = [forged_paths[i] for i in forge_indices[:forge_split]]
    train_masks = [mask_paths[i] for i in forge_indices[:forge_split]]
    val_forge = [forged_paths[i] for i in forge_indices[forge_split:]]
    val_masks = [mask_paths[i] for i in forge_indices[forge_split:]]
    
    print(f"\nTrain: {len(train_auth)} authentic + {len(train_forge)} forged = {len(train_auth) + len(train_forge)}")
    print(f"Val: {len(val_auth)} authentic + {len(val_forge)} forged = {len(val_auth) + len(val_forge)}")
    
    # Create datasets
    train_auth_dataset = AuthenticDataset(train_auth, img_size=IMG_SIZE, augment=True)
    train_forge_dataset = ForgedDataset(train_forge, train_masks, img_size=IMG_SIZE, augment=True)
    
    val_auth_dataset = AuthenticDataset(val_auth, img_size=IMG_SIZE, augment=False)
    val_forge_dataset = ForgedDataset(val_forge, val_masks, img_size=IMG_SIZE, augment=False)
    
    # Combine datasets
    train_dataset = ConcatDataset([train_auth_dataset, train_forge_dataset])
    val_dataset = ConcatDataset([val_auth_dataset, val_forge_dataset])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
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
    
    # Create model
    print("\nCreating model...")
    base_model = smp.FPN(
        encoder_name='timm-efficientnet-b2',
        encoder_weights='imagenet',
        in_channels=3,
        classes=1,
    )
    
    model = AttentionFPN(base_model).to(DEVICE)
    
    print(f"Model created with attention gate")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # FP-weighted loss
    criterion = FPWeightedCombinedLoss(focal_weight=0.7, dice_weight=0.3, fp_weight=FP_WEIGHT)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # Training loop
    print("\nStarting training...")
    print("Goal: Minimize FPR (False Positive Rate), Maximize Specificity")
    print()
    
    best_specificity = 0.0
    best_mcc = -1.0
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_metrics = validate(model, val_loader, criterion, DEVICE)
        
        scheduler.step()
        
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{EPOCHS} ({epoch_time:.1f}s) - "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"  Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, "
              f"F1: {val_metrics['f1']:.4f}, MCC: {val_metrics['mcc']:.4f}")
        print(f"  FPR: {val_metrics['fpr']:.4f}, Specificity: {val_metrics['specificity']:.4f}")
        
        # Save best model based on specificity (minimize FP)
        if val_metrics['specificity'] > best_specificity:
            best_specificity = val_metrics['specificity']
            best_mcc = val_metrics['mcc']
            save_path = os.path.join(MODELS_PATH, 'fp_reduction_best.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_specificity': best_specificity,
                'val_mcc': best_mcc,
                'val_fpr': val_metrics['fpr'],
                'val_precision': val_metrics['precision'],
                'val_recall': val_metrics['recall'],
                'config': {
                    'img_size': IMG_SIZE,
                    'fp_weight': FP_WEIGHT,
                    'in_channels': 3,
                    'has_attention': True,
                    'model_type': 'fp_reduction',
                }
            }, save_path)
            print(f"  ✓ Saved best model (Specificity: {best_specificity:.4f}, FPR: {val_metrics['fpr']:.4f})")
        
        print()
    
    print("="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Best Specificity: {best_specificity:.4f}")
    print(f"Best MCC: {best_mcc:.4f}")
    print(f"Model saved to: {os.path.join(MODELS_PATH, 'fp_reduction_best.pth')}")
    print()
    print("Next steps:")
    print("1. Update script_7_generate_submission.py to use this model as a veto")
    print("2. Re-run inference on test sets to verify FP reduction")


if __name__ == "__main__":
    main()
