"""
Hyperparameter Grid Search & Optimization
==========================================

This script performs a comprehensive grid search over key hyperparameters:
- Learning rate (1e-4, 5e-4, 1e-3, 5e-3)
- Batch size (4, 8, 16)
- Weight decay (1e-6, 1e-5, 1e-4)
- Loss function combinations (Dice/BCE ratios)
- Optimizer choices (Adam, SGD with momentum)

Each configuration is trained and evaluated on validation set.
Best hyperparameters are identified and saved.

Expected Improvement: +2-4% over default parameters
Execution Time: ~3-4 hours (parallel processing recommended)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
import json
import itertools
from datetime import datetime
from pathlib import Path
import signal
import time

# ============================================================================
# CONFIGURATION
# ============================================================================

home_dir = '/home/swatson/work/MachineLearning/kaggle/ScientificImageForgeryDetection'
TRAIN_IMAGES_PATH = os.path.join(home_dir, "train_images")
TRAIN_MASKS_PATH = os.path.join(home_dir, "train_masks")
MODELS_PATH = os.path.join(home_dir, "models")

# Hyperparameter search space - REDUCED FOR FASTER ITERATION
HP_SEARCH_SPACE = {
    'learning_rate': [1e-4, 1e-3],  # Reduced from 4 to 2 options
    'batch_size': [8, 16],  # Reduced from 3 to 2 options, removed 4 (too slow)
    'weight_decay': [1e-5, 1e-4],  # Reduced from 3 to 2 options
    'dice_weight': [0.5, 0.7],  # Reduced from 3 to 2 options
    'optimizer': ['adam'],  # Single optimizer - Adam is most stable
}

# Training parameters
EPOCHS = 30
EARLY_STOPPING_PATIENCE = 2  # More aggressive early stopping
IMG_SIZE = 256
VAL_SPLIT = 0.2

# ============================================================================
# ATTENTION U-NET ARCHITECTURE
# ============================================================================

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class AttentionUNet(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, filters=(64, 128, 256, 512, 1024)):
        super(AttentionUNet, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.Conv1 = ConvBlock(img_ch, filters[0])
        self.Conv2 = ConvBlock(filters[0], filters[1])
        self.Conv3 = ConvBlock(filters[1], filters[2])
        self.Conv4 = ConvBlock(filters[2], filters[3])
        self.Conv5 = ConvBlock(filters[3], filters[4])

        self.Up5 = nn.ConvTranspose2d(filters[4], filters[3], kernel_size=2, stride=2)
        self.AttGate5 = AttentionGate(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.UpConv5 = ConvBlock(filters[4], filters[3])

        self.Up4 = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=2, stride=2)
        self.AttGate4 = AttentionGate(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.UpConv4 = ConvBlock(filters[3], filters[2])

        self.Up3 = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=2, stride=2)
        self.AttGate3 = AttentionGate(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.UpConv3 = ConvBlock(filters[2], filters[1])

        self.Up2 = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=2, stride=2)
        self.AttGate2 = AttentionGate(F_g=filters[0], F_l=filters[0], F_int=32)
        self.UpConv2 = ConvBlock(filters[1], filters[0])

        self.Conv_1x1 = nn.Conv2d(filters[0], output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.Conv1(x)
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        d5 = self.Up5(x5)
        x4 = self.AttGate5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.UpConv5(d5)

        d4 = self.Up4(d5)
        x3 = self.AttGate4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.UpConv4(d4)

        d3 = self.Up3(d4)
        x2 = self.AttGate3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.UpConv3(d3)

        d2 = self.Up2(d3)
        x1 = self.AttGate2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.UpConv2(d2)

        out = self.Conv_1x1(d2)
        return out


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice


class CombinedLoss(nn.Module):
    def __init__(self, weight_dice=0.5, weight_bce=0.5, pos_weight=6.0):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.weight_dice = weight_dice
        self.weight_bce = weight_bce
        self.pos_weight = pos_weight
        self.bce_loss = None  # Will be created on first forward pass

    def forward(self, inputs, targets):
        dice = self.dice_loss(inputs, targets)
        
        # Create BCEWithLogitsLoss on same device as inputs (only once)
        if self.bce_loss is None:
            pos_weight_tensor = torch.tensor([self.pos_weight], device=inputs.device, dtype=inputs.dtype)
            self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor).to(inputs.device)
        
        bce = self.bce_loss(inputs, targets)
        return self.weight_dice * dice + self.weight_bce * bce


# ============================================================================
# DATASET
# ============================================================================

class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, img_size=256, augment=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img = cv2.imread(self.image_paths[idx])
            if img is None or (hasattr(img, 'size') and img.size == 0):
                raise ValueError("Empty image")
            if not isinstance(img, np.ndarray):
                raise ValueError("Not an ndarray")
            if img.size == 0 or img.shape[0] == 0 or img.shape[1] == 0:
                raise ValueError("Invalid shape")
        except Exception as e:
            # Use fallback image
            print(f"Warning: Error loading {self.image_paths[idx]}: {e}")
            valid_idx = idx % len(self.image_paths)
            img = cv2.imread(self.image_paths[valid_idx])
            if img is None or (hasattr(img, 'size') and img.size == 0):
                img = np.ones((self.img_size, self.img_size, 3), dtype=np.uint8) * 128
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))

        # Load mask from .npy file
        mask_path = self.mask_paths[idx]
        
        # Handle zero mask marker (for authentic images)
        if mask_path == "ZERO_MASK":
            # Authentic images have no forgery - all pixels are 0
            mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        elif mask_path.endswith('.npy'):
            mask = np.load(mask_path)
            if mask.ndim == 3:
                mask = np.max(mask, axis=0)  # Take max across all forgery regions
            
            mask = cv2.resize(mask.astype(np.float32), (self.img_size, self.img_size))
            if mask.max() > 1.5:
                mask = (mask / 255.0).astype(np.float32)
            else:
                mask = mask.astype(np.float32)
        else:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
            
            mask = cv2.resize(mask, (self.img_size, self.img_size))
            if mask.max() > 1.5:
                mask = (mask / 255.0).astype(np.float32)
            else:
                mask = mask.astype(np.float32)

        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img[0] = (img[0] - 0.485) / 0.229
        img[1] = (img[1] - 0.456) / 0.224
        img[2] = (img[2] - 0.406) / 0.225

        mask = torch.from_numpy(mask).unsqueeze(0).float()

        return img, mask


# ============================================================================
# METRICS
# ============================================================================

def calculate_dice(predictions, targets):
    predictions = (predictions > 0.5).float()
    targets = targets.float()
    intersection = (predictions * targets).sum()
    dice = (2.0 * intersection) / (predictions.sum() + targets.sum() + 1e-6)
    return dice.item()


def calculate_f1_metrics(pred_binary, target):
    """
    Calculate Precision, Recall, F1-Score, and Matthews Correlation Coefficient
    
    Args:
        pred_binary: Binary predictions (0 or 1)
        target: Ground truth binary labels
    
    Returns:
        dict with precision, recall, f1, mcc values
    """
    from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef
    
    pred_flat = pred_binary.flatten()
    target_flat = target.flatten()
    
    # Avoid division by zero
    if len(pred_flat) == 0:
        return {'precision': 0, 'recall': 0, 'f1': 0, 'mcc': 0}
    
    try:
        precision = precision_score(target_flat, pred_flat, zero_division=0)
        recall = recall_score(target_flat, pred_flat, zero_division=0)
        f1 = f1_score(target_flat, pred_flat, zero_division=0)
        mcc = matthews_corrcoef(target_flat, pred_flat)
    except:
        precision = recall = f1 = mcc = 0
    
    return {'precision': precision, 'recall': recall, 'f1': f1, 'mcc': mcc}


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def evaluate_hyperparameters(model, train_loader, val_loader, device, config, epochs=EPOCHS):
    """Train and evaluate a single hyperparameter configuration"""
    
    criterion = CombinedLoss(weight_dice=config['dice_weight'], 
                             weight_bce=1-config['dice_weight'], pos_weight=6.0)

    # Create optimizer based on config
    if config['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), 
                              lr=config['learning_rate'],
                              weight_decay=config['weight_decay'])
    else:  # SGD
        optimizer = optim.SGD(model.parameters(),
                             lr=config['learning_rate'],
                             momentum=0.9,
                             weight_decay=config['weight_decay'])

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2)

    best_val_mcc = -1.0  # Track best MCC (ranges -1 to +1)
    patience_counter = 0
    best_val_dice = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        val_dice_scores = []
        val_f1_scores = []
        val_precision_scores = []
        val_recall_scores = []
        val_mcc_scores = []

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                dice = calculate_dice(torch.sigmoid(outputs), masks)
                val_dice_scores.append(dice)
                
                # Calculate precision/recall/F1
                pred_binary = (torch.sigmoid(outputs) > 0.5).cpu().numpy().astype(np.uint8)
                mask_binary = masks.cpu().numpy().astype(np.uint8)
                metrics = calculate_f1_metrics(pred_binary, mask_binary)
                val_f1_scores.append(metrics['f1'])
                val_precision_scores.append(metrics['precision'])
                val_recall_scores.append(metrics['recall'])
                val_mcc_scores.append(metrics['mcc'])

        val_loss /= len(val_loader)
        val_dice = np.mean(val_dice_scores)
        val_f1 = np.mean(val_f1_scores)
        val_precision = np.mean(val_precision_scores)
        val_recall = np.mean(val_recall_scores)
        val_mcc = np.mean(val_mcc_scores)
        scheduler.step(val_loss)

        if val_mcc > best_val_mcc:
            best_val_mcc = val_mcc
            best_val_dice = val_dice
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                break

    return best_val_mcc, best_val_dice


# ============================================================================
# GRID SEARCH EXECUTION
# ============================================================================

def main():
    from datetime import datetime
    start_time = datetime.now()
    print(f"\nStart Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n" + "="*80)
    print("HYPERPARAMETER GRID SEARCH")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    os.makedirs(MODELS_PATH, exist_ok=True)

    # Load dataset - use properly labeled data
    # train_images/authentic_augmented = augmented authentic images (2700 total)
    # train_images/forged = images with forgery masks
    authentic_dir = os.path.join(home_dir, 'train_images', 'authentic_augmented')  # Augmented authentic images
    forged_dir = os.path.join(home_dir, 'train_images', 'forged')
    
    # Get all image files from both directories
    authentic_files = [f for f in os.listdir(authentic_dir) if f.endswith('.png')]
    forged_files = [f for f in os.listdir(forged_dir) if f.endswith('.png')]
    
    # Create full paths
    image_paths = []
    mask_paths = []
    
    # Add authentic images (none have explicit masks - use implicit zero)
    for f in authentic_files:
        img_path = os.path.join(authentic_dir, f)
        # Authentic images always use zero mask (no forgery)
        zero_mask_path = "ZERO_MASK"  # Special marker for zero mask
        image_paths.append(img_path)
        mask_paths.append(zero_mask_path)
    
    # Add forged images (all have explicit masks)
    for f in forged_files:
        img_path = os.path.join(forged_dir, f)
        mask_base = os.path.basename(f).replace('.png', '.npy')
        mask_path = os.path.join(TRAIN_MASKS_PATH, mask_base)
        # Only include if mask exists
        if os.path.exists(mask_path):
            image_paths.append(img_path)
            mask_paths.append(mask_path)
    
    # Sort together to maintain pairing (don't sort separately!)
    paired = sorted(zip(image_paths, mask_paths))
    image_paths = [p[0] for p in paired]
    mask_paths = [p[1] for p in paired]

    # Use ALL training data for training (no split)
    train_images = image_paths
    train_masks = mask_paths
    
    # Use ACTUAL validation set (validation_images/ and validation_masks/)
    val_images_dir = os.path.join(home_dir, 'validation_images')
    val_masks_dir = os.path.join(home_dir, 'validation_masks')
    
    val_images = []
    val_masks = []
    for f in sorted(os.listdir(val_images_dir)):
        if f.endswith('.png'):
            mask_file = os.path.join(val_masks_dir, f.replace('.png', '.npy'))
            if os.path.exists(mask_file):
                val_images.append(os.path.join(val_images_dir, f))
                val_masks.append(mask_file)

    train_dataset = SegmentationDataset(train_images, train_masks, IMG_SIZE, augment=True)
    val_dataset = SegmentationDataset(val_images, val_masks, IMG_SIZE, augment=False)

    # Generate hyperparameter combinations
    hp_combinations = list(itertools.product(
        HP_SEARCH_SPACE['learning_rate'],
        HP_SEARCH_SPACE['batch_size'],
        HP_SEARCH_SPACE['weight_decay'],
        HP_SEARCH_SPACE['dice_weight'],
        HP_SEARCH_SPACE['optimizer']
    ))

    print(f"Total configurations to test: {len(hp_combinations)}")
    print(f"Training set: {len(train_images)} samples")
    print(f"Validation set: {len(val_images)} samples (from validation_images/)")
    print(f"⏱️  Timeout: 10 hours for grid search. Will save best results found so far if timeout occurs.\n")

    results = []
    best_config = None
    best_val_dice = 0
    grid_search_start = time.time()
    timeout_seconds = 36000  # 10 hours timeout

    for idx, (lr, bs, wd, dw, opt) in enumerate(hp_combinations):
        # Check timeout
        elapsed = time.time() - grid_search_start
        if elapsed > timeout_seconds:
            print(f"\n⏱️  TIMEOUT: Grid search exceeded {timeout_seconds//3600} hours limit. Saving results...")
            break
        config = {
            'learning_rate': lr,
            'batch_size': bs,
            'weight_decay': wd,
            'dice_weight': dw,
            'optimizer': opt
        }

        print(f"\n[{idx+1}/{len(hp_combinations)}] Testing: LR={lr}, BS={bs}, WD={wd}, DW={dw:.1f}, OPT={opt}")

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False, num_workers=0)

        # Create model
        model = AttentionUNet(img_ch=3, output_ch=1).to(device)

        # Evaluate
        try:
            val_f1, val_dice = evaluate_hyperparameters(model, train_loader, val_loader, 
                                                          device, config, EPOCHS)
            
            config['val_f1'] = val_f1
            config['val_dice'] = val_dice
            results.append(config)

            print(f"  → Val F1: {val_f1:.4f}, Val Dice: {val_dice:.4f}")

            if val_dice > best_val_dice:
                best_val_dice = val_dice
                best_config = config.copy()
                print(f"  ✓ NEW BEST!")

        except Exception as e:
            print(f"  ✗ Error: {e}")

    # Sort results by validation dice
    results_sorted = sorted(results, key=lambda x: x['val_dice'], reverse=True)

    # Save results
    print("\n" + "="*80)
    print("TOP 10 CONFIGURATIONS")
    print("="*80)

    for i, result in enumerate(results_sorted[:10]):
        print(f"\n{i+1}. Val Dice: {result['val_dice']:.4f} (F1: {result['val_f1']:.4f})")
        print(f"   LR: {result['learning_rate']}, BS: {result['batch_size']}, "
              f"WD: {result['weight_decay']}, DW: {result['dice_weight']:.1f}, OPT: {result['optimizer']}")

    # Save best config
    results_path = os.path.join(MODELS_PATH, 'hyperparameter_search_results.json')
    with open(results_path, 'w') as f:
        json.dump(results_sorted, f, indent=2, default=str)
    print(f"\n✓ All results saved to: {results_path}")

    # Save best config
    best_config_path = os.path.join(MODELS_PATH, 'best_hyperparameters.json')
    with open(best_config_path, 'w') as f:
        json.dump(best_config, f, indent=2)
    print(f"✓ Best configuration saved to: {best_config_path}")

    print("\n" + "="*80)
    print("BEST HYPERPARAMETERS")
    print("="*80)
    if best_config is not None:
        print(f"Learning Rate: {best_config['learning_rate']}")
        print(f"Batch Size: {best_config['batch_size']}")
        print(f"Weight Decay: {best_config['weight_decay']}")
        print(f"Dice Weight: {best_config['dice_weight']}")
        print(f"Optimizer: {best_config['optimizer']}")
        print(f"Validation Dice: {best_config['val_dice']:.4f}")
    else:
        print("⚠️  Grid search timed out - no valid configurations tested")
        print("Using default hyperparameters for final training")

    print("\n" + "="*80)
    print("EXPECTED IMPROVEMENT: +2-4% with optimized hyperparameters")
    print("NEXT STEP: Train final models using best hyperparameters")
    print("="*80)
    
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"\nEnd Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Duration: {duration}")


if __name__ == "__main__":
    main()
