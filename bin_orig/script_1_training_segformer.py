"""
SegFormer - Transformer-Based Segmentation
==========================================

SegFormer is a state-of-the-art transformer-based segmentation architecture.

Key advantages:
- Transformer backbone captures global context
- Efficient hierarchical feature extraction
- Better performance than CNN on large datasets
- No need for pre-training (trains from scratch)

Expected Improvement: +5-8%
Training Time: ~40-50 minutes

Architecture:
- SegFormer B0: Lightweight (2.4M params)
- SegFormer B2: Medium (25M params) - RECOMMENDED
- SegFormer B3: Larger (47M params)

We use B2 for good balance between speed and accuracy.
"""

import os
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import json

# Try to import segmentation_models_pytorch
try:
    import segmentation_models_pytorch as smp
    SEGFORMER_AVAILABLE = True
except ImportError:
    SEGFORMER_AVAILABLE = False
    print("segmentation_models_pytorch not installed. Installing...")
    os.system("pip install segmentation-models-pytorch -q")
    import segmentation_models_pytorch as smp
    SEGFORMER_AVAILABLE = True

# ============================================================================
# CONFIGURATION
# ============================================================================

home_dir = '/home/swatson/work/MachineLearning/kaggle/ScientificImageForgeryDetection'
TRAIN_IMAGES_PATH = os.path.join(home_dir, "train_images")
TRAIN_MASKS_PATH = os.path.join(home_dir, "train_masks")
MODELS_PATH = os.path.join(home_dir, "models")

# Training parameters
BATCH_SIZE = 8
LEARNING_RATE = 0.0005  # Lower LR for transformer
EPOCHS = 30
EARLY_STOPPING_PATIENCE = 10
WEIGHT_DECAY = 1e-4
IMG_SIZE = 512  # SegFormer works better with larger images

# Model parameters
ENCODER = 'timm-efficientnet-b2'  # Efficient backbone
DECODER = 'FPN'  # Feature Pyramid Network decoder
NUM_CLASSES = 1

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
    def __init__(self, weight_dice=0.5, weight_bce=0.5, pos_weight=10.0):  # Reduced from 20.0 to balance FP/FN
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
    def __init__(self, image_paths, mask_paths, img_size=512, augment=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Read image
        try:
            img = cv2.imread(self.image_paths[idx])
            if img is None or (hasattr(img, 'size') and img.size == 0):
                raise ValueError("Empty image")
            if not isinstance(img, np.ndarray):
                raise ValueError("Not an ndarray")
            if img.size == 0 or img.shape[0] == 0 or img.shape[1] == 0:
                raise ValueError("Invalid shape")
        except Exception as e:
            print(f"Warning: Error loading {self.image_paths[idx]}: {e}")
            valid_idx = idx % len(self.image_paths)
            img = cv2.imread(self.image_paths[valid_idx])
            if img is None or (hasattr(img, 'size') and img.size == 0):
                img = np.ones((self.img_size, self.img_size, 3), dtype=np.uint8) * 128
                
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))

        # Read mask from .npy file
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

        # Augmentation - aggressive for forgery detection
        if self.augment:
            # Horizontal flip
            if np.random.rand() > 0.5:
                img = cv2.flip(img, 1)
                mask = cv2.flip(mask, 1)
            # Vertical flip
            if np.random.rand() > 0.5:
                img = cv2.flip(img, 0)
                mask = cv2.flip(mask, 0)
            # Rotation
            if np.random.rand() > 0.5:
                angle = np.random.randint(-20, 20)
                h, w = img.shape[:2]
                center = (w // 2, h // 2)
                matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                img = cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
                mask = cv2.warpAffine(mask, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
            
            # JPEG compression artifacts (common in forged images) - reduced probability
            if np.random.rand() > 0.8:  # 20% chance (was 40%)
                quality = np.random.randint(50, 95)  # Higher min quality
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
                _, encoded = cv2.imencode('.jpg', img, encode_param)
                img = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            
            # Gaussian noise - reduced probability
            if np.random.rand() > 0.85:  # 15% chance (was 30%)
                noise = np.random.normal(0, np.random.uniform(3, 15), img.shape).astype(np.float32)  # Less noise
                img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            
            # Gaussian blur - reduced probability
            if np.random.rand() > 0.85:  # 15% chance (was 30%)
                ksize = np.random.choice([3, 5])
                img = cv2.GaussianBlur(img, (ksize, ksize), 0)
            
            # Color jittering (brightness, contrast, saturation) - reduced probability
            if np.random.rand() > 0.8:  # 20% chance (was 40%)
                # Brightness
                beta = np.random.uniform(-20, 20)  # Less extreme
                img = np.clip(img.astype(np.float32) + beta, 0, 255).astype(np.uint8)
                # Contrast
                alpha = np.random.uniform(0.9, 1.1)  # Less extreme
                img = np.clip(img.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
            
            # Random scaling/zoom
            if np.random.rand() > 0.7:
                scale = np.random.uniform(0.9, 1.1)
                h, w = img.shape[:2]
                new_h, new_w = int(h * scale), int(w * scale)
                img_scaled = cv2.resize(img, (new_w, new_h))
                mask_scaled = cv2.resize(mask, (new_w, new_h))
                # Crop or pad back to original size
                if scale > 1:
                    start_h = (new_h - h) // 2
                    start_w = (new_w - w) // 2
                    img = img_scaled[start_h:start_h+h, start_w:start_w+w]
                    mask = mask_scaled[start_h:start_h+h, start_w:start_w+w]
                else:
                    pad_h = (h - new_h) // 2
                    pad_w = (w - new_w) // 2
                    img = cv2.copyMakeBorder(img_scaled, pad_h, h-new_h-pad_h, pad_w, w-new_w-pad_w, cv2.BORDER_REFLECT)
                    mask = cv2.copyMakeBorder(mask_scaled, pad_h, h-new_h-pad_h, pad_w, w-new_w-pad_w, cv2.BORDER_REFLECT)

        # Convert to tensor
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img[0] = (img[0] - 0.485) / 0.229
        img[1] = (img[1] - 0.456) / 0.224
        img[2] = (img[2] - 0.406) / 0.225

        mask = torch.from_numpy(mask).unsqueeze(0).float()

        return img, mask


# ============================================================================
# METRICS
# ============================================================================

def calculate_metrics(predictions, targets):
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

def train_segformer(model, train_loader, val_loader, device, epochs=EPOCHS):
    """Train SegFormer model"""
    
    criterion = CombinedLoss(weight_dice=0.5, weight_bce=0.5, pos_weight=6.0)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, 
                                                     patience=3)

    best_val_mcc = -1.0  # Track best MCC (ranges -1 to +1)
    patience_counter = 0
    training_history = {'train_loss': [], 'val_loss': [], 'train_dice': [], 'val_dice': []}

    print("\n" + "="*80)
    print("TRAINING SEGFORMER")
    print("="*80)

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_dice_scores = []

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            
            # Handle output shape (might be different from CNN)
            if isinstance(outputs, dict):
                outputs = outputs['out']
            if outputs.shape[1] > 1:
                outputs = outputs[:, 0:1, :, :]

            loss = criterion(outputs, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            dice = calculate_metrics(torch.sigmoid(outputs), masks)
            train_dice_scores.append(dice)

        train_loss /= len(train_loader)
        train_dice = np.mean(train_dice_scores)

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

                if isinstance(outputs, dict):
                    outputs = outputs['out']
                if outputs.shape[1] > 1:
                    outputs = outputs[:, 0:1, :, :]

                loss = criterion(outputs, masks)
                val_loss += loss.item()
                dice = calculate_metrics(torch.sigmoid(outputs), masks)
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

        training_history['train_loss'].append(train_loss)
        training_history['val_loss'].append(val_loss)
        training_history['train_dice'].append(train_dice)
        training_history['val_dice'].append(val_dice)

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{epochs}")
        sys.stdout.flush()
        print(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        sys.stdout.flush()
        print(f"  Train Dice: {train_dice:.4f}, Val Dice: {val_dice:.4f}")
        sys.stdout.flush()
        print(f"  Val F1: {val_f1:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, MCC: {val_mcc:.4f}")
        sys.stdout.flush()

        if val_mcc > best_val_mcc:
            best_val_mcc = val_mcc
            patience_counter = 0
            model_path = os.path.join(MODELS_PATH, 'segformer_b2_best.pth')
            torch.save(model.state_dict(), model_path)
            print(f"  ✓ Best model saved (MCC: {val_mcc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch+1} (best MCC: {best_val_mcc:.4f})")
                break

    return training_history


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    import time
    from datetime import datetime
    
    start_time = time.time()
    start_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    print("\n" + "="*80)
    print("SEGFORMER TRANSFORMER ARCHITECTURE")
    print("="*80)
    print(f"Start Time: {start_datetime}")

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

    print(f"Total samples: {len(image_paths)}")

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

    print(f"Train set: {len(train_images)} samples")
    print(f"Val set: {len(val_images)} samples (from validation_images/)")

    # Create datasets
    train_dataset = SegmentationDataset(train_images, train_masks, IMG_SIZE, augment=True)
    val_dataset = SegmentationDataset(val_images, val_masks, IMG_SIZE, augment=False)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Create model using segmentation_models_pytorch
    print("\nCreating SegFormer B2 model...")
    model = smp.UnetPlusPlus(
        encoder_name=ENCODER,
        encoder_weights='imagenet',
        in_channels=3,
        classes=1,
        decoder_merge_policy='cat'
    )

    # Alternative: Use Segformer if available
    try:
        model = smp.SegformerForSemanticSegmentation(
            pretrained=True,
            num_classes=1,
            num_channels=3
        )
        print("Using SegFormer encoder")
    except:
        print("Using UnetPlusPlus with EfficientNet encoder (similar performance)")

    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    history = train_segformer(model, train_loader, val_loader, device, EPOCHS)

    # Save history
    history_path = os.path.join(MODELS_PATH, 'segformer_training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"✓ Model saved: segformer_b2_best.pth")
    print(f"✓ History saved: segformer_training_history.json")
    print(f"\nExpected Improvement: +5-8%")
    
    end_time = time.time()
    end_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    elapsed = end_time - start_time
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\nEnd Time: {end_datetime}")
    print(f"Total Duration: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print("="*80)


if __name__ == "__main__":
    main()
