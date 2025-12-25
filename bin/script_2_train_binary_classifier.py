#!/usr/bin/env python3
"""
Script 2: Two-Stage Pipeline - Binary Classifier Training
Stage 1: Binary classifier to reject obvious non-forgeries
Stage 2: Segmentation model for remaining images

Based on error analysis:
- FP are high-confidence, large detections
- Need to reject authentic images before segmentation

Now includes hard negative mining support:
- Loads hard negatives from hard_negative_ids.txt
- Adds extra weight to hard negatives during training
- This teaches the classifier to correctly reject tricky authentic images
"""

import os
import sys
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
from pathlib import Path
import time
import argparse

# ============================================================================
# CONFIGURATION
# ============================================================================

IMG_SIZE = 384  # Smaller for classifier
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
TRAIN_IMAGES_PATH = PROJECT_DIR / 'train_images'
MODELS_DIR = PROJECT_DIR / 'models'

# Hard negative configuration
HARD_NEGATIVE_DIR = PROJECT_DIR / 'test_authentic_100'
HARD_NEGATIVE_FILE = PROJECT_DIR / 'hard_negative_ids.txt'
HARD_NEGATIVE_WEIGHT = 3.0  # Weight multiplier for hard negatives in sampling

# Default hard negatives (used if file doesn't exist)
DEFAULT_HARD_NEGATIVE_IDS = [
]


def load_hard_negative_ids():
    """Load hard negative IDs from file, or return defaults if file doesn't exist."""
    if HARD_NEGATIVE_FILE.exists():
        print(f"Loading hard negatives from: {HARD_NEGATIVE_FILE}")
        with open(HARD_NEGATIVE_FILE, 'r') as f:
            ids = []
            for line in f:
                line = line.strip()
                if line.isdigit():
                    ids.append(int(line))
                elif line:
                    ids.append(line)  # Handle non-numeric IDs
        print(f"Loaded {len(ids)} hard negative IDs from file")
        return ids
    else:
        print(f"Hard negative file not found, using {len(DEFAULT_HARD_NEGATIVE_IDS)} default IDs")
        return DEFAULT_HARD_NEGATIVE_IDS


def load_hard_negative_images():
    """Load hard negative image paths."""
    if not HARD_NEGATIVE_DIR.exists():
        print(f"Hard negative directory not found: {HARD_NEGATIVE_DIR}")
        return []
    
    hard_negative_ids = load_hard_negative_ids()
    hard_negatives = []
    
    for img_id in hard_negative_ids:
        # Try both .png and .jpg extensions
        for ext in ['.png', '.jpg']:
            img_path = HARD_NEGATIVE_DIR / f"{img_id}{ext}"
            if img_path.exists():
                hard_negatives.append(img_path)
                break
    
    print(f"Found {len(hard_negatives)} hard negative images")
    return hard_negatives

# ============================================================================
# DATASET
# ============================================================================

class BinaryClassificationDataset(Dataset):
    """Dataset for forged vs authentic classification."""
    
    def __init__(self, forged_paths, authentic_paths, transform, hard_negative_paths=None):
        # Regular samples
        self.samples = [(p, 1, False) for p in forged_paths] + [(p, 0, False) for p in authentic_paths]
        
        # Add hard negatives (authentic images that cause false positives)
        # These are labeled as 0 (authentic) but marked as hard negatives
        if hard_negative_paths:
            self.samples += [(p, 0, True) for p in hard_negative_paths]
        
        self.transform = transform
        self.hard_negative_indices = [i for i, (_, _, is_hn) in enumerate(self.samples) if is_hn]
        np.random.shuffle(self.samples)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label, is_hard_negative = self.samples[idx]
        
        image = cv2.imread(str(img_path))
        if image is None:
            # Fallback for corrupted images
            image = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return image, torch.tensor(label, dtype=torch.float32)
    
    def get_sample_weights(self, hard_negative_weight=HARD_NEGATIVE_WEIGHT):
        """Get sample weights for WeightedRandomSampler."""
        weights = []
        for _, label, is_hard_negative in self.samples:
            if is_hard_negative:
                weights.append(hard_negative_weight)  # Oversample hard negatives
            else:
                weights.append(1.0)
        return weights


# ============================================================================
# MODEL
# ============================================================================

class ForgeryClassifier(nn.Module):
    """Binary classifier: forged (1) vs authentic (0)."""
    
    def __init__(self, backbone='efficientnet_b2'):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features).squeeze(-1)


# ============================================================================
# TRAINING
# ============================================================================

def get_transforms(is_train=True):
    if is_train:
        return A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
            ], p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    return total_loss / len(loader), correct / total


def validate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    tp = fp = tn = fn = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = torch.sigmoid(model(images))
            preds = (outputs > 0.5).float()
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            tp += ((preds == 1) & (labels == 1)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            tn += ((preds == 0) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()
    
    acc = correct / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return acc, precision, recall, tp, fp, tn, fn


def main():
    parser = argparse.ArgumentParser(description='Train binary classifier with hard negative support')
    parser.add_argument('--no-hard-negatives', action='store_true',
                        help='Disable hard negative mining')
    parser.add_argument('--hard-negative-weight', type=float, default=HARD_NEGATIVE_WEIGHT,
                        help=f'Weight for hard negatives in sampling (default: {HARD_NEGATIVE_WEIGHT})')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help=f'Number of epochs (default: {EPOCHS})')
    args = parser.parse_args()
    
    print("=" * 60)
    print("TRAINING BINARY CLASSIFIER (Stage 1)")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Hard negatives: {'disabled' if args.no_hard_negatives else 'enabled'}")
    print()
    
    # Get all images
    forged_dir = TRAIN_IMAGES_PATH / 'forged'
    authentic_dir = TRAIN_IMAGES_PATH / 'authentic'
    
    forged_images = sorted(list(forged_dir.glob('*.png')))
    authentic_images = sorted(list(authentic_dir.glob('*.png')))
    
    # Load hard negatives
    if args.no_hard_negatives:
        hard_negatives = []
    else:
        hard_negatives = load_hard_negative_images()
    
    print(f"Forged images: {len(forged_images)}")
    print(f"Authentic images: {len(authentic_images)}")
    print(f"Hard negatives: {len(hard_negatives)}")
    
    # Split for validation
    np.random.seed(42)
    
    forged_idx = np.random.permutation(len(forged_images))
    auth_idx = np.random.permutation(len(authentic_images))
    
    val_size_forged = int(0.1 * len(forged_images))
    val_size_auth = int(0.1 * len(authentic_images))
    
    train_forged = [forged_images[i] for i in forged_idx[val_size_forged:]]
    val_forged = [forged_images[i] for i in forged_idx[:val_size_forged]]
    train_authentic = [authentic_images[i] for i in auth_idx[val_size_auth:]]
    val_authentic = [authentic_images[i] for i in auth_idx[:val_size_auth]]
    
    # Split hard negatives (80% train, 20% val)
    if hard_negatives:
        np.random.shuffle(hard_negatives)
        hn_split = int(0.8 * len(hard_negatives))
        train_hard_negatives = hard_negatives[:hn_split]
        val_hard_negatives = hard_negatives[hn_split:]
    else:
        train_hard_negatives = []
        val_hard_negatives = []
    
    print(f"Train: {len(train_forged)} forged, {len(train_authentic)} authentic, {len(train_hard_negatives)} hard neg")
    print(f"Val: {len(val_forged)} forged, {len(val_authentic)} authentic, {len(val_hard_negatives)} hard neg")
    
    # Datasets
    train_dataset = BinaryClassificationDataset(
        train_forged, train_authentic, get_transforms(is_train=True),
        hard_negative_paths=train_hard_negatives
    )
    val_dataset = BinaryClassificationDataset(
        val_forged, val_authentic, get_transforms(is_train=False),
        hard_negative_paths=val_hard_negatives
    )
    
    # Use weighted sampler to oversample hard negatives
    if train_hard_negatives:
        sample_weights = train_dataset.get_sample_weights(args.hard_negative_weight)
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4)
        print(f"Using weighted sampling with hard_negative_weight={args.hard_negative_weight}")
    else:
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Model
    model = ForgeryClassifier().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    
    num_epochs = args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Training
    best_acc = 0
    save_path = MODELS_DIR / 'binary_classifier_best.pth'
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(1, num_epochs + 1):
        start = time.time()
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_acc, precision, recall, tp, fp, tn, fn = validate(model, val_loader, DEVICE)
        scheduler.step()
        
        elapsed = time.time() - start
        print(f"Epoch {epoch}/{num_epochs} - Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f}, "
              f"Val Acc: {val_acc:.3f}, Time: {elapsed:.0f}s")
        print(f"  Precision: {precision:.3f}, Recall: {recall:.3f}")
        print(f"  TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'precision': precision,
                'recall': recall,
                'hard_negatives_used': len(hard_negatives),
            }, save_path)
            print(f"  ✓ Saved best model (Acc: {val_acc:.3f})")
    
    print(f"\nTraining complete! Best Acc: {best_acc:.3f}")
    print(f"Model saved to: {save_path}")
    print(f"Hard negatives used: {len(hard_negatives)}")


if __name__ == '__main__':
    main()
