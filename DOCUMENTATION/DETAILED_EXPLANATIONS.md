# Detailed Technical Explanations

## Table of Contents

1. [Script 1: Base Model Training (v3)](#script-1-detailed-breakdown)
2. [Script 2: Binary Classifier Training](#script-2-detailed-breakdown)
3. [Script 3: Hard Negative Fine-tuning (v4)](#script-3-detailed-breakdown)
4. [Script 4: Inference Pipeline](#script-4-detailed-breakdown)
5. [Core Components](#core-components)
6. [Supplementary Scripts](#supplementary-scripts-detailed-breakdown)
   - [Generate Forged Augmented Images](#generate-forged-augmented-images)
   - [Generate Authentic Augmented Images](#generate-authentic-augmented-images)
   - [Find Hard Negatives](#find-hard-negatives)
   - [Validate Test](#validate-test)
   - [Full Pipeline Orchestration](#full-pipeline-orchestration)
7. [Mathematical Foundations](#mathematical-foundations)

---

## Script 1: Detailed Breakdown

**File**: `bin/script_1_train_v3.py`

### Overall Flow

```
script_1_train_v3.py
├── Configuration (IMG_SIZE, BATCH_SIZE, EPOCHS, LR, DEVICE, paths)
├── Load hard negative IDs (if available)
├── For each model in [highres_no_ela, hard_negative, high_recall, enhanced_aug]:
│   ├── Collect data (forged, authentic, hard negatives)
│   ├── Create datasets with augmentations
│   ├── Build FPN + Attention model
│   ├── For each epoch:
│   │   ├── Load batches (weighted random sampling)
│   │   ├── Forward pass: image → model → logits
│   │   ├── Compute loss (Focal + Dice + FP penalty)
│   │   ├── Backward pass: loss.backward()
│   │   ├── Update weights: optimizer.step()
│   │   └── Log progress
│   └── Save best checkpoint
└── Exit
```

### Configuration Section

```python
IMG_SIZE = 512              # All images resized to 512×512
BATCH_SIZE = 4              # 4 images per batch (memory limit)
EPOCHS = 25                 # 25 training passes through data
LR = 1e-4                   # Learning rate: 0.0001
DEVICE = torch.device('cuda')  # Use GPU
```

**Why these values?**
- `IMG_SIZE=512`: Balances detail preservation vs. memory. Smaller (256) loses detail; larger (1024) requires more VRAM
- `BATCH_SIZE=4`: Limited by VRAM on 8GB GPUs. Would use 8-16 on 16GB+
- `EPOCHS=25`: Good convergence point. Diminishing returns after 25
- `LR=1e-4`: Standard for fine-tuning. Smaller than from-scratch (1e-3)

### Data Collection Flow

```python
def collect_data(skip_hard_negatives=False):
    # 1. FORGED IMAGES
    forged_images = []
    forged_masks = []
    
    # Original forged (2,751 images)
    for img_path in FORGED_ORIGINAL.glob('*.png'):
        mask_path = MASKS_ORIGINAL / f"{img_path.stem}.npy"
        if mask_path.exists():
            forged_images.append(img_path)
            forged_masks.append(mask_path)
    # Result: 2,751 pairs
    
    # Augmented forged (7,564 images)
    for img_path in FORGED_AUGMENTED.glob('*.png'):
        mask_path = MASKS_AUGMENTED / f"{img_path.stem}.npy"
        if mask_path.exists():
            forged_images.append(img_path)
            forged_masks.append(mask_path)
    # Result: ~10,315 total forged
    
    # 2. AUTHENTIC IMAGES
    authentic_images = []
    
    # Original authentic (~2,400 images)
    for img_path in AUTHENTIC_DIR.glob('*.png'):
        authentic_images.append(img_path)
    
    # Augmented v1 & v2 (sample to balance)
    authentic_images.extend(AUTHENTIC_AUG_DIR.glob('*.png')[:2000])
    authentic_images.extend(AUTHENTIC_AUG_V2_DIR.glob('*.png')[:2000])
    # Result: ~12,200 total authentic
    
    # 3. HARD NEGATIVES
    hard_negatives = []  # Authentic images producing FPs
    hard_negative_ids = load_hard_negative_ids(skip_hard_negatives)
    for img_id in hard_negative_ids:
        img_path = HARD_NEGATIVE_DIR / f"{img_id}.png"
        if img_path.exists():
            hard_negatives.append(img_path)
    # Result: ~73 hard negatives
    
    return forged_images, forged_masks, authentic_images, hard_negatives
```

**Key Statistics**:
- Total images: ~22,500 (10,315 forged + 12,200 authentic)
- Ratio: ~1:1 forged to authentic (balanced)
- Training data split optimized for deep learning

### Dataset Classes

#### ForgeryDataset (for forged images)

```python
class ForgeryDataset(Dataset):
    """Handles paired images and masks."""
    
    def __init__(self, image_paths, mask_paths, img_size=512, augment=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        
        if augment:
            # Heavy augmentation to improve robustness
            self.transform = A.Compose([
                # Geometric augmentations
                A.RandomResizedCrop(size=(512, 512), scale=(0.7, 1.0)),
                    # ↑ Crop 70-100% of image, resize back to 512
                    # Helps network learn multi-scale forgery patterns
                A.HorizontalFlip(p=0.5),    # 50% chance
                A.VerticalFlip(p=0.5),      # 50% chance
                A.RandomRotate90(p=0.5),    # 90° rotations
                
                # Color augmentations
                A.OneOf([
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                ], p=0.5),                  # Pick one, 50% probability
                
                # Blur and noise (simulate real images)
                A.GaussianBlur(blur_limit=(3, 5), p=0.2),
                A.GaussNoise(std_range=(0.01, 0.03), p=0.2),
                
                # Normalization (ImageNet stats)
                A.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225]),
                ToTensorV2()  # Convert to torch tensor
            ], additional_targets={'mask': 'mask'})
```

**Why these augmentations?**
- **RandomResizedCrop**: Teaches model to find forgeries at different scales
- **Flips**: Forgery location shouldn't matter (translation invariance)
- **ColorJitter**: Images have different lighting, saturation
- **Blur/Noise**: Simulates JPEG compression, camera noise
- **ImageNet Normalize**: Matches EfficientNet B2 expectations
  - Formula: `(pixel - mean) / std` for each channel
  - Red channel: `(x - 0.485) / 0.229`, etc.

#### AuthenticDataset (for authentic images)

```python
class AuthenticDataset(Dataset):
    """Handles authentic images without masks."""
    
    def __getitem__(self, idx):
        image = cv2.imread(str(self.image_paths[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        transformed = self.transform(image=image)
        image_tensor = transformed['image']
        
        # Create zero mask (no forgery)
        mask_tensor = torch.zeros(1, self.img_size, self.img_size)
        
        return image_tensor, mask_tensor
```

**Key point**: Authentic images return all-zero masks, which trains the model to predict low confidence on clean images.

### Weighted Sampling Strategy

```python
# Compute sample weights based on importance
weights = []

# 1. Forged weights (prioritize small forgeries)
for mask_path in forged_masks:
    mask = np.load(str(mask_path))
    ratio = (mask > 0.5).sum() / mask.size * 100  # % of image that's forged
    
    if ratio < 1.0:
        weights.append(3.0)    # Small forgeries (hard to detect) - 3x weight
    elif ratio < 5.0:
        weights.append(2.0)    # Medium forgeries - 2x weight
    else:
        weights.append(1.0)    # Large forgeries (easy to detect) - normal weight

# 2. Authentic weights
weights.extend([1.0] * len(authentic_images))  # Normal weight

# 3. Hard negative weights (very important)
weights.extend([5.0] * len(hard_negatives))    # 5x weight to focus training

# Use WeightedRandomSampler
sampler = WeightedRandomSampler(weights, num_samples=len(combined_dataset), replacement=True)
dataloader = DataLoader(combined_dataset, batch_size=BATCH_SIZE, sampler=sampler)
```

**Why weighted sampling?**
- **Small forgeries**: Hard to detect, need more training examples
- **Hard negatives**: Most critical - FPs come from authentic images like these
- **Large forgeries**: Easy, don't need as much training

**Result**: Every epoch sees:
- ~50% weight on small/hard forgeries
- ~30% weight on medium forgeries
- ~20% weight on large forgeries + authentic

### Model Architecture

#### AttentionGate

```python
class AttentionGate(nn.Module):
    """Spatial attention mechanism."""
    def __init__(self, in_channels):
        super().__init__()
        hidden1 = max(8, in_channels // 2)    # Compress
        hidden2 = max(4, in_channels // 4)    # Further compress
        
        self.conv = nn.Sequential(
            # Path 1: Learn which spatial regions are important
            nn.Conv2d(in_channels, hidden1, kernel_size=1),      # 1×1 conv
            nn.BatchNorm2d(hidden1),                              # Normalize
            nn.ReLU(inplace=True),                                # Activate
            
            # Path 2: Spatial filtering
            nn.Conv2d(hidden1, hidden2, kernel_size=3, padding=1),  # 3×3 conv
            nn.BatchNorm2d(hidden2),
            nn.ReLU(inplace=True),
            
            # Output: Single channel (0-1) for each spatial location
            nn.Conv2d(hidden2, 1, kernel_size=1),
            nn.Sigmoid()  # Output: 0 (ignore) to 1 (focus)
        )
    
    def forward(self, x):
        # Compute attention map
        attention = self.conv(x)  # Shape: (B, 1, H, W)
        
        # Apply attention: multiply element-wise
        return x * attention  # Broadcast multiplication
```

**What it does**:
1. Takes feature map (B, C, H, W)
2. Learns which spatial regions contain forgeries
3. Multiplies original features by attention map (0-1)
4. Result: Regions marked as important are enhanced, others suppressed

**Example**:
```
Original:  [1.5, 2.0, 0.5, 1.0]
Attention: [0.1, 0.9, 0.2, 0.8]
Output:    [0.15, 1.8, 0.1, 0.8]  (element-wise multiply)
```

#### AttentionFPN (complete model)

```python
class AttentionFPN(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model  # FPN with EfficientNet B2
        self.attention = AttentionGate(1)  # Single channel output
    
    def forward(self, x):
        # 1. FPN produces raw forgery probability map
        x = self.base(x)  # Input: (B, 3, 512, 512) → Output: (B, 1, 512, 512)
        
        # 2. Attention gate refines the map
        x = self.attention(x)  # Multiply by learned attention weights
        
        return x  # Final: (B, 1, 512, 512) logits
```

**Architecture summary**:
```
Input Image (B, 3, 512, 512)
    ↓
EfficientNet B2 Encoder (ImageNet pretrained)
    ↓
FPN Decoder (multi-scale feature fusion)
    ↓ 
Output Logits (B, 1, 512, 512)
    ↓
Attention Gate (spatial refinement)
    ↓
Final Output (B, 1, 512, 512)
```

### Loss Functions

#### FocalLoss

```python
class FocalLoss(nn.Module):
    """Addresses class imbalance by down-weighting easy examples."""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # Down-weight negative samples
        self.gamma = gamma  # Focus on hard examples
    
    def forward(self, pred, target):
        # Standard binary cross-entropy
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        # BCE = -[y*log(p) + (1-y)*log(1-p)]
        #   where y is target, p is probability
        
        # Probability of ground truth class
        pt = torch.exp(-bce)  # High if prediction confident, low if uncertain
        
        # Focal loss: multiply BCE by (1-pt)^gamma
        #   Easy examples: pt → 1.0, (1-pt)^gamma → 0, loss → 0
        #   Hard examples: pt → 0.5, (1-pt)^gamma → large, loss → large
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        
        return focal.mean()
```

**Example calculation**:
```
True label: 1 (positive sample)
Predicted probability: 0.9 (high confidence)
BCE = -log(0.9) ≈ 0.105
pt = exp(-0.105) ≈ 0.90
Focal = 0.25 * (0.10)^2.0 * 0.105 ≈ 0.0003  ← Very small loss

True label: 1 (positive sample)
Predicted probability: 0.3 (low confidence - HARD)
BCE = -log(0.3) ≈ 1.204
pt = exp(-1.204) ≈ 0.30
Focal = 0.25 * (0.70)^2.0 * 1.204 ≈ 0.147  ← Large loss
```

**Why Focal Loss?**
- Binary cross-entropy treats all errors equally
- In imbalanced datasets, easy negatives dominate
- Focal loss down-weights easy examples, focuses on hard ones
- Result: Better learning from challenging cases

#### DiceLoss

```python
class CombinedLoss(nn.Module):
    def forward(self, pred, target):
        # ... Focal loss computation ...
        
        # Dice Loss: 2*|A∩B| / (|A| + |B|)
        pred_sigmoid = torch.sigmoid(pred)  # Convert logits to probabilities
        intersection = (pred_sigmoid * target).sum()  # Overlap
        dice = (2 * intersection + 1) / (pred_sigmoid.sum() + target.sum() + 1)
        dice_loss = 1 - dice
```

**Why Dice Loss for segmentation?**
- BCEWithLogits assumes pixel independence
- Dice cares about overall overlap (IoU-like)
- Example:
  - True: [1,1,0,0], Pred: [0.9, 0.8, 0.2, 0.1]
  - BCE: Sums individual errors
  - Dice: Penalizes imperfect overlap

**Example**:
```
Intersection = (0.9×1) + (0.8×1) + (0.2×0) + (0.1×0) = 1.7
Union = (0.9+0.8+0.2+0.1) + (1+1+0+0) = 2.0 + 2.0 = 4.0
Dice = 2*1.7/4.0 = 0.85
Loss = 1 - 0.85 = 0.15
```

#### FP Penalty

```python
class CombinedLoss(nn.Module):
    def forward(self, pred, target):
        # ... Other losses ...
        
        # FP Penalty: Extra cost for predicting forgery on authentic images
        is_authentic = (target.sum(dim=(1,2,3)) == 0).float()
        # is_authentic[i] = 1 if image i has no forgery (all zeros), else 0
        
        fp_loss = (pred_sigmoid * is_authentic.view(-1, 1, 1, 1)).mean() * self.fp_penalty
        # Multiply prediction by flag, average, scale by penalty
        
        # If image is authentic and pred_sigmoid is high → high loss
        # If image is forged → masked out, no contribution
        
        return (self.focal_weight * focal_loss + 
                self.dice_weight * dice_loss + 
                fp_loss)
```

**Why FP Penalty?**
- Without it: Model might over-predict forgery to improve dice/focal
- With it: Extra penalty for false positives forces conservative predictions
- Trades recall for precision

**Combined Loss Formula**:
$$L = 0.5 \times L_{Focal} + 0.5 \times L_{Dice} + 2.0 \times L_{FP}$$

### Training Loop

```python
def train_model(model_name, forged_images, forged_masks, authentic_images, hard_negatives):
    # 1. Create model and move to device
    model = AttentionFPN(base_model).to(DEVICE)
    
    # 2. Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,              # Learning rate
        weight_decay=1e-4   # L2 regularization
    )
    # AdamW = Adam with decoupled weight decay
    # Helps prevent overfitting
    
    # 3. Setup learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS  # Cosine annealing over full training
    )
    # Learning rate: decreases from LR to 0 following cosine curve
    
    # 4. Loss function
    criterion = CombinedLoss(fp_penalty=2.0)
    
    # 5. Training loop
    best_loss = float('inf')
    for epoch in range(EPOCHS):
        model.train()  # Enable dropout, batch norm updates
        epoch_loss = 0
        
        for batch_idx, (images, masks) in enumerate(dataloader):
            # Move batch to device
            images = images.to(DEVICE)  # (B, 3, 512, 512)
            masks = masks.to(DEVICE)     # (B, 1, 512, 512)
            
            # Forward pass
            optimizer.zero_grad()                    # Reset gradients
            outputs = model(images)                  # (B, 1, 512, 512) logits
            loss = criterion(outputs, masks)         # Compute loss
            
            # Backward pass
            loss.backward()                          # Compute gradients
            
            # Update weights
            optimizer.step()                         # Gradient descent step
            
            epoch_loss += loss.item()
        
        # Update learning rate
        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, output_path)
            print(f"Epoch {epoch+1}: Loss={avg_loss:.4f} ✓ SAVED")
        else:
            print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}")
```

**Training dynamics**:
- **Epoch 1-5**: Rapid loss decrease (learning basic patterns)
- **Epoch 5-15**: Moderate decrease (refining details)
- **Epoch 15-25**: Diminishing returns (minor improvements)
- **Best checkpoint**: Usually epoch 18-22, saves automatically

---

## Script 2: Detailed Breakdown

**File**: `bin/script_2_train_binary_classifier.py`

### Purpose and Architecture

**Goal**: Create fast binary classifier to filter obvious authentic images.

```
Input Image (B, 3, 384, 384)
    ↓
EfficientNet B2 Backbone (ImageNet pretrained)
    ↓
Global Average Pool → (B, num_features)
    ↓
FC(256) → ReLU → Dropout(0.3)
    ↓
FC(1) → Sigmoid → Probability
    ↓
Output: 0.0-1.0 (authentic to forged)
```

**Key differences from script 1**:
- Classification task (1 class), not segmentation
- Smaller input size (384 vs 512) - faster
- Simpler model (just classifier head)
- Simpler loss (BCE with logits)

### Dataset Implementation

```python
class BinaryClassificationDataset(Dataset):
    """Combines forged, authentic, and hard negative images."""
    
    def __init__(self, forged_paths, authentic_paths, transform, hard_negative_paths=None):
        # Create tuples: (image_path, label, is_hard_negative)
        self.samples = []
        self.samples += [(p, 1, False) for p in forged_paths]      # 1 = forged
        self.samples += [(p, 0, False) for p in authentic_paths]   # 0 = authentic
        
        if hard_negative_paths:
            # Hard negatives are authentic (label=0) but flagged
            self.samples += [(p, 0, True) for p in hard_negative_paths]
        
        self.transform = transform
        np.random.shuffle(self.samples)
    
    def __getitem__(self, idx):
        img_path, label, is_hard_negative = self.samples[idx]
        
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return image, torch.tensor(label, dtype=torch.float32)
    
    def get_sample_weights(self, hard_negative_weight=3.0):
        """Compute weights for WeightedRandomSampler."""
        weights = []
        for _, label, is_hard_negative in self.samples:
            if is_hard_negative:
                weights.append(hard_negative_weight)  # Higher weight for hard cases
            else:
                weights.append(1.0)
        return weights
```

**Why separate the hard negative flag?**
- During dataset iteration, return only image + label
- But still track which are hard for weighted sampling
- Sampler can oversample hard negatives without changing labels

### ForgeryClassifier Model

```python
class ForgeryClassifier(nn.Module):
    def __init__(self, backbone='efficientnet_b2'):
        super().__init__()
        
        # Backbone: EfficientNet B2 (ImageNet pretrained)
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)
        # num_classes=0 means remove final classification head, keep features
        
        # Classifier head
        num_features = self.backbone.num_features  # Usually 1408 for B2
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 256),           # Reduce dimension
            nn.ReLU(inplace=True),                  # Activate
            nn.Dropout(0.3),                        # Regularization
            nn.Linear(256, 1)                       # Binary output
        )
    
    def forward(self, x):
        # 1. Extract features
        features = self.backbone(x)  # (B, 1408)
        
        # 2. Classify
        logits = self.classifier(features)  # (B, 1)
        
        return logits.squeeze(-1)  # (B,)
```

**Classifier head design**:
- 1408 → 256: Compress learned features to lower dimension
- ReLU: Non-linearity for complex decision boundary
- Dropout(0.3): Prevents overfitting, improves generalization
- 256 → 1: Final binary decision

### Training with Weighted Sampler

```python
# Create dataset
train_dataset = BinaryClassificationDataset(
    train_forged, train_authentic,
    get_transforms(is_train=True),
    hard_negative_paths=train_hard_negatives
)

# Get sample weights (higher for hard negatives)
sample_weights = train_dataset.get_sample_weights(hard_negative_weight=3.0)

# WeightedRandomSampler: samples according to weights
# Hard negatives appear 3x more often in each batch
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

# Create dataloader with sampler
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
```

**Example sampling**:
```
Dataset: [forged1, forged2, ..., auth1, auth2, ..., hard_neg1, hard_neg2]
Weights: [1.0, 1.0, ..., 1.0, 1.0, ..., 3.0, 3.0]

When sampling batch of 16:
- Expect ~8 regular samples (weight 1.0 each)
- Expect ~6-8 hard negatives (weight 3.0 each)
```

### Metrics and Validation

```python
def validate(model, loader, device):
    model.eval()  # Disable dropout, use running batch norm
    
    tp = fp = tn = fn = 0
    
    with torch.no_grad():  # No gradient computation
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = torch.sigmoid(model(images))  # Convert logits to probabilities
            preds = (outputs > 0.5).float()         # Apply threshold
            
            # Compute confusion matrix
            tp += ((preds == 1) & (labels == 1)).sum().item()  # Correct positive
            fp += ((preds == 1) & (labels == 0)).sum().item()  # False positive
            tn += ((preds == 0) & (labels == 0)).sum().item()  # True negative
            fn += ((preds == 0) & (labels == 1)).sum().item()  # False negative
    
    # Compute metrics
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return accuracy, precision, recall, tp, fp, tn, fn
```

**Confusion Matrix Interpretation**:
```
                 Predicted Positive    Predicted Negative
Actual Positive       TP (correct)           FN (miss)
Actual Negative       FP (alarm)             TN (correct)

Precision = TP / (TP + FP)  : Of predictions marked forged, how many correct?
Recall = TP / (TP + FN)     : Of actual forgeries, how many found?
Accuracy = (TP + TN) / Total: Overall correctness
```

---

## Script 3: Detailed Breakdown

**File**: `bin/script_3_train_v4.py`

### Hard Negative Mining Strategy

**Core Insight**: Most false positives come from authentic images that resemble forgeries.

**Process**:
1. Run v3 inference on all authentic test images
2. Find cases where model predicts "forged" (confidence > threshold)
3. These are hard negatives - challenging authentic images
4. Retrain with these cases emphasized (5x weight)
5. Increase FP penalty (2.0 → 4.0)
6. Result: v4 models reject hard negatives better

### HardNegativeDatasetV4

```python
class HardNegativeDatasetV4(Dataset):
    """Dataset with forged images + all hard negative authentic images."""
    
    def __init__(self, forged_paths, mask_paths, hard_negative_paths, 
                 img_size=512, hard_neg_weight=5.0):
        self.samples = []
        self.weights = []
        
        # 1. Forged samples with size-based weighting
        for img_path, mask_path in zip(forged_paths, mask_paths):
            self.samples.append({'image': img_path, 'mask': mask_path, 'is_forged': True})
            
            # Weight by forgery size
            mask = np.load(mask_path)
            forgery_ratio = (mask > 0.5).sum() / mask.size * 100
            
            if forgery_ratio < 1.0:
                self.weights.append(3.0)    # Small is hard
            elif forgery_ratio < 5.0:
                self.weights.append(2.0)    # Medium
            else:
                self.weights.append(1.0)    # Large is easy
        
        # 2. Hard negative samples (very high weight)
        for img_path in hard_negative_paths:
            self.samples.append({'image': img_path, 'mask': None, 'is_forged': False})
            self.weights.append(hard_neg_weight)  # 5.0 - very important!
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = cv2.imread(str(sample['image']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask (or create zero mask for hard negatives)
        if sample['is_forged'] and sample['mask'] is not None:
            mask = np.load(sample['mask'])
            # ... process mask ...
        else:
            mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        
        # Apply augmentations
        transformed = self.transform(image=image, mask=mask)
        
        return image_tensor, mask_tensor
```

**Sampling distribution with weights**:
```
Example: 10k forged, 600 hard negatives
Forged weights: ~15k total (various sizes)
Hard negative weights: 600 * 5.0 = 3000

Total weight: 15k + 3k = 18k
Hard negative fraction: 3000/18000 = 16.7%

Each batch of 4: ~0.67 hard negative samples expected
→ Hard negatives appear frequently, forcing model to learn them
```

### Enhanced Loss Function (v4)

```python
class CombinedLossV4(nn.Module):
    """Enhanced loss with STRONGER FP penalty."""
    
    def __init__(self, focal_weight=0.5, dice_weight=0.5, fp_penalty=4.0):
        super().__init__()
        self.focal = FocalLoss(alpha=0.25, gamma=2.0)
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.fp_penalty = fp_penalty  # 4.0 instead of 2.0
    
    def forward(self, pred, target):
        # Focal loss for hard example focus
        focal_loss = self.focal(pred, target)
        
        # Dice loss for segmentation overlap
        pred_sigmoid = torch.sigmoid(pred)
        intersection = (pred_sigmoid * target).sum()
        dice = (2 * intersection + 1) / (pred_sigmoid.sum() + target.sum() + 1)
        dice_loss = 1 - dice
        
        # FP penalty: STRONGER in v4
        is_authentic = (target.sum(dim=(1,2,3)) == 0).float()
        # This is 1.0 if sample is authentic (sum of target = 0)
        
        fp_loss = (pred_sigmoid * is_authentic.view(-1, 1, 1, 1)).mean() * self.fp_penalty
        
        # v3: 0.5*focal + 0.5*dice + 2.0*fp
        # v4: 0.5*focal + 0.5*dice + 4.0*fp  ← Doubled FP penalty!
        return self.focal_weight * focal_loss + self.dice_weight * dice_loss + fp_loss
```

**Why double the FP penalty?**

```
Example prediction on authentic image:

v3: FP loss = mean(sigmoid(pred)) * 2.0
    If pred_sigmoid ≈ 0.3, loss = 0.3 * 2.0 = 0.6

v4: FP loss = mean(sigmoid(pred)) * 4.0
    If pred_sigmoid ≈ 0.3, loss = 0.3 * 4.0 = 1.2  ← 2x stronger!

This forces the model to predict even lower probabilities on authentic images.
```

### Transfer Learning from v3

```python
def load_model(model_path, in_channels=3):
    """Load pre-trained v3 model."""
    base_model = smp.FPN(...)
    model = AttentionFPN(base_model).to(DEVICE)
    
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model

# In training:
model = load_model(MODELS_DIR / 'highres_no_ela_v3_best.pth')
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
```

**Transfer learning benefits**:
- **Faster convergence**: Model already knows forgery patterns
- **Better final performance**: Fine-tuning usually beats training from scratch
- **Stable training**: Weights don't randomly initialize, reducing variance
- **Lower learning rate**: 1e-4 is fine since weights are pre-trained

---

## Script 4: Detailed Breakdown

**File**: `bin/script_4_two_pass_pipeline.py`

### Two-Pass Architecture

```
Test Image
    ↓
PASS 1: Binary Classifier
├─ Input: Resize to 384×384
├─ Model: EfficientNet B2 + classifier head
├─ Output: Probability 0.0-1.0
└─ Decision:
   ├─ If prob < 0.25: STOP → Return "AUTHENTIC"
   └─ If prob ≥ 0.25: CONTINUE to Pass 2
            ↓
        PASS 2: Ensemble Segmentation
        ├─ Input: Resize to 512×512
        ├─ Models: 4× (FPN + Attention)
        │   ├─ highres_no_ela_v4
        │   ├─ hard_negative_v4
        │   ├─ high_recall_v4
        │   └─ enhanced_aug_v4
        ├─ TTA: 4× (original, H-flip, V-flip, HV-flip)
        ├─ Aggregation: Average + MAX
        ├─ Adaptive Threshold: Based on brightness
        ├─ Component Filtering: Remove noise < 300px
        └─ Output:
           ├─ Binary mask (0/1)
           ├─ Forgery pixels count
           └─ Confidence score
            ↓
        Return "FORGED" (if any pixel marked) or "AUTHENTIC"
```

### Preprocessing Pipeline

```python
def _preprocess_classifier(self, img_path):
    """Prepare image for classifier."""
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to classifier size
    img = cv2.resize(img, (CLASSIFIER_SIZE, CLASSIFIER_SIZE))  # 384×384
    
    # Normalize
    img = img.astype(np.float32) / 255.0  # 0.0-1.0
    
    # ImageNet normalization
    img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    
    # Convert to tensor
    img = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()
    
    return img.to(DEVICE)

def _preprocess_segmentation(self, img_path):
    """Prepare image for segmentation."""
    img = cv2.imread(str(img_path))
    original_size = (img.shape[1], img.shape[0])  # width, height
    
    # Convert to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to segmentation size
    img_resized = cv2.resize(img_rgb, (SEG_SIZE, SEG_SIZE))  # 512×512
    
    # Normalize
    img_norm = img_resized.astype(np.float32) / 255.0
    img_norm = (img_norm - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    
    # Convert to tensor
    tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).float()
    
    return tensor.to(DEVICE), img, original_size
```

**Why two different sizes?**
- Classifier: 384×384 (fast, global features)
- Segmentation: 512×512 (detailed, pixel-level accuracy)

### Test-Time Augmentation (TTA)

```python
def _apply_tta(self, model, img_tensor):
    """Apply 4× TTA with MAX aggregation."""
    preds = []
    
    with torch.no_grad():
        # 1. Original
        pred = torch.sigmoid(model(img_tensor))
        preds.append(pred)
        
        # 2. Horizontal flip (flip left-right)
        pred = torch.sigmoid(model(torch.flip(img_tensor, dims=[3])))  # dims[3] is width
        pred = torch.flip(pred, dims=[3])  # Flip prediction back
        preds.append(pred)
        
        # 3. Vertical flip (flip up-down)
        pred = torch.sigmoid(model(torch.flip(img_tensor, dims=[2])))  # dims[2] is height
        pred = torch.flip(pred, dims=[2])
        preds.append(pred)
        
        # 4. Both flips
        pred = torch.sigmoid(model(torch.flip(img_tensor, dims=[2, 3])))
        pred = torch.flip(pred, dims=[2, 3])
        preds.append(pred)
    
    # Stack and take MAX
    stacked = torch.stack(preds)  # (4, B, 1, H, W)
    result = stacked.max(dim=0)[0]  # (B, 1, H, W)
    
    return result
```

**Why TTA with MAX?**

```
Single prediction on authentic image: [0.1, 0.2, 0.15, 0.25, 0.12, ...]
4× TTA predictions (max over each position):
  Original: [0.1, 0.2, 0.15, 0.25, 0.12]
  H-flip:   [0.09, 0.18, 0.14, 0.24, 0.11]
  V-flip:   [0.11, 0.21, 0.16, 0.26, 0.13]
  HV-flip:  [0.12, 0.22, 0.17, 0.27, 0.14]
  
  MAX:      [0.12, 0.22, 0.17, 0.27, 0.14]  ← Highest from any flip

Effect: Slightly higher predictions but more consistent
(flips might catch forgery at different orientations)
```

**Why MAX instead of MEAN?**
- MAX: Conservative (trusts any model seeing forgery)
- MEAN: Average opinion

For high-recall system, MAX is better.

### Adaptive Thresholding

```python
def _get_adaptive_threshold(self, img_bgr, base_threshold=0.35):
    """Adjust threshold based on image brightness."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)  # 0-255 scale
    
    if brightness < 50:
        # Dark image: forgeries harder to see
        return max(0.20, base_threshold - 0.10)  # Lower threshold: 0.20-0.25
    elif brightness < 80:
        # Slightly dark
        return max(0.25, base_threshold - 0.05)  # 0.25-0.30
    elif brightness > 200:
        # Very bright: easier to see forgeries
        return min(0.50, base_threshold + 0.05)  # Higher threshold: 0.40-0.50
    
    return base_threshold  # Normal: 0.35
```

**Why adaptive?**

```
Dark image (mean=30):
- Forgery regions might be very subtle
- Need lower threshold to catch them
- Risk: Higher false positives

Bright image (mean=220):
- Forgery regions stand out clearly
- Can afford higher threshold
- Benefit: Fewer false positives
```

### Connected Component Filtering

```python
def _remove_small_components(self, mask):
    """Remove regions smaller than min_area."""
    if self.min_area <= 0:
        return mask
    
    # Label connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    #   num_labels: number of components (including background)
    #   labels: image with component IDs
    #   stats: bounding box + area for each component
    
    cleaned = mask.copy()
    
    # Remove components smaller than min_area
    for i in range(1, num_labels):  # Skip 0 (background)
        if stats[i, cv2.CC_STAT_AREA] < self.min_area:
            cleaned[labels == i] = 0  # Erase small component
    
    return cleaned
```

**Example**:
```
Original mask:
  [1, 1, 0, 0]      10 connected pixels (2 components)
  [1, 1, 0, 0]
  [0, 0, 1, 0]      1 isolated pixel (component 3)
  [0, 0, 0, 0]

Component areas: [background: 16, comp2: 4, comp3: 1]
min_area = 300:   All filtered out (all < 300)
Result: All zeros (no forgery detected)
```

### Ensemble Aggregation

```python
# Load 4 segmentation models
all_preds = []
with torch.no_grad():
    for model in self.seg_models:  # 4 models
        if self.use_tta:
            pred = self._apply_tta(model, img_tensor)  # 4× flips
        else:
            pred = torch.sigmoid(model(img_tensor))
        all_preds.append(pred)

# Average predictions from all 4 models
mask = torch.stack(all_preds).mean(dim=0).squeeze().cpu().numpy()
# (4, 1, 512, 512) → (1, 512, 512) → (512, 512)

# Each pixel is now 0.0-1.0 (average of 4 model predictions)
```

**Ensemble benefit**:
```
Example pixel: 4 models predict [0.2, 0.3, 0.25, 0.35]
Average: 0.275

Single model might say "no forgery" (0.2)
Ensemble says "slightly forged" (0.275)
More robust!
```

### Complete Inference Pipeline

```python
def predict(self, img_path):
    """Run full two-pass pipeline."""
    
    result = {
        'image': str(img_path),
        'is_forged': False,
        'classifier_prob': 0.0,
        'passed_to_segmentation': False,
        'seg_max_prob': 0.0,
        'forgery_area': 0,
        'mask': None
    }
    
    # PASS 1: Classifier
    passed, prob = self.run_pass1(img_path)
    result['classifier_prob'] = prob
    
    if not passed:
        # Confident authentic, stop here
        return result
    
    # PASS 2: Segmentation
    result['passed_to_segmentation'] = True
    is_forged, mask, max_prob, area = self.run_pass2(img_path)
    
    result['is_forged'] = is_forged
    result['seg_max_prob'] = max_prob
    result['forgery_area'] = area
    result['mask'] = mask
    
    return result
```

---

## Core Components

### FPN (Feature Pyramid Network)

```python
base_model = smp.FPN(
    encoder_name='timm-efficientnet-b2',
    encoder_weights='imagenet',
    in_channels=3,
    classes=1
)
```

**Architecture**:
```
Input Image (512×512×3)
    ↓
Encoder (EfficientNet B2):
  Layer0: 512×512 → 256×256 (stride 2)
  Layer1: 256×256 → 128×128 (stride 4)
  Layer2: 128×128 → 64×64   (stride 8)
  Layer3: 64×64   → 32×32   (stride 16)
  Layer4: 32×32   → 16×16   (stride 32)
    ↓
Features at multiple scales: 16×16, 32×32, 64×64, 128×128, 256×256
    ↓
Decoder (FPN):
  Upsample and fuse features from multiple scales
    ↓
Output: 512×512×1 (single channel)
```

**Why FPN for segmentation?**
- Multi-scale features capture both large and small forgeries
- Skip connections preserve fine details
- Better for localizing precise boundaries

### EfficientNet B2 Backbone

**Why B2 specifically?**
- B0-B7: Scaling from tiny to massive
- B2: Sweet spot of accuracy vs. speed
  - Parameters: ~9M (small enough for training)
  - Speed: Fast inference (~100ms/image)
  - Accuracy: Good feature extraction

**Compound scaling**:
```
B0: 224×224 images, 1.0× model width/depth
B1: 240×240 images, 1.1× model width/depth
B2: 260×260 images, 1.2× model width/depth
...
B7: 600×600 images, 2.0× model width/depth
```

---

## Supplementary Scripts: Detailed Breakdown

### Generate Forged Augmented Images

**File**: `bin/generate_forged_augmented.py`

#### Purpose
Creates multiple augmented versions of forged images to increase training data and improve robustness. Crucially, applies the **same transformations to both image and mask** to maintain alignment.

#### Flow

```
for each forged image:
    ├─ Load image from train_images/forged
    ├─ Load corresponding mask from train_masks
    ├─ For i in range(NUM_AUGMENTATIONS):
    │  ├─ Apply augmentation (transforms image AND mask identically)
    │  ├─ Skip if forgery cropped out (mask empty)
    │  └─ Save augmented image + mask to output directories
    └─ Report statistics
```

#### Augmentation Pipeline

```python
augmentation = A.Compose([
    # Geometric transforms (affect both image and mask equally)
    A.OneOf([
        A.HorizontalFlip(p=1.0),          # Flip left-right
        A.VerticalFlip(p=1.0),            # Flip up-down
        A.RandomRotate90(p=1.0),          # 90° rotations
        A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT),  # Small rotations
    ], p=0.8),  # 80% chance for at least one
    
    # Scaling/cropping
    A.OneOf([
        A.RandomResizedCrop(scale=(0.8, 1.0)),  # Crop 80-100%, resize back
        A.RandomScale(scale_limit=(-0.2, 0.2)), # Zoom in/out
    ], p=0.3),  # 30% chance
    
    # Color transforms (image only, doesn't affect mask)
    A.OneOf([
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20),
    ], p=0.5),
    
    # Blur/noise (image only)
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 5)),
        A.GaussNoise(std_range=(0.01, 0.02)),
        A.ImageCompression(quality_range=(85, 95)),  # JPEG artifacts
    ], p=0.3),
])
```

**Key Design**: Uses `additional_targets={'mask': 'mask'}` to apply geometric transforms to both image and mask, while color transforms only affect image.

#### Configuration

```python
NUM_AUGMENTATIONS_PER_IMAGE = 3   # Each original → 3 augmented versions
RANDOM_SEED = 42                   # Reproducibility
```

#### Output

```
train_images/forged_augmented/     # 2,751 × 3 = 7,564 images
train_masks_augmented/              # Corresponding masks
```

#### Example Augmentation Process

```
Input: image_001.png (with mask_001.npy)

Augmentation 1:
  - Randomly rotate 10°
  - Increase brightness 15%
  - Result: image_001_aug0.png, mask_001_aug0.npy
  
Augmentation 2:
  - Horizontal flip
  - Gaussian blur
  - Result: image_001_aug1.png, mask_001_aug1.npy
  
Augmentation 3:
  - Random crop 90% and resize
  - Reduce contrast 10%
  - Result: image_001_aug2.png, mask_001_aug2.npy
```

**Critical**: If augmentation crops out the forgery region (mask becomes empty), the sample is skipped to avoid corrupting training data.

---

### Generate Authentic Augmented Images

**File**: `bin/generate_authentic_augmented.py`

#### Purpose
Creates diverse augmented versions of authentic images to:
1. Balance dataset (more authentic samples than forged)
2. Improve model robustness to image variations
3. Reduce overfitting to specific authentic image patterns

#### Flow

```
for each authentic image:
    ├─ Load image
    ├─ For i in range(multiplier):
    │  ├─ Randomly choose augmentation strength (light/medium/strong)
    │  ├─ Apply augmentation pipeline
    │  └─ Save with name: img_id_aug{i}_{strength}.jpg
    └─ Report progress
```

#### Augmentation Strengths

**Light** (mild variations):
```python
A.OneOf([
    A.HorizontalFlip(p=1.0),
    A.VerticalFlip(p=1.0),
    A.Rotate(limit=15, p=1.0),
], p=0.5)
# Color: ±10% brightness/contrast, ±5 hue shift
# 30% chance for any augmentation
```

**Medium** (realistic variations):
```python
A.OneOf([
    A.HorizontalFlip(p=1.0),
    A.VerticalFlip(p=1.0),
    A.Rotate(limit=30, p=1.0),
    A.Affine(scale=(0.9, 1.1), rotate=(-15, 15), p=1.0),
], p=0.6)
# Color: ±15% brightness/contrast, ±10 hue shift
# Blur: Gaussian 3-5px, Gaussian noise, ISO noise
# Compression: JPEG quality 70-95%
# 50% chance for any augmentation
```

**Strong** (heavy variations):
```python
A.OneOf([
    A.HorizontalFlip(p=1.0),
    A.VerticalFlip(p=1.0),
    A.Rotate(limit=45, p=1.0),
    A.Affine(scale=(0.8, 1.2), rotate=(-30, 30), shear=(-10, 10), p=1.0),
    A.Perspective(scale=(0.02, 0.05), p=1.0),  # Perspective distortion
], p=0.7)
# Color: ±20% brightness/contrast, ±15 hue, gamma, CLAHE
# Blur: Gaussian 3-7px, motion blur
# Noise: Gaussian, ISO, multiplicative
# Compression: JPEG 50-90%, downscale/upscale
# 60% chance for any augmentation
```

#### Configuration

```python
parser.add_argument('--multiplier', type=int, default=3)
    # Creates 3 versions per source image
parser.add_argument('--strengths', type=str, default='light,medium,strong')
    # Mix of different augmentation strengths
parser.add_argument('--workers', type=int, default=4)
    # Parallel augmentation for speed
```

#### Output

```
train_images/authentic_augmented/
├── image_1_aug0_light.jpg
├── image_1_aug1_medium.jpg
├── image_1_aug2_strong.jpg
├── image_2_aug0_light.jpg
...
```

**Total**: ~2,400 originals × 3 versions = ~7,200 augmented authentic images

#### Why Multiple Strengths?

```
Light augmentations: Teach invariance to minor variations
  - Helpful for: Rotation, flipping, minor color shifts

Medium augmentations: Realistic real-world variations
  - Helpful for: Lighting changes, compression, blur

Strong augmentations: Boundary cases
  - Helpful for: Extreme lighting, perspective changes, noise
```

---

### Find Hard Negatives

**File**: `bin/find_hard_negatives.py`

#### Purpose
Identifies **false positives** - authentic images that the current model incorrectly predicts as forged. These become "hard negatives" for targeted retraining.

#### Core Concept

```
Hard Negative = Authentic image that model incorrectly detects as forged

Example:
  True label: AUTHENTIC (no forgery)
  Model prediction: FORGED (high confidence)
  → This is a false positive
  → It's "hard" because it fools the model
  → Use it for retraining to teach model to reject it
```

#### Flow

```
for each authentic image in test set:
    ├─ Run two-pass inference (classifier + segmentation)
    ├─ If model predicts "forged":
    │  ├─ This is a false positive (hard negative)
    │  ├─ Record the image ID
    │  └─ Save to output file
    └─ Next image

Output: hard_negative_ids.txt
├─ Line 1: img_001
├─ Line 2: img_045
├─ Line 3: img_789
└─ ...
```

#### Configuration

```python
DEFAULT_CLASSIFIER_THRESHOLD = 0.25  # Threshold for Pass 1
DEFAULT_SEG_THRESHOLD = 0.35         # Threshold for pixel detection
DEFAULT_MIN_AREA = 300               # Minimum forgery region size
```

**These must match inference parameters for meaningful hard negative detection.**

#### Inference in Hard Negative Mining

```python
def run_inference(classifier, seg_models, img_path, classifier_threshold, seg_threshold, min_area):
    # Pass 1: Classifier
    if classifier_prob < classifier_threshold:
        return False  # Classified as authentic, skip Pass 2
    
    # Pass 2: Segmentation ensemble
    # - Load image, apply TTA
    # - Ensemble 4 models (average predictions)
    # - Apply adaptive threshold
    # - Remove small components (< min_area)
    # - Return: is_forged boolean
    
    return is_forged, prob, area
```

#### Hard Negative Statistics

**Example output**:
```
Found 677 hard negatives from 2,377 authentic images
Breakdown by brightness:
  - Dark images (brightness < 50): 173
  - Normal images (brightness 50-200): 504

Distribution by forgery area:
  - < 100 pixels: 234 (false positive tiny region)
  - 100-500 pixels: 389
  - > 500 pixels: 54
```

#### Why Hard Negatives Matter

```
Initial v3 Model Performance:
  - True positive rate (recall): 75%
  - False positive rate: 8%

Problem: 8% of authentic images marked as forged

Root Cause: Certain authentic images look similar to forgery patterns:
  - High contrast edges (natural in some images)
  - Specific color gradients
  - Texture patterns

Solution: Retrain with hard negatives
  Model sees: "These images are authentic, not forged"
  
v4 Model (retrained on hard negatives):
  - True positive rate: 79% (slightly better)
  - False positive rate: 3.4% (much better!)
```

#### Usage

```bash
# Find hard negatives using default thresholds
python bin/find_hard_negatives.py

# Save to custom file
python bin/find_hard_negatives.py --output my_hard_negs.txt

# Use custom thresholds for mining
python bin/find_hard_negatives.py \
    --classifier-threshold 0.30 \
    --seg-threshold 0.40 \
    --min-area 400

# Specify authentic directory
python bin/find_hard_negatives.py \
    --authentic-dir /path/to/authentic/images
```

#### Output File Format

```
hard_negative_ids.txt:
img_001
img_045
img_089
img_234
...
```

Then used by training scripts:
```python
# In script_3_train_v4.py
hard_neg_ids = load_hard_negative_ids()
for img_id in hard_neg_ids:
    img_path = PROJECT_DIR / 'train_images' / 'authentic' / f"{img_id}.png"
    # Add to training with 5x weight
```

---

### Validate Test

**File**: `bin/validate_test.py`

#### Purpose
Evaluates model performance on known test sets (test_authentic_100 and test_forged_100) with detailed metrics and optional parameter sweep.

#### Flow

```
Validation Mode:
for each authentic image:
    ├─ Run two-pass inference
    ├─ If predicted forged: FP (false positive)
    └─ If predicted authentic: TN (true negative)

for each forged image:
    ├─ Run two-pass inference
    ├─ If predicted forged: TP (true positive)
    └─ If predicted authentic: FN (false negative)

Compute metrics:
├─ Accuracy = (TP + TN) / Total
├─ Recall = TP / (TP + FN)
├─ Precision = TP / (TP + FP)
└─ FP Rate = FP / (FP + TN)

Parameter Sweep Mode (--sweep):
for each combination of (cls_threshold, seg_threshold, min_area):
    ├─ Run validation with these parameters
    ├─ Compute metrics
    └─ Track best configuration
```

#### Configuration

```python
DEFAULT_CLASSIFIER_THRESHOLD = 0.20  # Optimized for best net score
DEFAULT_SEG_THRESHOLD = 0.35
DEFAULT_MIN_AREA = 25
```

#### Metrics Explanation

```
Confusion Matrix:
                   Predicted Forged    Predicted Authentic
Actual Forged            TP                    FN
Actual Authentic         FP                    TN

Accuracy:    (TP + TN) / Total
  → Overall correctness, but misleading with imbalance

Recall:      TP / (TP + FN)
  → Of actual forgeries, how many detected?
  → Critical for NOT missing forgeries
  → Want: MAXIMIZED (catch all forgeries)

Precision:   TP / (TP + FP)
  → Of predictions marked forged, how many correct?
  → Critical for NOT over-detecting
  → Want: HIGH (don't flag innocent images)

FP Rate:     FP / (FP + TN)
  → Of authentic images, what % marked forged?
  → Want: MINIMIZED

Net Score:   TP - FP (or weighted combination)
  → Balance metric for competition
```

#### Parameter Sweep

**Purpose**: Find optimal threshold combination.

```bash
python bin/validate_test.py --sweep
```

**Output**:
```
Cls    Seg    Area   TP   FP   Net     Recall   FP Rate  Precision
0.05   0.15   25     87   18   69      87.0%    18.0%    82.9%
0.05   0.15   50     86   16   70      86.0%    16.0%    84.3%
...
0.20   0.35   25     88   10   78 ✓    88.0%    10.0%    89.8%  ← BEST
...
0.30   0.50   300    85   2    83      85.0%    2.0%     97.7%
```

**Sweep Ranges**:
```python
cls_thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
seg_thresholds = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
min_areas = [25, 50, 100, 150, 200, 300]
# Total: 6 × 6 × 6 = 216 combinations
# Each tests on 100 authentic + 100 forged = 200 images
```

#### Usage

```bash
# Single validation with defaults
python bin/validate_test.py

# With custom thresholds
python bin/validate_test.py \
    --classifier-threshold 0.25 \
    --seg-threshold 0.40 \
    --min-area 300

# Parameter sweep (slow, ~10-15 minutes)
python bin/validate_test.py --sweep
```

---

### Full Pipeline Orchestration

**File**: `bin/full_pipeline.py`

#### Purpose
Orchestrates the complete workflow from training to inference, managing all phases and hard negative mining.

#### Pipeline Phases

```
FULL PIPELINE:

Phase 1: Initial Training (2-3 hours)
├─ Train segmentation ensemble (v3 or v4)
│  └─ 4 models: highres_no_ela, hard_negative, high_recall, enhanced_aug
└─ Train binary classifier
   └─ Fast forged vs authentic filter

Phase 2: Hard Negative Mining (5-15 min)
├─ Run inference on authentic test images
├─ Identify false positives (hard negatives)
└─ Save hard_negative_ids.txt

Phase 3: Retraining with Hard Negatives (3-4 hours)
├─ Retrain segmentation ensemble
│  └─ Now with hard negatives included (5x weight)
│  └─ Produces v4 models
└─ Retrain binary classifier
   └─ With hard negatives included

Phase 4: Inference (5-20 min)
├─ Run two-pass pipeline on test set
├─ Generate predictions and masks
└─ Create submission.csv
```

#### Architecture

```python
class PipelineLogger:
    """Unified logging to console and file."""
    log() -> Writes to console and log file
    section() -> Pretty separator for phase headers
    error() -> Logs errors
    warning() -> Logs warnings

def run_script() -> Executes Python scripts with args

def phase1_initial_training() -> Trains initial models

def phase2_hard_negative_mining() -> Finds hard negatives

def phase3_retrain_with_hard_negatives() -> Retrains with hard negs

def phase4_inference() -> Runs predictions
```

#### Control Options

```bash
# Full pipeline (all phases)
python full_pipeline.py

# Skip initial training (models exist)
python full_pipeline.py --skip-initial

# Only inference
python full_pipeline.py --inference-only

# Only specific phase
python full_pipeline.py --phase 2  # Only mining

# Custom parameters
python full_pipeline.py \
    --classifier-threshold 0.30 \
    --seg-threshold 0.40 \
    --min-area 300

# Custom directories
python full_pipeline.py \
    --authentic-dir /path/to/test/images \
    --test-dir /path/to/test/images \
    --output-dir /path/to/output
```

#### Workflow Example

```bash
# Day 1: Complete initial training and mining
python full_pipeline.py --skip-retrain  # Phases 1-2

# Monitor: tail -f bin/logs/full_pipeline_*.log

# After Phase 2 completes, verify hard negatives:
cat hard_negative_ids.txt | wc -l  # How many found?

# Day 2: Retrain with hard negatives + inference
python full_pipeline.py --skip-initial --skip-mining  # Phases 3-4

# Check results
cat submission.csv | head -20
```

#### Log File Format

```
[2025-01-15 10:00:00] [INFO] ======================================================================
[2025-01-15 10:00:00] [INFO]  FULL TRAINING PIPELINE WITH HARD NEGATIVE MINING
[2025-01-15 10:00:00] [INFO] ======================================================================
[2025-01-15 10:00:00] [INFO] Project directory: /home/.../ScientificImageForgeryDetection
[2025-01-15 10:00:00] [INFO] 
[2025-01-15 10:00:00] [INFO] ======================================================================
[2025-01-15 10:00:00] [INFO]  PHASE 1: INITIAL MODEL TRAINING
[2025-01-15 10:00:00] [INFO] ======================================================================
[2025-01-15 10:00:05] [INFO] Running: python bin/script_1_train_v3.py --no-hard-negatives
[2025-01-15 10:00:10] [INFO] Initial segmentation training complete.
...
```

#### Error Handling

```python
if phase_fails:
    logger.error("Phase X failed!")
    
    if running_full_pipeline:
        sys.exit(1)  # Stop pipeline
    else:
        continue     # Continue with other phases if --phase specified
```

#### Key Design Decisions

1. **Phases are Sequential**: Each phase depends on previous
   - Can't do Phase 3 without Phase 1
   - Can skip phases with flags (e.g., --skip-initial)

2. **Flexible Execution**: Fine-grained control
   - Run full pipeline: `python full_pipeline.py`
   - Skip to mining: `python full_pipeline.py --skip-initial`
   - Only inference: `python full_pipeline.py --inference-only`

3. **Logging**: Dual output to console and file
   - Real-time monitoring: `tail -f bin/logs/full_pipeline_*.log`
   - Post-run analysis: Review log file

4. **Error Recovery**: Graceful degradation
   - Hard negative mining failure doesn't stop pipeline
   - Can rerun individual phases

---

## Mathematical Foundations

### Binary Cross-Entropy Loss

$$BCE = -[y \log(p) + (1-y) \log(1-p)]$$

Where:
- $y$ = target (0 or 1)
- $p$ = predicted probability (0-1)

**Interpretation**:
- If $y=1$ (true positive): loss = $-\log(p)$
  - $p=0.9$: loss ≈ 0.105 (small)
  - $p=0.1$: loss ≈ 2.303 (large)
- If $y=0$ (true negative): loss = $-\log(1-p)$
  - $p=0.1$: loss ≈ 0.105 (small)
  - $p=0.9$: loss ≈ 2.303 (large)

### Dice Coefficient

$$Dice = \frac{2|A \cap B|}{|A| + |B|}$$

Where:
- $A$ = predicted positive pixels
- $B$ = ground truth positive pixels
- $A \cap B$ = overlap

**Range**: 0 (no overlap) to 1 (perfect overlap)

**Example**:
```
True mask:  [1, 1, 0, 0]
Pred mask:  [1, 0.9, 0.1, 0]

Intersection: min(1,1) + min(1,0.9) + min(0,0.1) + min(0,0) = 1 + 0.9 = 1.9
Sum of masks: (1+1+0+0) + (1+0.9+0.1+0) = 2 + 2 = 4
Dice = 2×1.9/4 = 0.95
```

### Weighted Random Sampling

**Probability of selecting sample $i$**:

$$P(i) = \frac{w_i}{\sum_j w_j}$$

Where $w_i$ is the weight of sample $i$.

**Example** with replacement in epoch:
```
Dataset: 3 samples with weights [1, 1, 5]
Total weight: 7

Probabilities: [1/7≈14%, 1/7≈14%, 5/7≈71%]

In batch of 100:
Expected: 14, 14, 71 samples respectively
Actual: Some variation due to randomness
```

---

## Performance Analysis

### Inference Timing

**Pass 1 (Classifier)**:
```
Preprocessing: 5ms
Forward pass: 3ms
Sigmoid + threshold: 1ms
Total: ~10ms
```

**Pass 2 (Segmentation)**:
```
Preprocessing: 5ms
Model 1 (no TTA): 150ms
Model 2 (no TTA): 150ms
Model 3 (no TTA): 150ms
Model 4 (no TTA): 150ms
Ensemble average: 2ms
Thresholding: 3ms
Component filtering: 5ms
Total (no TTA): ~615ms

With 4× TTA: 600ms × 4 = 2400ms (6× slower)
```

**Expected per-image throughput**:
```
Without TTA:
- 50% stop at Pass 1 (10ms): average 10ms
- 50% pass to Pass 2 (625ms): average 625ms
- Overall: 317.5ms/image

With TTA:
- 50% stop at Pass 1 (10ms): average 10ms
- 50% pass to Pass 2 with TTA (2410ms): average 2410ms
- Overall: 1210ms/image
```

---

**Last Updated**: December 2025  
**Status**: Complete technical reference
