"""
Test Script 9 on 100 forged training images
This will show us how the ensemble classifier performs on known forged images
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import json
from pathlib import Path
import time
from datetime import datetime

import rle_utils as rle_compress

# ============================================================================
# CONFIGURATION
# ============================================================================

DATASET_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_IMAGES_PATH = os.path.join(DATASET_PATH, "test_forged_100")
MODELS_PATH = os.path.join(DATASET_PATH, "models")
LOG_PATH = os.path.join(DATASET_PATH, "log")
OUTPUT_PATH = os.path.join(DATASET_PATH, "output")

# Ensure directories exist
os.makedirs(LOG_PATH, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)

OUTPUT_SIZE = 256  # Common output size for ensemble averaging
BATCH_SIZE = 4     # Reduced for larger images
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Models to include in ensemble - scripts 1, 2, 4, 5
ENSEMBLE_MODELS = {
    'script_1_segformer': os.path.join(MODELS_PATH, 'segformer_b2_best.pth'),
    'script_2_lovasz': os.path.join(MODELS_PATH, 'attention_unet_lovasz_best.pth'),
    'script_4_simclr': os.path.join(MODELS_PATH, 'attention_unet_simclr_pretrained_best.pth'),
    'script_5_ensemble_0': os.path.join(MODELS_PATH, 'ensemble_0_best.pth'),
    'script_5_ensemble_1': os.path.join(MODELS_PATH, 'ensemble_1_best.pth'),
    'script_5_ensemble_2': os.path.join(MODELS_PATH, 'ensemble_2_best.pth'),
    'script_5_ensemble_3': os.path.join(MODELS_PATH, 'ensemble_3_best.pth'),
}

# Model weights based on validation F1 scores
MODEL_WEIGHTS = {
    'script_1_segformer': 3.0,      # Best: F1=0.57
    'script_2_lovasz': 2.0,         # Good: F1=0.47
    'script_4_simclr': 1.5,         # Good: F1=0.45
    'script_5_ensemble_0': 1.5,     # Moderate
    'script_5_ensemble_1': 1.5,
    'script_5_ensemble_2': 1.5,
    'script_5_ensemble_3': 1.5,
}

# Image sizes for each model (based on training)
MODEL_IMG_SIZES = {
    'script_1_segformer': 512,      # Script 1 trained on 512x512
    'script_2_lovasz': 256,         # Script 2 trained on 256x256
    'script_4_simclr': 256,         # Script 4 trained on 256x256
    'script_5_ensemble_0': 256,     # Script 5 trained on 256x256
    'script_5_ensemble_1': 256,
    'script_5_ensemble_2': 256,
    'script_5_ensemble_3': 256,
}

# ============================================================================
# ARCHITECTURES
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
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
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
        self.Att5 = AttentionGate(filters[3], filters[3], filters[2])
        self.UpConv5 = ConvBlock(filters[4], filters[3])

        self.Up4 = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=2, stride=2)
        self.Att4 = AttentionGate(filters[2], filters[2], filters[1])
        self.UpConv4 = ConvBlock(filters[3], filters[2])

        self.Up3 = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=2, stride=2)
        self.Att3 = AttentionGate(filters[1], filters[1], filters[0])
        self.UpConv3 = ConvBlock(filters[2], filters[1])

        self.Up2 = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=2, stride=2)
        self.Att2 = AttentionGate(filters[0], filters[0], filters[0]//2)
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
        x4 = self.Att5(d5, x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.UpConv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(d4, x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.UpConv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(d3, x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.UpConv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(d2, x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.UpConv2(d2)

        out = self.Conv_1x1(d2)
        return torch.sigmoid(out)


class SegFormer(nn.Module):
    def __init__(self, num_classes=1):
        super(SegFormer, self).__init__()
        from transformers import SegformerForSemanticSegmentation
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b2-finetuned-ade-512-512",
            num_labels=1,
            ignore_mismatched_sizes=True
        )

    def forward(self, x):
        outputs = self.model(x)
        logits = outputs.logits
        logits = torch.nn.functional.interpolate(
            logits, size=x.shape[-2:], mode='bilinear', align_corners=False
        )
        return torch.sigmoid(logits)


class TestDataset(Dataset):
    """Dataset that returns images at original size (will be resized per-model)"""
    def __init__(self, image_dir):
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        self.image_dir = image_dir

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = cv2.imread(img_path)
        
        if image is None:
            print(f"Warning: Could not load {img_path}")
            image = np.zeros((256, 256, 3), dtype=np.uint8)
        
        # Return as numpy array, will resize per-model
        return {
            'image': image,
            'filename': self.image_files[idx]
        }


def prepare_image_for_model(image_np, target_size):
    """Resize image, normalize with ImageNet stats, and convert to tensor"""
    # Convert BGR to RGB (cv2 reads BGR, but training uses RGB)
    image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    
    # Resize
    image = cv2.resize(image, (target_size, target_size))
    image = image.astype(np.float32) / 255.0
    
    # Apply ImageNet normalization (same as training)
    image[..., 0] = (image[..., 0] - 0.485) / 0.229
    image[..., 1] = (image[..., 1] - 0.456) / 0.224
    image[..., 2] = (image[..., 2] - 0.406) / 0.225
    
    image = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(image)


def load_model(model_path, model_type='attention_unet'):
    """Load a trained model with flexible state dict loading"""
    if model_type == 'segformer':
        model = SegFormer().to(DEVICE)
    else:
        model = AttentionUNet(img_ch=3, output_ch=1).to(DEVICE)
    
    try:
        checkpoint = torch.load(model_path, map_location=DEVICE)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            fixed_state_dict = {}
            for key, value in state_dict.items():
                new_key = key.replace('AttGate', 'Att')
                fixed_state_dict[new_key] = value
            
            try:
                model.load_state_dict(fixed_state_dict, strict=False)
            except RuntimeError:
                model.load_state_dict(fixed_state_dict, strict=False)
        
        print(f"✓ Loaded: {os.path.basename(model_path)}")
    except Exception as e:
        print(f"✗ Failed to load {model_path}: {e}")
        return None
    
    model.eval()
    return model


def get_ensemble_predictions(test_dataset):
    """Generate ensemble predictions by running each model at its native resolution"""
    
    print("\n" + "="*80)
    print("LOADING ENSEMBLE MODELS")
    print("="*80)
    
    models = {}
    
    for name, path in ENSEMBLE_MODELS.items():
        if not os.path.exists(path):
            print(f"✗ Missing: {name} ({path})")
            continue
        
        model_type = 'segformer' if 'segformer' in name else 'attention_unet'
        model = load_model(path, model_type)
        if model is not None:
            models[name] = model
    
    print(f"\n✓ Loaded {len(models)} models")
    
    print("\n" + "="*80)
    print("GENERATING ENSEMBLE PREDICTIONS (per-model native resolution)")
    print("="*80)
    
    all_predictions = {}
    
    with torch.no_grad():
        for idx in range(len(test_dataset)):
            sample = test_dataset[idx]
            image_np = sample['image']
            filename = sample['filename']
            
            model_predictions = []
            
            for model_name, model in models.items():
                try:
                    # Get native size for this model
                    img_size = MODEL_IMG_SIZES.get(model_name, 256)
                    
                    # Prepare image at model's native resolution
                    image_tensor = prepare_image_for_model(image_np, img_size)
                    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)
                    
                    # Run inference
                    pred = model(image_tensor)
                    pred = pred.cpu().numpy()[0]  # [1, H, W] -> [H, W] or [1, 1, H, W] -> [1, H, W]
                    
                    # Resize prediction to common output size
                    if pred.shape[-1] != OUTPUT_SIZE or pred.shape[-2] != OUTPUT_SIZE:
                        pred_squeezed = pred.squeeze()  # Remove channel dims if present
                        pred_resized = cv2.resize(pred_squeezed, (OUTPUT_SIZE, OUTPUT_SIZE))
                        pred = pred_resized[np.newaxis, :, :]  # Add channel back [1, H, W]
                    
                    # No weight scaling with MAX pooling
                    model_predictions.append(pred)
                    
                except Exception as e:
                    print(f"Error with {model_name} on {filename}: {e}")
                    continue
            
            if model_predictions:
                # MAX pooling across models (better than weighted average!)
                # If ANY model confidently detects forgery, flag the region
                ensemble_pred = np.max(model_predictions, axis=0)
                all_predictions[filename] = ensemble_pred
            
            if (idx + 1) % 20 == 0:
                print(f"  Processed {idx + 1}/{len(test_dataset)} images")
    
    return all_predictions


def main():
    script_start_time = datetime.now()
    script_start = time.time()
    
    print("\n" + "="*80)
    print("TEST: ENSEMBLE INFERENCE ON 100 FORGED TRAINING IMAGES")
    print("="*80)
    print(f"START TIME: {script_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {DEVICE}")
    print(f"Output size: {OUTPUT_SIZE}x{OUTPUT_SIZE}")
    print(f"Model sizes: SegFormer=512, Others=256")
    
    test_dataset = TestDataset(TEST_IMAGES_PATH)
    
    print(f"\nTest images: {len(test_dataset)}")
    
    predictions = get_ensemble_predictions(test_dataset)
    
    print(f"\nGenerated predictions for {len(predictions)} images")
    
    # Analyze predictions
    all_preds = np.concatenate([v.flatten() for v in predictions.values()])
    
    print("\n" + "="*80)
    print("PREDICTION ANALYSIS ON KNOWN FORGED IMAGES")
    print("="*80)
    print(f"Prediction statistics:")
    print(f"  Min: {all_preds.min():.4f}")
    print(f"  Max: {all_preds.max():.4f}")
    print(f"  Mean: {all_preds.mean():.4f}")
    print(f"  Std: {all_preds.std():.4f}")
    print(f"  Median: {np.median(all_preds):.4f}")
    
    # Test using MAX PREDICTION threshold (much better than area-based)
    print(f"\n" + "="*80)
    print("CLASSIFICATION USING MAX PREDICTION THRESHOLD")
    print("(If ANY pixel is confidently forged, classify image as forged)")
    print("="*80)
    
    thresholds_to_test = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    for threshold in thresholds_to_test:
        forged_count = 0
        authentic_count = 0
        max_preds = []
        
        for filename, pred in predictions.items():
            max_pred = pred.max()
            max_preds.append(max_pred)
            
            # Use MAX prediction threshold
            if max_pred > threshold:
                forged_count += 1
            else:
                authentic_count += 1
        
        max_preds_array = np.array(max_preds)
        print(f"\nThreshold {threshold:.2f}:")
        print(f"  - Avg max prediction: {max_preds_array.mean():.4f}")
        print(f"  - Median max prediction: {np.median(max_preds_array):.4f}")
        print(f"  - Classified as forged: {forged_count}/100 ({forged_count}%)")
        print(f"  - Classified as authentic: {authentic_count}/100 ({authentic_count}%)")
    
    # Generate submission file using MAX prediction threshold
    print("\n" + "="*80)
    print("GENERATING SUBMISSION FILE (using MAX prediction threshold)")
    print("="*80)
    
    max_threshold = 0.3  # Optimal for image classification (F1=0.865)
    pixel_threshold = 0.5  # For generating the mask
    submission_rows = []
    forged_count = 0
    authentic_count = 0
    
    for filename, pred in predictions.items():
        max_pred = pred.max()
        image_id = filename.replace('.png', '')
        
        # Classify based on max prediction
        if max_pred > max_threshold:
            # Generate mask for RLE encoding
            mask = (pred.squeeze() > pixel_threshold).astype(np.uint8)
            rle = rle_compress.encode_rle(mask)
            submission_rows.append({'case_id': image_id, 'annotation': rle})
            forged_count += 1
        else:
            submission_rows.append({'case_id': image_id, 'annotation': 'authentic'})
            authentic_count += 1
    
    import pandas as pd
    df = pd.DataFrame(submission_rows)
    submission_path = os.path.join(OUTPUT_PATH, 'submission_forged_100.csv')
    df.to_csv(submission_path, index=False)
    
    print(f"\n✓ Submission saved: {submission_path}")
    print(f"  - Total images: {len(df)}")
    print(f"  - Images marked as forged: {forged_count} ({forged_count/len(df)*100:.1f}%)")
    print(f"  - Images marked as authentic: {authentic_count} ({authentic_count/len(df)*100:.1f}%)")
    print(f"  - Max prediction threshold: {max_threshold}")
    print(f"  - Pixel threshold for mask: {pixel_threshold}")
    
    script_elapsed = time.time() - script_start
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
    print(f"ELAPSED TIME: {script_elapsed:.2f} seconds ({script_elapsed/60:.2f} minutes)")
    print("="*80)


if __name__ == "__main__":
    main()
