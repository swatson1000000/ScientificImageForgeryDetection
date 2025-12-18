"""
Test INDIVIDUAL models on validation images WITH ground truth masks
This will show us actual pixel-level F1 performance per model
"""

import os
import numpy as np
import torch
import torch.nn as nn
import cv2

# ============================================================================
# CONFIGURATION
# ============================================================================

DATASET_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VAL_IMAGES_PATH = os.path.join(DATASET_PATH, "validation_images")
VAL_MASKS_PATH = os.path.join(DATASET_PATH, "validation_masks")
MODELS_PATH = os.path.join(DATASET_PATH, "models")

OUTPUT_SIZE = 256
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
            model.load_state_dict(fixed_state_dict, strict=False)
        
        print(f"✓ Loaded: {os.path.basename(model_path)}")
    except Exception as e:
        print(f"✗ Failed to load {model_path}: {e}")
        return None
    
    model.eval()
    return model


def calculate_metrics(pred, target, threshold=0.5):
    """Calculate pixel-level metrics"""
    pred_binary = (pred > threshold).astype(np.float32)
    target_binary = (target > 0.5).astype(np.float32)
    
    tp = np.sum(pred_binary * target_binary)
    fp = np.sum(pred_binary * (1 - target_binary))
    fn = np.sum((1 - pred_binary) * target_binary)
    tn = np.sum((1 - pred_binary) * (1 - target_binary))
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    
    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'iou': iou,
    }


def main():
    print("\n" + "="*80)
    print("TEST: INDIVIDUAL MODELS ON VALIDATION DATA")
    print("="*80)
    print(f"Device: {DEVICE}")
    
    # Models to test
    MODELS_TO_TEST = {
        'SegFormer (script_1)': ('segformer_b2_best.pth', 'segformer', 512),
        'Lovász (script_2)': ('attention_unet_lovasz_best.pth', 'attention_unet', 256),
        'SimCLR (script_4)': ('attention_unet_simclr_pretrained_best.pth', 'attention_unet', 256),
        'Ensemble_0': ('ensemble_0_best.pth', 'attention_unet', 256),
    }
    
    # Get validation files
    val_files = sorted([f for f in os.listdir(VAL_IMAGES_PATH) if f.endswith('.png')])[:100]
    print(f"\nTesting on {len(val_files)} validation images...")
    
    # Test each model individually
    for model_name, (model_file, model_type, img_size) in MODELS_TO_TEST.items():
        model_path = os.path.join(MODELS_PATH, model_file)
        if not os.path.exists(model_path):
            print(f"\n✗ Missing: {model_name}")
            continue
            
        print(f"\n" + "="*60)
        print(f"Testing: {model_name}")
        print("="*60)
        
        model = load_model(model_path, model_type)
        if model is None:
            continue
        
        # Collect metrics across all thresholds
        all_metrics = {thresh: {'f1': [], 'precision': [], 'recall': [], 'iou': []} 
                       for thresh in [0.3, 0.4, 0.5]}
        
        pred_values = []  # To check prediction distribution
        
        with torch.no_grad():
            for idx, filename in enumerate(val_files):
                img_path = os.path.join(VAL_IMAGES_PATH, filename)
                mask_filename = filename.replace('.png', '.npy')
                mask_path = os.path.join(VAL_MASKS_PATH, mask_filename)
                
                image_np = cv2.imread(img_path)
                if image_np is None or not os.path.exists(mask_path):
                    continue
                
                mask_np = np.load(mask_path)
                if mask_np.ndim == 3:
                    mask_np = np.max(mask_np, axis=0)
                mask_resized = cv2.resize(mask_np.astype(np.float32), (OUTPUT_SIZE, OUTPUT_SIZE))
                
                # Get model prediction
                image_tensor = prepare_image_for_model(image_np, img_size)
                image_tensor = image_tensor.unsqueeze(0).to(DEVICE)
                
                pred = model(image_tensor).cpu().numpy()[0].squeeze()
                if pred.shape[0] != OUTPUT_SIZE:
                    pred = cv2.resize(pred, (OUTPUT_SIZE, OUTPUT_SIZE))
                
                pred_values.extend(pred.flatten().tolist())
                
                # Calculate metrics
                for thresh in [0.3, 0.4, 0.5]:
                    metrics = calculate_metrics(pred, mask_resized, threshold=thresh)
                    for key in ['f1', 'precision', 'recall', 'iou']:
                        all_metrics[thresh][key].append(metrics[key])
        
        # Print results
        print(f"\nPrediction stats: min={min(pred_values):.4f}, max={max(pred_values):.4f}, mean={np.mean(pred_values):.4f}")
        
        for thresh in [0.3, 0.4, 0.5]:
            f1_mean = np.mean(all_metrics[thresh]['f1'])
            prec_mean = np.mean(all_metrics[thresh]['precision'])
            rec_mean = np.mean(all_metrics[thresh]['recall'])
            print(f"Threshold {thresh}: F1={f1_mean:.4f}, Prec={prec_mean:.4f}, Rec={rec_mean:.4f}")
    
    print("\n" + "="*80)
    print("DONE")
    print("="*80)


if __name__ == "__main__":
    main()
