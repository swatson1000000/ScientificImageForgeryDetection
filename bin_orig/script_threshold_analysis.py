"""
Threshold Analysis Script
Tests different decision thresholds to optimize F1, precision, and recall
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
from pathlib import Path
import json
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

# ============================================================================
# CONFIGURATION
# ============================================================================

DATASET_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_FORGED_PATH = os.path.join(DATASET_PATH, "test_forged_100")
TEST_AUTHENTIC_PATH = os.path.join(DATASET_PATH, "test_authentic_100")
MODELS_PATH = os.path.join(DATASET_PATH, "models")
LOG_PATH = os.path.join(DATASET_PATH, "log")
OUTPUT_PATH = os.path.join(DATASET_PATH, "output")

os.makedirs(LOG_PATH, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)

IMG_SIZE = 256
BATCH_SIZE = 8
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = os.path.join(MODELS_PATH, 'attention_unet_focal_loss_best.pth')

# ============================================================================
# MODEL ARCHITECTURE
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
        e1 = self.Conv1(x)
        e2 = self.Maxpool(e1)
        e2 = self.Conv2(e2)
        e3 = self.Maxpool(e2)
        e3 = self.Conv3(e3)
        e4 = self.Maxpool(e3)
        e4 = self.Conv4(e4)
        e5 = self.Maxpool(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        e4 = self.AttGate5(d5, e4)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.UpConv5(d5)

        d4 = self.Up4(d5)
        e3 = self.AttGate4(d4, e3)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.UpConv4(d4)

        d3 = self.Up3(d4)
        e2 = self.AttGate3(d3, e2)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.UpConv3(d3)

        d2 = self.Up2(d3)
        e1 = self.AttGate2(d2, e1)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.UpConv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


# ============================================================================
# DATASET
# ============================================================================

class TestDataset(Dataset):
    def __init__(self, image_dir, label, img_size=256):
        self.image_dir = image_dir
        self.label = label
        self.img_size = img_size
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read {img_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.img_size, self.img_size))
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        
        return image, self.label, img_name


# ============================================================================
# INFERENCE WITH THRESHOLD ANALYSIS
# ============================================================================

def analyze_thresholds():
    print("="*80)
    print("THRESHOLD ANALYSIS")
    print("="*80)
    
    # Load model
    model = AttentionUNet(img_ch=3, output_ch=1).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    # Load datasets
    forged_dataset = TestDataset(TEST_FORGED_PATH, label=1)
    authentic_dataset = TestDataset(TEST_AUTHENTIC_PATH, label=0)
    
    forged_loader = DataLoader(forged_dataset, batch_size=BATCH_SIZE, shuffle=False)
    authentic_loader = DataLoader(authentic_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Get predictions for all images
    all_predictions = []
    all_labels = []
    all_filenames = []
    
    print("\nProcessing forged images...")
    with torch.no_grad():
        for images, labels, filenames in forged_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            predictions = torch.sigmoid(outputs).cpu().numpy()
            
            # Average across spatial dimensions to get image-level score
            for i in range(predictions.shape[0]):
                score = predictions[i].mean()  # Average the mask
                all_predictions.append(score)
                all_labels.append(1)  # Forged
                all_filenames.append(filenames[i])
    
    print(f"Processed {len([l for l in all_labels if l == 1])} forged images")
    
    print("Processing authentic images...")
    with torch.no_grad():
        for images, labels, filenames in authentic_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            predictions = torch.sigmoid(outputs).cpu().numpy()
            
            for i in range(predictions.shape[0]):
                score = predictions[i].mean()
                all_predictions.append(score)
                all_labels.append(0)  # Authentic
                all_filenames.append(filenames[i])
    
    print(f"Processed {len([l for l in all_labels if l == 0])} authentic images")
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Test different thresholds
    thresholds = np.arange(0.1, 1.0, 0.05)
    results = []
    
    print("\n" + "="*80)
    print("THRESHOLD ANALYSIS RESULTS")
    print("="*80)
    print(f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Accuracy':<12}")
    print("-"*60)
    
    for threshold in thresholds:
        predictions = (all_predictions >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, predictions, average='binary', zero_division=0
        )
        accuracy = np.mean(predictions == all_labels)
        
        results.append({
            'threshold': float(threshold),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'accuracy': float(accuracy)
        })
        
        print(f"{threshold:<12.2f} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {accuracy:<12.4f}")
    
    # Find best thresholds
    best_f1 = max(results, key=lambda x: x['f1'])
    best_precision = max(results, key=lambda x: x['precision'])
    best_recall = max(results, key=lambda x: x['recall'])
    
    print("\n" + "="*80)
    print("BEST THRESHOLDS")
    print("="*80)
    print(f"Best F1 (threshold={best_f1['threshold']:.2f}): F1={best_f1['f1']:.4f}, Precision={best_f1['precision']:.4f}, Recall={best_f1['recall']:.4f}")
    print(f"Best Precision (threshold={best_precision['threshold']:.2f}): Precision={best_precision['precision']:.4f}, F1={best_precision['f1']:.4f}")
    print(f"Best Recall (threshold={best_recall['threshold']:.2f}): Recall={best_recall['recall']:.4f}, F1={best_recall['f1']:.4f}")
    
    # Save results
    output_file = os.path.join(OUTPUT_PATH, 'threshold_analysis.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")
    
    return results, all_predictions, all_labels


if __name__ == "__main__":
    analyze_thresholds()
