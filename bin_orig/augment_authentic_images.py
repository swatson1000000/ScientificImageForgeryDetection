"""
Augment Authentic Images
========================

This script creates augmented versions of authentic images to balance
the training dataset. With only 100 authentic vs 2,751 forged images,
the class imbalance causes models to predict "forged" too often (low precision).

Target: Generate ~2,700 augmented authentic images for a 1:1 ratio.

Augmentations used (that preserve "authentic" nature):
- Random crops and resizes
- Horizontal/vertical flips
- Color jitter (brightness, contrast, saturation)
- JPEG compression artifacts (realistic)
- Gaussian blur (slight)
- Random rotation
- Gaussian noise

Usage:
    python bin/augment_authentic_images.py [--target_count 2700] [--seed 42]
"""

import os
import sys
import argparse
import numpy as np
import cv2
from pathlib import Path
import albumentations as A
from tqdm import tqdm
import random


def create_augmentation_pipeline(strength='medium'):
    """
    Create augmentation pipeline for authentic images.
    
    These augmentations are chosen to:
    1. Preserve the "authentic" nature of images
    2. Add realistic variations (compression, noise, blur)
    3. NOT introduce forgery-like artifacts
    """
    
    if strength == 'light':
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
        ])
    
    elif strength == 'medium':
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=1.0),
            ], p=0.5),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.GaussNoise(var_limit=(5.0, 30.0), p=1.0),
            ], p=0.3),
            A.ImageCompression(quality_lower=70, quality_upper=95, p=0.3),
        ])
    
    else:  # strong
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=15, p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
                A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=30, p=1.0),
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
            ], p=0.6),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3), p=1.0),
            ], p=0.4),
            A.ImageCompression(quality_lower=60, quality_upper=95, p=0.4),
            A.RandomCrop(height=224, width=224, p=0.2),
        ])


def augment_and_save(image_path, output_dir, num_augments, transform, start_idx):
    """
    Load an image, apply augmentations, and save augmented versions.
    """
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Warning: Could not load {image_path}")
        return 0
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_h, original_w = image.shape[:2]
    
    saved_count = 0
    base_name = image_path.stem
    
    for i in range(num_augments):
        # Apply augmentation
        augmented = transform(image=image)
        aug_image = augmented['image']
        
        # If RandomCrop was applied and image is smaller, resize back
        if aug_image.shape[0] != original_h or aug_image.shape[1] != original_w:
            aug_image = cv2.resize(aug_image, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        
        # Convert back to BGR for saving
        aug_image = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
        
        # Save with unique name
        output_name = f"{base_name}_aug_{start_idx + i:04d}.png"
        output_path = output_dir / output_name
        cv2.imwrite(str(output_path), aug_image)
        saved_count += 1
    
    return saved_count


def main():
    parser = argparse.ArgumentParser(description='Augment authentic images for training')
    parser.add_argument('--target_count', type=int, default=2700,
                        help='Target number of total authentic images (original + augmented)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--strength', type=str, default='medium',
                        choices=['light', 'medium', 'strong'],
                        help='Augmentation strength')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for augmented images')
    args = parser.parse_args()
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Paths
    home_dir = Path('/home/swatson/work/MachineLearning/kaggle/ScientificImageForgeryDetection')
    authentic_dir = home_dir / 'test_authentic_100'
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = home_dir / 'train_images' / 'authentic_augmented'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of authentic images
    authentic_images = list(authentic_dir.glob('*.png')) + list(authentic_dir.glob('*.jpg'))
    num_original = len(authentic_images)
    
    print(f"=" * 60)
    print(f"Authentic Image Augmentation")
    print(f"=" * 60)
    print(f"Source directory: {authentic_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Original authentic images: {num_original}")
    print(f"Target total count: {args.target_count}")
    print(f"Augmentation strength: {args.strength}")
    print(f"=" * 60)
    
    if num_original == 0:
        print("ERROR: No authentic images found!")
        sys.exit(1)
    
    # Calculate how many augmented images we need
    num_augments_needed = args.target_count - num_original
    if num_augments_needed <= 0:
        print(f"Already have {num_original} images, no augmentation needed.")
        return
    
    # Calculate augments per image (distribute evenly)
    augments_per_image = num_augments_needed // num_original
    extra_augments = num_augments_needed % num_original
    
    print(f"Augments per image: {augments_per_image}")
    print(f"Extra augments for first {extra_augments} images")
    print(f"Total augmented images to generate: {num_augments_needed}")
    print()
    
    # Create augmentation pipeline
    transform = create_augmentation_pipeline(args.strength)
    
    # First, copy original images to output directory
    print("Step 1: Copying original authentic images...")
    for img_path in tqdm(authentic_images, desc="Copying originals"):
        output_path = output_dir / img_path.name
        if not output_path.exists():
            image = cv2.imread(str(img_path))
            cv2.imwrite(str(output_path), image)
    
    # Generate augmented images
    print("\nStep 2: Generating augmented images...")
    total_generated = 0
    aug_idx = 0
    
    for i, img_path in enumerate(tqdm(authentic_images, desc="Augmenting")):
        # Determine how many augments for this image
        num_augs = augments_per_image + (1 if i < extra_augments else 0)
        
        if num_augs > 0:
            generated = augment_and_save(img_path, output_dir, num_augs, transform, aug_idx)
            total_generated += generated
            aug_idx += num_augs
    
    # Summary
    final_count = len(list(output_dir.glob('*.png'))) + len(list(output_dir.glob('*.jpg')))
    
    print()
    print(f"=" * 60)
    print(f"Augmentation Complete!")
    print(f"=" * 60)
    print(f"Original images: {num_original}")
    print(f"Augmented images generated: {total_generated}")
    print(f"Total images in output directory: {final_count}")
    print(f"Output directory: {output_dir}")
    print(f"=" * 60)
    
    # Create a zero mask for authentic images (they have no forgery)
    print("\nStep 3: Creating zero masks for authentic images...")
    mask_output_dir = home_dir / 'train_masks'
    
    # Get image dimensions from first image
    sample_img = cv2.imread(str(authentic_images[0]))
    h, w = sample_img.shape[:2]
    
    # Create zero mask template
    zero_mask = np.zeros((h, w), dtype=np.float32)
    
    # Save masks for all augmented images
    augmented_images = list(output_dir.glob('*.png')) + list(output_dir.glob('*.jpg'))
    masks_created = 0
    
    for img_path in tqdm(augmented_images, desc="Creating masks"):
        mask_name = img_path.stem + '.npy'
        mask_path = mask_output_dir / mask_name
        
        if not mask_path.exists():
            # Get actual image dimensions (in case they vary)
            img = cv2.imread(str(img_path))
            if img is not None:
                img_h, img_w = img.shape[:2]
                mask = np.zeros((img_h, img_w), dtype=np.float32)
                np.save(str(mask_path), mask)
                masks_created += 1
    
    print(f"Zero masks created: {masks_created}")
    print(f"Mask directory: {mask_output_dir}")
    print()
    print("✅ Done! The augmented authentic images are ready for training.")
    print()
    print("Next steps:")
    print("1. Update training scripts to include authentic_augmented directory")
    print("2. Or move images to train_images/authentic/ directory")
    print("3. Re-run the training pipeline")


if __name__ == '__main__':
    main()
