#!/usr/bin/env python3
"""
Generate augmented forged images with corresponding masks.

Creates multiple augmented versions of each forged image,
applying the same transformations to both image and mask.
"""

import os
import numpy as np
import cv2
from pathlib import Path
import albumentations as A
import random

# Configuration
NUM_AUGMENTATIONS_PER_IMAGE = 3  # Create 3 augmented versions per image
RANDOM_SEED = 42

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
FORGED_DIR = PROJECT_DIR / 'train_images' / 'forged'
MASKS_DIR = PROJECT_DIR / 'train_masks'
OUTPUT_IMAGES_DIR = PROJECT_DIR / 'train_images' / 'forged_augmented'
OUTPUT_MASKS_DIR = PROJECT_DIR / 'train_masks_augmented'

# Create output directories
OUTPUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_MASKS_DIR.mkdir(parents=True, exist_ok=True)

# Augmentation pipeline - geometric + color transforms
# These are applied identically to image and mask
augmentation = A.Compose([
    A.OneOf([
        A.HorizontalFlip(p=1.0),
        A.VerticalFlip(p=1.0),
        A.RandomRotate90(p=1.0),
        A.Rotate(limit=15, p=1.0, border_mode=cv2.BORDER_REFLECT),
    ], p=0.8),
    A.OneOf([
        A.RandomResizedCrop(size=(512, 512), scale=(0.8, 1.0), p=1.0),
        A.RandomScale(scale_limit=(-0.2, 0.2), p=1.0),
    ], p=0.3),
    A.OneOf([
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=1.0),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=1.0),
    ], p=0.5),
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        A.GaussNoise(std_range=(0.01, 0.02), p=1.0),
        A.ImageCompression(quality_range=(85, 95), p=1.0),
    ], p=0.3),
])

def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    print("=" * 60)
    print("GENERATING AUGMENTED FORGED IMAGES")
    print("=" * 60)
    print(f"Source: {FORGED_DIR}")
    print(f"Output images: {OUTPUT_IMAGES_DIR}")
    print(f"Output masks: {OUTPUT_MASKS_DIR}")
    print(f"Augmentations per image: {NUM_AUGMENTATIONS_PER_IMAGE}")
    print()
    
    # Get all forged images with masks
    forged_images = sorted(FORGED_DIR.glob('*.png'))
    print(f"Found {len(forged_images)} forged images")
    
    generated_count = 0
    skipped_count = 0
    
    for i, img_path in enumerate(forged_images):
        # Check if mask exists
        mask_path = MASKS_DIR / f"{img_path.stem}.npy"
        if not mask_path.exists():
            skipped_count += 1
            continue
        
        # Load image and mask
        image = cv2.imread(str(img_path))
        if image is None:
            skipped_count += 1
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = np.load(mask_path)
        if mask.ndim == 3:
            mask = mask[0]
        
        # Resize mask to match image if needed
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Generate augmented versions
        for aug_idx in range(NUM_AUGMENTATIONS_PER_IMAGE):
            try:
                # Apply augmentation
                transformed = augmentation(image=image, mask=mask)
                aug_image = transformed['image']
                aug_mask = transformed['mask']
                
                # Skip if mask is now empty (forgery cropped out)
                if aug_mask.max() < 0.1:
                    continue
                
                # Save augmented image
                aug_name = f"{img_path.stem}_aug{aug_idx}"
                out_img_path = OUTPUT_IMAGES_DIR / f"{aug_name}.png"
                out_mask_path = OUTPUT_MASKS_DIR / f"{aug_name}.npy"
                
                cv2.imwrite(str(out_img_path), cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
                np.save(out_mask_path, aug_mask.astype(np.float32))
                
                generated_count += 1
                
            except Exception as e:
                print(f"Error augmenting {img_path.name} (aug {aug_idx}): {e}")
                continue
    
    print()
    print("=" * 60)
    print("AUGMENTATION COMPLETE")
    print("=" * 60)
    print(f"Original forged images: {len(forged_images)}")
    print(f"Skipped (no mask or error): {skipped_count}")
    print(f"Augmented images generated: {generated_count}")
    print(f"Output: {OUTPUT_IMAGES_DIR}")
    print(f"Masks: {OUTPUT_MASKS_DIR}")
    
    # Show total forged data now
    total_forged = len(forged_images) + generated_count
    print(f"\nTotal forged data available: {total_forged}")


if __name__ == '__main__':
    main()
