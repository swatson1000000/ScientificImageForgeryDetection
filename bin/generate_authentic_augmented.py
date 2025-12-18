#!/usr/bin/env python3
"""
Generate More Augmented Authentic Images
=========================================
Creates additional augmented versions of authentic images to reduce false positives.
Uses diverse augmentations that mimic real-world image variations.

Usage:
    python generate_authentic_augmented.py --input <authentic_dir> --output <output_dir> --multiplier 3
"""

import os
import sys
import argparse
import random
from pathlib import Path
from PIL import Image
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add albumentations for augmentations
try:
    import albumentations as A
except ImportError:
    print("Installing albumentations...")
    os.system("pip install albumentations")
    import albumentations as A


def get_augmentation_pipeline(strength='medium'):
    """
    Create augmentation pipeline that mimics real-world image variations.
    These should NOT create forgery-like artifacts.
    """
    
    if strength == 'light':
        return A.Compose([
            A.OneOf([
                A.HorizontalFlip(p=1.0),
                A.VerticalFlip(p=1.0),
                A.Rotate(limit=15, p=1.0),
            ], p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0),
                A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=10, p=1.0),
            ], p=0.3),
        ])
    
    elif strength == 'medium':
        return A.Compose([
            # Geometric transforms
            A.OneOf([
                A.HorizontalFlip(p=1.0),
                A.VerticalFlip(p=1.0),
                A.Rotate(limit=30, p=1.0),
                A.Affine(scale=(0.9, 1.1), rotate=(-15, 15), p=1.0),
            ], p=0.6),
            
            # Color/lighting variations (common in real images)
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=1.0),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=15, p=1.0),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=1.0),
            ], p=0.5),
            
            # Blur/noise (common in real photos)
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.GaussNoise(std_range=(0.02, 0.1), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.1, 0.3), p=1.0),
            ], p=0.3),
            
            # JPEG compression artifacts (very common)
            A.ImageCompression(quality_range=(70, 95), p=0.4),
        ])
    
    else:  # strong
        return A.Compose([
            # Geometric transforms
            A.OneOf([
                A.HorizontalFlip(p=1.0),
                A.VerticalFlip(p=1.0),
                A.Rotate(limit=45, p=1.0),
                A.Affine(scale=(0.8, 1.2), rotate=(-30, 30), shear=(-10, 10), p=1.0),
                A.Perspective(scale=(0.02, 0.05), p=1.0),
            ], p=0.7),
            
            # Color/lighting variations
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=20, p=1.0),
                A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.08, p=1.0),
                A.RandomGamma(gamma_limit=(80, 120), p=1.0),
                A.CLAHE(clip_limit=2.0, p=1.0),
            ], p=0.6),
            
            # Blur/noise
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MotionBlur(blur_limit=(3, 7), p=1.0),
                A.GaussNoise(std_range=(0.05, 0.15), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.4), p=1.0),
                A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
            ], p=0.4),
            
            # JPEG compression
            A.ImageCompression(quality_range=(50, 90), p=0.5),
            
            # Downscale then upscale (common in web images)
            A.Downscale(scale_range=(0.5, 0.9), p=0.2),
        ])


def augment_image(args):
    """Augment a single image with multiple variations."""
    img_path, output_dir, num_augments, strengths = args
    
    try:
        # Load image
        img = np.array(Image.open(img_path).convert('RGB'))
        
        base_name = Path(img_path).stem
        results = []
        
        for i in range(num_augments):
            # Pick random strength for variety
            strength = random.choice(strengths)
            transform = get_augmentation_pipeline(strength)
            
            # Apply augmentation
            augmented = transform(image=img)['image']
            
            # Save
            out_name = f"{base_name}_aug{i+1}_{strength}.jpg"
            out_path = output_dir / out_name
            
            Image.fromarray(augmented).save(out_path, quality=95)
            results.append(out_name)
        
        return img_path, results, None
    
    except Exception as e:
        return img_path, [], str(e)


def main():
    parser = argparse.ArgumentParser(description='Generate augmented authentic images')
    parser.add_argument('--input', type=str, required=True,
                        help='Directory containing authentic images')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for augmented images')
    parser.add_argument('--multiplier', type=int, default=3,
                        help='Number of augmented versions per image (default: 3)')
    parser.add_argument('--strengths', type=str, default='light,medium,strong',
                        help='Comma-separated augmentation strengths to use')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of source images (for testing)')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    strengths = args.strengths.split(',')
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all images
    extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
    images = [f for f in input_dir.iterdir() 
              if f.suffix.lower() in extensions]
    
    if args.limit:
        images = images[:args.limit]
    
    print(f"=" * 60)
    print("AUTHENTIC IMAGE AUGMENTATION")
    print(f"=" * 60)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Source images: {len(images)}")
    print(f"Augmentations per image: {args.multiplier}")
    print(f"Strengths: {strengths}")
    print(f"Total images to generate: {len(images) * args.multiplier}")
    print()
    
    # Prepare tasks
    tasks = [(str(img), output_dir, args.multiplier, strengths) for img in images]
    
    # Process with progress
    completed = 0
    errors = 0
    total_generated = 0
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(augment_image, task): task for task in tasks}
        
        for future in as_completed(futures):
            completed += 1
            img_path, results, error = future.result()
            
            if error:
                errors += 1
                print(f"  ERROR: {Path(img_path).name}: {error}")
            else:
                total_generated += len(results)
    
    print()
    print(f"=" * 60)
    print("COMPLETE")
    print(f"=" * 60)
    print(f"Source images processed: {completed}")
    print(f"Augmented images generated: {total_generated}")
    print(f"Errors: {errors}")
    print(f"Output: {output_dir}")
    print()


if __name__ == '__main__':
    main()
