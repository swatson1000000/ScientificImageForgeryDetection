"""
Extract and display sample training images for visual inspection
"""
import os
import sys
import random
import shutil
from pathlib import Path

DATASET_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_IMAGES_PATH = os.path.join(DATASET_PATH, "train_images")

authentic_dir = os.path.join(TRAIN_IMAGES_PATH, "authentic")
forged_dir = os.path.join(TRAIN_IMAGES_PATH, "forged")

# Create inspection directories
inspect_dir = os.path.join(DATASET_PATH, "inspection")
inspect_auth = os.path.join(inspect_dir, "authentic_samples")
inspect_forge = os.path.join(inspect_dir, "forged_samples")

os.makedirs(inspect_auth, exist_ok=True)
os.makedirs(inspect_forge, exist_ok=True)

print("="*80)
print("PREPARING VISUAL INSPECTION SAMPLES")
print("="*80)

# Get all images
all_authentic = sorted([f for f in os.listdir(authentic_dir) if f.endswith('.png')])
all_forged = sorted([f for f in os.listdir(forged_dir) if f.endswith('.png')])

# Select 10 random from each
sample_auth = random.sample(all_authentic, min(10, len(all_authentic)))
sample_forge = random.sample(all_forged, min(10, len(all_forged)))

print(f"\nSelecting 10 random samples from each category...")
print(f"Total authentic available: {len(all_authentic)}")
print(f"Total forged available: {len(all_forged)}")

# Copy samples
print(f"\nCopying samples to inspection folder...")
for img in sample_auth:
    src = os.path.join(authentic_dir, img)
    dst = os.path.join(inspect_auth, img)
    shutil.copy2(src, dst)
    print(f"  ✓ {img}")

print()
for img in sample_forge:
    src = os.path.join(forged_dir, img)
    dst = os.path.join(inspect_forge, img)
    shutil.copy2(src, dst)
    print(f"  ✓ {img}")

print("\n" + "="*80)
print("INSPECTION COMPLETE")
print("="*80)

print(f"""
SAMPLES READY FOR INSPECTION:

Location: {inspect_dir}/

Folders:
  - authentic_samples/  (labeled as authentic in training)
  - forged_samples/     (labeled as forged in training)

INSTRUCTIONS:

1. Open file explorer or use terminal to view images:
   
   # View authentic samples
   ls -lh {inspect_auth}
   
   # View forged samples
   ls -lh {inspect_forge}

2. Open images in an image viewer to examine them

3. VISUAL INSPECTION QUESTIONS:

   For each sample, ask yourself:
   
   Q1: Does the image look naturally captured or artificially altered?
   Q2: Can you spot any obvious copy-paste areas?
   Q3: Are there color/lighting inconsistencies?
   Q4: Does it look like part of an image was spliced in?
   
4. TALLY YOUR OBSERVATIONS:

   Authentic samples (should look clean):
     - How many actually look clean?
     - How many look suspicious/tampered?
   
   Forged samples (should look tampered):
     - How many actually look tampered?
     - How many look clean/authentic?

5. REPORT BACK:
   
   Tell me:
     - For authentic_samples: X/10 look clean
     - For forged_samples: Y/10 look forged
   
   If X is low (<7) and Y is low (<7), labels are likely inverted!

""")

print("="*80)
