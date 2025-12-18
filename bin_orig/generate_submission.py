#!/usr/bin/env python3
"""Quick submission generator - RLE encoding for forged, authentic for clean"""

import os
import sys
import numpy as np
import torch
import pandas as pd

import rle_utils as rle_compress

DATASET_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_PATH = os.path.join(DATASET_PATH, "models")
VAL_IMAGES_PATH = os.path.join(DATASET_PATH, "validation_images")
LOG_PATH = os.path.join(DATASET_PATH, "log")
OUTPUT_PATH = os.path.join(DATASET_PATH, "output")

# Ensure directories exist
os.makedirs(LOG_PATH, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Read the old submission that has RLE masks
old_df = pd.read_csv(os.path.join(OUTPUT_PATH, 'submission_ensemble.csv'))

print(f"Processing {len(old_df)} images...")
print(f"Columns: {old_df.columns.tolist()}")

# Generate new submission with mixed authentic/forged
new_rows = []

for idx, row in old_df.iterrows():
    if idx % 100 == 0:
        print(f"  Processed {idx}...")
    
    case_id = row['case_id']
    annotation = row['annotation']
    
    # Check if it has an RLE (forged) or is "authentic"
    # RLE strings typically contain numbers separated by spaces
    # "authentic" is just the word
    
    if annotation == 'authentic':
        new_rows.append({'case_id': case_id, 'annotation': 'authentic'})
    else:
        # It's an RLE string - keep it as is for forged images
        new_rows.append({'case_id': case_id, 'annotation': annotation})

print(f"  Total: {len(new_rows)}")

# Create new dataframe
new_df = pd.DataFrame(new_rows)

# Save
output_path = os.path.join(OUTPUT_PATH, 'submission_ensemble.csv')
new_df.to_csv(output_path, index=False)

# Stats
authentic_count = (new_df['annotation'] == 'authentic').sum()
forged_count = len(new_df) - authentic_count

print(f"\n✓ Submission saved: {output_path}")
print(f"  - Total images: {len(new_df)}")
print(f"  - Authentic: {authentic_count} ({authentic_count/len(new_df)*100:.1f}%)")
print(f"  - Forged: {forged_count} ({forged_count/len(new_df)*100:.1f}%)")

# Show sample
print(f"\nFirst 10 rows:")
print(new_df.head(10).to_string(index=False))
