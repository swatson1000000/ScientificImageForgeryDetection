#!/bin/bash

DATASET_PATH="/home/swatson/work/MachineLearning/kaggle/ScientificImageForgeryDetection"
TRAIN_IMAGES="${DATASET_PATH}/train_images"
BACKUP_DIR="${DATASET_PATH}/train_images_backup_original"

echo "="*80
echo "SWAPPING AUTHENTIC/FORGED LABELS"
echo "="*80

# Create backup first
if [ ! -d "$BACKUP_DIR" ]; then
    echo ""
    echo "Creating backup of original structure..."
    cp -r "$TRAIN_IMAGES" "$BACKUP_DIR"
    echo "✓ Backup created at: $BACKUP_DIR"
fi

# Swap directories
echo ""
echo "Swapping directories..."
cd "$TRAIN_IMAGES" || exit 1

# Create temporary directory
mkdir -p temp_swap
mv authentic temp_swap/authentic_temp
mv forged authentic
mv temp_swap/authentic_temp forged
rmdir temp_swap

echo "✓ Swap complete!"
echo ""
echo "New structure:"
ls -d */ 

echo ""
echo "="*80
echo "IMPORTANT NOTE"
echo "="*80
echo ""
echo "The training data has been swapped:"
echo "  OLD: train_images/authentic/ -> NOW: train_images/forged/"
echo "  OLD: train_images/forged/   -> NOW: train_images/authentic/"
echo ""
echo "BACKUP: train_images_backup_original/"
echo ""
echo "NEXT: Retrain the models with the corrected labels"
echo ""

