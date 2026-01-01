#!/bin/bash
# Clean up generated data and logs (keeps directories)
# Usage: ./clean.sh [--hard-negative]
# Use --hard-negative flag to also remove hard_negative_ids.txt file

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CLEAN_HARD_NEGATIVES=false

# Parse command line arguments
for arg in "$@"; do
    if [ "$arg" = "--hard-negative" ]; then
        CLEAN_HARD_NEGATIVES=true
    fi
done

echo "Cleaning up..."

# Remove files from log directories
if [ -d "$SCRIPT_DIR/logs" ]; then
    rm -rf "$SCRIPT_DIR/logs"/*
    echo "Cleaned: $SCRIPT_DIR/logs"
fi

if [ -d "$PROJECT_DIR/log" ]; then
    rm -rf "$PROJECT_DIR/log"/*
    echo "Cleaned: $PROJECT_DIR/log"
fi

# Remove augmented forged images
if [ -d "$PROJECT_DIR/train_images/forged_augmented" ]; then
    rm -rf "$PROJECT_DIR/train_images/forged_augmented"/*
    echo "Cleaned: $PROJECT_DIR/train_images/forged_augmented"
fi

# Remove augmented authentic images
if [ -d "$PROJECT_DIR/train_images/authentic_augmented" ]; then
    rm -rf "$PROJECT_DIR/train_images/authentic_augmented"/*
    echo "Cleaned: $PROJECT_DIR/train_images/authentic_augmented"
fi

# Remove trained models
if [ -d "$PROJECT_DIR/models" ]; then
    rm -f "$PROJECT_DIR/models"/*.pth
    echo "Cleaned: $PROJECT_DIR/models (removed *.pth files)"
fi

# Remove predictions directory
if [ -d "$PROJECT_DIR/predictions" ]; then
    rm -rf "$PROJECT_DIR/predictions"/*
    echo "Cleaned: $PROJECT_DIR/predictions"
fi

# Remove hard negatives file (optional)
if [ "$CLEAN_HARD_NEGATIVES" = true ]; then
    if [ -f "$PROJECT_DIR/hard_negative_ids.txt" ]; then
        rm -f "$PROJECT_DIR/hard_negative_ids.txt"
        echo "Removed: $PROJECT_DIR/hard_negative_ids.txt"
    fi
else
    echo "Skipped: hard_negative_ids.txt (use --hard-negative flag to remove)"
fi

echo "Done."
