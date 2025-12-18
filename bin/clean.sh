#!/bin/bash
# Clean up generated data and logs (keeps directories)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

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

echo "Done."
