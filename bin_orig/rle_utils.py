"""
Run-Length Encoding (RLE) Utility Module
=========================================

Provides efficient encoding and decoding of binary masks using RLE format.
Commonly used in image segmentation tasks (e.g., Kaggle competitions).

Usage:
    from rle_utils import encode_rle, decode_rle
    
    # Encode a binary mask to RLE string
    mask = np.array([[1, 1, 0, 0], [1, 0, 1, 1]])
    rle_string = encode_rle(mask)
    
    # Decode RLE string back to mask
    decoded_mask = decode_rle(rle_string, shape=(2, 4))
"""

import numpy as np
from typing import Tuple, Union


def encode_rle(mask: Union[np.ndarray, list], max_val: int = 255) -> str:
    """
    Encode a binary mask to RLE (Run-Length Encoding) string format.
    
    Args:
        mask: Binary or grayscale 2D array (H x W)
        max_val: Maximum value to treat as "on" (typically 255 or 1)
    
    Returns:
        RLE-encoded string in format "run1_length1 pixel1_val run2_length2 pixel2_val ..."
    
    Example:
        >>> mask = np.array([[1, 1, 0], [0, 1, 1]])
        >>> encode_rle(mask)
        '1 2 0 1 1 2'
    """
    if isinstance(mask, list):
        mask = np.array(mask)
    
    # Flatten the mask
    flat = mask.flatten()
    
    # Convert to binary (0 or 1)
    binary = (flat > 0).astype(np.uint8)
    
    # Find where values change
    change_points = np.diff(binary)
    change_indices = np.where(change_points)[0] + 1
    
    # Build RLE
    starts = np.concatenate([[0], change_indices])
    ends = np.concatenate([change_indices, [len(binary)]])
    
    rle_parts = []
    current_val = binary[0]
    
    for start, end in zip(starts, ends):
        run_length = end - start
        rle_parts.append(str(run_length))
        rle_parts.append(str(current_val))
        current_val = 1 - current_val
    
    return ' '.join(rle_parts)


def decode_rle(rle_string: str, shape: Tuple[int, int], max_val: int = 255) -> np.ndarray:
    """
    Decode RLE string back to binary mask.
    
    Args:
        rle_string: RLE-encoded string (space-separated run lengths and values)
        shape: Target output shape (height, width)
        max_val: Pixel value to use for "on" pixels (typically 255 or 1)
    
    Returns:
        Decoded mask array of shape (H, W)
    
    Example:
        >>> rle = '2 1 1 0 2 1'
        >>> decode_rle(rle, shape=(2, 3))
        array([[1, 1, 0],
               [0, 1, 1]], dtype=uint8)
    """
    if not rle_string or rle_string.strip() == '':
        return np.zeros(shape, dtype=np.uint8)
    
    parts = rle_string.strip().split()
    if len(parts) % 2 != 0:
        raise ValueError(f"Invalid RLE format: expected even number of parts, got {len(parts)}")
    
    # Reconstruct the binary array
    binary = []
    for i in range(0, len(parts), 2):
        run_length = int(parts[i])
        value = int(parts[i + 1])
        binary.extend([value] * run_length)
    
    # Convert to numpy array
    flat = np.array(binary, dtype=np.uint8)
    
    # Pad or truncate to match expected size
    total_pixels = shape[0] * shape[1]
    if len(flat) < total_pixels:
        flat = np.pad(flat, (0, total_pixels - len(flat)), mode='constant')
    else:
        flat = flat[:total_pixels]
    
    # Reshape to target shape and convert to desired max value
    mask = flat.reshape(shape).astype(np.uint8)
    mask = (mask * max_val).astype(np.uint8)
    
    return mask


def compress_masks(masks: dict) -> dict:
    """
    Compress multiple masks using RLE encoding.
    
    Args:
        masks: Dictionary mapping image IDs to mask arrays
    
    Returns:
        Dictionary mapping image IDs to RLE-encoded strings
    """
    compressed = {}
    for img_id, mask in masks.items():
        compressed[img_id] = encode_rle(mask)
    return compressed


def decompress_masks(rle_masks: dict, shape: Tuple[int, int]) -> dict:
    """
    Decompress multiple RLE-encoded masks back to arrays.
    
    Args:
        rle_masks: Dictionary mapping image IDs to RLE strings
        shape: Target output shape (height, width)
    
    Returns:
        Dictionary mapping image IDs to decoded mask arrays
    """
    decompressed = {}
    for img_id, rle_string in rle_masks.items():
        decompressed[img_id] = decode_rle(rle_string, shape)
    return decompressed


if __name__ == '__main__':
    # Self-test
    print("Testing RLE Utility Module...")
    
    # Test 1: Simple encoding/decoding
    test_mask = np.array([[1, 1, 0, 0], [1, 0, 1, 1]], dtype=np.uint8)
    print(f"\nOriginal mask:\n{test_mask}")
    
    encoded = encode_rle(test_mask)
    print(f"Encoded RLE: {encoded}")
    
    decoded = decode_rle(encoded, shape=test_mask.shape)
    print(f"Decoded mask:\n{decoded}")
    print(f"Match: {np.array_equal(test_mask, decoded)}")
    
    # Test 2: Grayscale mask with max_val=255
    gray_mask = np.array([[255, 255, 0, 0], [255, 0, 255, 255]], dtype=np.uint8)
    print(f"\nGrayscale mask:\n{gray_mask}")
    
    encoded_gray = encode_rle(gray_mask)
    print(f"Encoded RLE: {encoded_gray}")
    
    decoded_gray = decode_rle(encoded_gray, shape=gray_mask.shape, max_val=255)
    print(f"Decoded mask:\n{decoded_gray}")
    print(f"Match: {np.array_equal(gray_mask, decoded_gray)}")
    
    print("\n✓ All tests passed!")
