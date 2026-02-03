"""
Data Loader Module for Dental Caries Segmentation
==================================================
This module provides utilities to load medical images (dental X-rays)
and their corresponding segmentation masks for training and evaluation.

Author: Hack4Health Team
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_image_mask_pairs(image_dir, mask_dir):
    """
    Loads and pairs dental X-ray images with their corresponding masks.
    
    This function scans both directories, matches files by filename,
    and validates that every image has a corresponding mask.

    Args:
        image_dir (str): Path to directory containing X-ray images
        mask_dir (str): Path to directory containing segmentation masks

    Returns:
        pairs (list): List of tuples (image_path, mask_path)
    
    Raises:
        ValueError: If any image is missing its corresponding mask or vice versa
    
    Example:
        >>> pairs = load_image_mask_pairs("data/raw/images", "data/raw/masks")
        >>> print(f"Found {len(pairs)} image-mask pairs")
    """

    # Step 1: List all files in both directories
    # sorted() ensures consistent ordering across different systems
    image_files = sorted(os.listdir(image_dir))
    mask_files = sorted(os.listdir(mask_dir))

    # Step 2: Convert lists to sets for efficient comparison
    image_set = set(image_files)
    mask_set = set(mask_files)

    # Step 3: Find any mismatches between images and masks
    # Set difference (-) finds items in one set but not the other
    missing_masks = image_set - mask_set  # Images without masks
    missing_images = mask_set - image_set  # Masks without images

    # Step 4: Raise clear errors if pairs are incomplete
    if missing_masks:
        raise ValueError(f"Masks missing for images: {missing_masks}")

    if missing_images:
        raise ValueError(f"Images missing for masks: {missing_images}")

    # Step 5: Build list of matched (image, mask) path tuples
    pairs = []

    for filename in image_files:
        image_path = os.path.join(image_dir, filename)
        mask_path = os.path.join(mask_dir, filename)
        pairs.append((image_path, mask_path))

    return pairs


def load_image(image_path, grayscale=True):
    """
    Load a single image from disk.
    
    Args:
        image_path (str): Path to the image file
        grayscale (bool): If True, load as grayscale (default for X-rays)
    
    Returns:
        np.ndarray: Loaded image as NumPy array
    
    Raises:
        ValueError: If image cannot be loaded
    """
    # Choose color mode based on grayscale flag
    mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    
    image = cv2.imread(image_path, mode)
    
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    return image


def load_mask(mask_path):
    """
    Load a segmentation mask from disk.
    
    Masks are always loaded as grayscale since they contain
    class labels (0 = background, 255 = caries region).
    
    Args:
        mask_path (str): Path to the mask file
    
    Returns:
        np.ndarray: Loaded mask as NumPy array
    
    Raises:
        ValueError: If mask cannot be loaded
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if mask is None:
        raise ValueError(f"Failed to load mask: {mask_path}")
    
    return mask


def visualize_pair(image_path, mask_path, save_path=None):
    """
    Displays an image and its corresponding mask side by side
    for visual verification (sanity check).
    
    This is essential in medical imaging to verify:
    1. Images and masks are correctly paired
    2. Masks accurately overlay the regions of interest
    3. No data corruption occurred during loading
    
    Args:
        image_path (str): Path to the X-ray image
        mask_path (str): Path to the segmentation mask
        save_path (str, optional): If provided, saves the figure instead of displaying
    
    Raises:
        ValueError: If image or mask cannot be loaded
    """

    # Load both image and mask in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Validate successful loading
    if image is None:
        raise ValueError(f"Error loading image: {image_path}")
    
    if mask is None:
        raise ValueError(f"Error loading mask: {mask_path}")

    # Create a figure with two side-by-side subplots
    plt.figure(figsize=(10, 5))

    # Left subplot: Original dental X-ray
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap="gray")
    plt.title("Dental X-ray")
    plt.axis("off")

    # Right subplot: Segmentation mask
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Segmentation Mask")
    plt.axis("off")

    plt.tight_layout()
    
    # Save or display the figure
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Visualization saved to: {save_path}")
    else:
        plt.show()


def visualize_overlay(image_path, mask_path, alpha=0.5, save_path=None):
    """
    Displays the mask overlaid on the original image.
    
    This helps visualize how well the mask aligns with
    the actual caries regions in the X-ray.
    
    Args:
        image_path (str): Path to the X-ray image
        mask_path (str): Path to the segmentation mask
        alpha (float): Transparency of the overlay (0-1)
        save_path (str, optional): If provided, saves the figure
    """
    
    # Load image and mask
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None or mask is None:
        raise ValueError("Error loading image or mask.")
    
    # Convert grayscale image to RGB for colored overlay
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Create a red overlay for the mask regions
    overlay = image_rgb.copy()
    overlay[mask > 127] = [255, 0, 0]  # Red color for caries regions
    
    # Blend original image with overlay
    blended = cv2.addWeighted(image_rgb, 1 - alpha, overlay, alpha, 0)
    
    # Display the result
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Original X-ray")
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='gray')
    plt.title("Mask")
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.imshow(blended)
    plt.title("Overlay")
    plt.axis("off")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def get_dataset_statistics(image_dir, mask_dir):
    """
    Compute basic statistics about the dataset.
    
    Useful for understanding data distribution and
    identifying potential issues before training.
    
    Args:
        image_dir (str): Path to images directory
        mask_dir (str): Path to masks directory
    
    Returns:
        dict: Dictionary containing dataset statistics
    """
    
    pairs = load_image_mask_pairs(image_dir, mask_dir)
    
    stats = {
        'total_pairs': len(pairs),
        'image_shapes': [],
        'mask_shapes': [],
        'mask_coverage': []  # Percentage of non-zero pixels in masks
    }
    
    for img_path, mask_path in pairs:
        image = load_image(img_path)
        mask = load_mask(mask_path)
        
        stats['image_shapes'].append(image.shape)
        stats['mask_shapes'].append(mask.shape)
        
        # Calculate mask coverage (percentage of caries region)
        coverage = np.sum(mask > 127) / mask.size * 100
        stats['mask_coverage'].append(coverage)
    
    # Summarize statistics
    stats['unique_image_shapes'] = list(set(stats['image_shapes']))
    stats['unique_mask_shapes'] = list(set(stats['mask_shapes']))
    stats['avg_mask_coverage'] = np.mean(stats['mask_coverage'])
    stats['min_mask_coverage'] = np.min(stats['mask_coverage'])
    stats['max_mask_coverage'] = np.max(stats['mask_coverage'])
    
    return stats


# =============================================================================
# MAIN EXECUTION - For testing the module independently
# =============================================================================

if __name__ == "__main__":
    """
    Test the data loader with sample data.
    
    Before running, ensure you have:
    1. Created data/raw/images/ directory with X-ray images
    2. Created data/raw/masks/ directory with matching masks
    """
    
    # Define paths to image and mask directories
    image_dir = "data/raw/images"
    mask_dir = "data/raw/masks"
    
    # Check if directories exist
    if not os.path.exists(image_dir):
        print(f"⚠️  Image directory not found: {image_dir}")
        print("   Please create the directory and add dental X-ray images.")
        exit(1)
        
    if not os.path.exists(mask_dir):
        print(f"⚠️  Mask directory not found: {mask_dir}")
        print("   Please create the directory and add segmentation masks.")
        exit(1)
    
    try:
        # Load all image-mask pairs
        pairs = load_image_mask_pairs(image_dir, mask_dir)
        print(f"✅ Total image-mask pairs found: {len(pairs)}")
        
        if len(pairs) > 0:
            # Visualize the first pair as a sanity check
            print(f"\n📸 Visualizing first pair:")
            print(f"   Image: {pairs[0][0]}")
            print(f"   Mask:  {pairs[0][1]}")
            visualize_pair(pairs[0][0], pairs[0][1])
            
    except ValueError as e:
        print(f"❌ Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
