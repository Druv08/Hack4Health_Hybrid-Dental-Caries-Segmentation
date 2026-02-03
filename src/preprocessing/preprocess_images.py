"""
Classical Image Preprocessing Module for Dental Caries Segmentation
====================================================================
This module applies classical image processing techniques to enhance
dental X-ray images before feeding them to the deep learning model.

Preprocessing Pipeline:
1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
2. Gaussian/Median filtering for noise reduction
3. Normalization to [0, 1] range

Why Classical Preprocessing Matters:
- Dental X-rays often have low contrast in carious regions
- Radiographic noise can confuse the segmentation model
- Consistent intensity range improves training stability

Author: Hack4Health Team
"""

import os
import cv2
import numpy as np


def create_dir(path):
    """
    Create directory if it does not exist.
    
    Args:
        path (str): Path to the directory to create
    """
    os.makedirs(path, exist_ok=True)


def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    CLAHE enhances local contrast while preventing over-amplification
    of noise. This is particularly useful for dental X-rays where
    carious lesions may have subtle intensity differences.
    
    Args:
        image (np.ndarray): Grayscale input image (uint8)
        clip_limit (float): Threshold for contrast limiting (default: 2.0)
            - Higher values = more contrast but potential noise amplification
            - Lower values = more natural appearance but less enhancement
        tile_grid_size (tuple): Size of grid for histogram equalization
            - Smaller tiles = more local adaptation
            - (8, 8) is a good balance for dental X-rays
    
    Returns:
        np.ndarray: Contrast-enhanced image (uint8)
    
    Note:
        Unlike global histogram equalization, CLAHE divides the image
        into small tiles and equalizes each independently, preventing
        over-enhancement in uniform regions.
    """
    # Create CLAHE object with specified parameters
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )
    
    # Apply CLAHE to the image
    enhanced = clahe.apply(image)
    
    return enhanced


def apply_filter(image, filter_type="gaussian", kernel_size=5):
    """
    Apply noise reduction filtering to the image.
    
    Dental X-rays often contain radiographic noise that can interfere
    with accurate segmentation. Filtering reduces this noise while
    (ideally) preserving important edge information.
    
    Args:
        image (np.ndarray): Input image (uint8 or float32)
        filter_type (str): Type of filter to apply
            - "gaussian": Good for general noise reduction
            - "median": Better for salt-and-pepper noise, preserves edges
            - "bilateral": Preserves edges while smoothing (slower)
        kernel_size (int): Size of the filter kernel (must be odd)
    
    Returns:
        np.ndarray: Filtered image
    
    Raises:
        ValueError: If unsupported filter type is specified
    """
    if filter_type == "gaussian":
        # Gaussian blur: weighted average favoring center pixels
        # Good for general noise reduction
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    elif filter_type == "median":
        # Median filter: replaces each pixel with median of neighborhood
        # Excellent for salt-and-pepper noise, preserves edges better
        return cv2.medianBlur(image, kernel_size)
    
    elif filter_type == "bilateral":
        # Bilateral filter: considers both spatial and intensity differences
        # Best edge preservation but computationally expensive
        return cv2.bilateralFilter(image, kernel_size, 75, 75)
    
    else:
        raise ValueError(f"Unsupported filter type: {filter_type}. "
                        f"Use 'gaussian', 'median', or 'bilateral'.")


def normalize_image(image):
    """
    Normalize image pixel values to range [0, 1].
    
    Normalization ensures consistent input range for the neural network,
    which improves training stability and convergence.
    
    Args:
        image (np.ndarray): Input image (any dtype)
    
    Returns:
        np.ndarray: Normalized image with values in [0, 1] (float32)
    
    Note:
        For deep learning, normalized inputs help:
        - Prevent gradient explosion/vanishing
        - Allow consistent learning rates across datasets
        - Improve model generalization
    """
    # Convert to float32 for precision
    image = image.astype(np.float32)
    
    # Scale from [0, 255] to [0, 1]
    normalized = image / 255.0
    
    return normalized


def preprocess_single_image(image, clip_limit=2.0, filter_type="gaussian"):
    """
    Apply the full preprocessing pipeline to a single image.
    
    Pipeline order:
    1. CLAHE for contrast enhancement
    2. Filtering for noise reduction
    3. Normalization to [0, 1]
    
    Args:
        image (np.ndarray): Grayscale input image (uint8)
        clip_limit (float): CLAHE clip limit
        filter_type (str): Type of noise reduction filter
    
    Returns:
        np.ndarray: Preprocessed image (float32, range [0, 1])
    """
    # Step 1: Enhance contrast with CLAHE
    image = apply_clahe(image, clip_limit=clip_limit)
    
    # Step 2: Reduce noise with filtering
    image = apply_filter(image, filter_type=filter_type)
    
    # Step 3: Normalize to [0, 1]
    image = normalize_image(image)
    
    return image


def preprocess_and_save(input_dir, output_dir, clip_limit=2.0, 
                        filter_type="gaussian", save_as_uint8=True):
    """
    Apply preprocessing pipeline to all images in a directory.
    
    Processes each image and saves the result to the output directory.
    Original images are not modified.
    
    Args:
        input_dir (str): Path to directory containing input images
        output_dir (str): Path to directory for saving processed images
        clip_limit (float): CLAHE clip limit (default: 2.0)
        filter_type (str): Type of filter (default: "gaussian")
        save_as_uint8 (bool): If True, save as uint8 [0-255] for compatibility
                              If False, save as float (requires special format)
    
    Returns:
        int: Number of images processed
    """
    # Create output directory if it doesn't exist
    create_dir(output_dir)
    
    # Get list of image files (excluding hidden files)
    image_files = [f for f in os.listdir(input_dir) if not f.startswith('.')]
    processed_count = 0
    
    for filename in image_files:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        # Load image in grayscale
        image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            print(f"[WARNING] Could not read image: {input_path}")
            continue
        
        # Apply preprocessing pipeline
        processed = preprocess_single_image(
            image, 
            clip_limit=clip_limit, 
            filter_type=filter_type
        )
        
        # Convert back to uint8 for saving (most image formats require this)
        if save_as_uint8:
            processed = (processed * 255).astype(np.uint8)
        
        # Save processed image
        cv2.imwrite(output_path, processed)
        processed_count += 1
    
    return processed_count


def run_preprocessing(split_dir, output_base, clip_limit=2.0, 
                      filter_type="gaussian"):
    """
    Run preprocessing for train, validation, and test splits.
    
    Processes images from each split and saves them to the output
    directory while maintaining the same folder structure.
    
    Args:
        split_dir (str): Base directory containing train/val/test splits
        output_base (str): Base directory for processed output
        clip_limit (float): CLAHE clip limit
        filter_type (str): Type of noise reduction filter
    
    Output Structure:
        output_base/
        ├── train/images/
        ├── val/images/
        └── test/images/
    """
    print("=" * 50)
    print("Starting Classical Preprocessing Pipeline")
    print("=" * 50)
    print(f"Input directory:  {split_dir}")
    print(f"Output directory: {output_base}")
    print(f"CLAHE clip limit: {clip_limit}")
    print(f"Filter type:      {filter_type}")
    print("-" * 50)
    
    total_processed = 0
    
    for split in ["train", "val", "test"]:
        input_images = os.path.join(split_dir, split, "images")
        output_images = os.path.join(output_base, split, "images")
        
        # Check if input directory exists
        if not os.path.exists(input_images):
            print(f"[WARNING] Skipping {split}: directory not found")
            continue
        
        # Process images
        count = preprocess_and_save(
            input_images, 
            output_images,
            clip_limit=clip_limit,
            filter_type=filter_type
        )
        
        print(f"[OK] {split}: {count} images processed")
        total_processed += count
    
    print("-" * 50)
    print(f"Total images processed: {total_processed}")
    print("Preprocessing complete!")
    
    return total_processed


def visualize_preprocessing(image_path, save_path=None):
    """
    Visualize the effect of each preprocessing step.
    
    Creates a side-by-side comparison showing:
    - Original image
    - After CLAHE
    - After filtering
    - Final normalized result
    
    Useful for tuning preprocessing parameters and
    demonstrating the pipeline to stakeholders.
    
    Args:
        image_path (str): Path to input image
        save_path (str, optional): If provided, saves the visualization
    """
    import matplotlib.pyplot as plt
    
    # Load original image
    original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if original is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Apply each step separately for visualization
    after_clahe = apply_clahe(original)
    after_filter = apply_filter(after_clahe, filter_type="gaussian")
    after_normalize = normalize_image(after_filter)
    
    # Create visualization
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Original
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # After CLAHE
    axes[1].imshow(after_clahe, cmap='gray')
    axes[1].set_title('After CLAHE')
    axes[1].axis('off')
    
    # After Filtering
    axes[2].imshow(after_filter, cmap='gray')
    axes[2].set_title('After Filtering')
    axes[2].axis('off')
    
    # Final Result
    axes[3].imshow(after_normalize, cmap='gray')
    axes[3].set_title('Final (Normalized)')
    axes[3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Visualization saved to: {save_path}")
    else:
        plt.show()


# =============================================================================
# MAIN EXECUTION - For running preprocessing directly
# =============================================================================

if __name__ == "__main__":
    """
    Run the preprocessing pipeline on split data.
    
    Before running, ensure you have:
    1. Split data in data/splits/ (from Step 2)
    2. Each split contains an 'images' subdirectory
    """
    
    # Define input and output paths
    split_data_dir = "data/splits"
    processed_data_dir = "data/processed"
    
    # Check if split directory exists
    if not os.path.exists(split_data_dir):
        print(f"[ERROR] Split directory not found: {split_data_dir}")
        print("   Please run dataset splitting (Step 2) first.")
        exit(1)
    
    # Run preprocessing with default parameters
    # Adjust clip_limit and filter_type based on your data
    run_preprocessing(
        split_dir=split_data_dir,
        output_base=processed_data_dir,
        clip_limit=2.0,        # Default CLAHE clip limit
        filter_type="gaussian"  # Options: gaussian, median, bilateral
    )
