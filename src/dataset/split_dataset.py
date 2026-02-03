"""
Dataset Splitting Module for Dental Caries Segmentation
========================================================
This module splits paired image-mask data into train, validation, 
and test sets while preserving the pairing relationship.

Splitting Strategy:
- Train: 70% (for model learning)
- Validation: 15% (for hyperparameter tuning)
- Test: 15% (for final evaluation)

Author: Hack4Health Team
"""

import os
import shutil
import random


def create_dir(path):
    """
    Create directory if it does not exist.
    
    Uses os.makedirs with exist_ok=True to:
    - Create all intermediate directories if needed
    - Not raise an error if directory already exists
    
    Args:
        path (str): Path to the directory to create
    """
    os.makedirs(path, exist_ok=True)


def split_dataset(
    image_dir,
    mask_dir,
    output_dir,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42
):
    """
    Splits paired images and masks into train, validation, and test sets.
    
    CRITICAL: Image-mask pairing is preserved throughout the split.
    This prevents data leakage and ensures valid evaluation.
    
    Args:
        image_dir (str): Path to source images directory
        mask_dir (str): Path to source masks directory
        output_dir (str): Path to output directory for splits
        train_ratio (float): Proportion for training set (default: 0.7)
        val_ratio (float): Proportion for validation set (default: 0.15)
        test_ratio (float): Proportion for test set (default: 0.15)
        seed (int): Random seed for reproducibility (default: 42)
    
    Output Structure:
        output_dir/
        ├── train/
        │   ├── images/
        │   └── masks/
        ├── val/
        │   ├── images/
        │   └── masks/
        └── test/
            ├── images/
            └── masks/
    
    Raises:
        AssertionError: If ratios don't sum to 1.0 or if image-mask mismatch
    
    Example:
        >>> split_dataset("data/raw/images", "data/raw/masks", "data/splits")
        Dataset split completed successfully.
        Train: 70
        Validation: 15
        Test: 15
    """

    # =========================================================================
    # STEP 1: Validate split ratios
    # =========================================================================
    # Ratios must sum to 1.0 (100%) to use all data
    ratio_sum = train_ratio + val_ratio + test_ratio
    assert abs(ratio_sum - 1.0) < 1e-9, \
        f"Ratios must sum to 1.0, got {ratio_sum}"

    # =========================================================================
    # STEP 2: Load and validate file lists
    # =========================================================================
    # sorted() ensures consistent ordering across different operating systems
    image_files = sorted(os.listdir(image_dir))
    mask_files = sorted(os.listdir(mask_dir))

    # Ensure every image has a corresponding mask and vice versa
    # This is CRITICAL for medical imaging - mismatched data leads to
    # incorrect training and unreliable model predictions
    image_set = set(image_files)
    mask_set = set(mask_files)
    
    if image_set != mask_set:
        missing_masks = image_set - mask_set
        missing_images = mask_set - image_set
        error_msg = "Image-mask mismatch detected!\n"
        if missing_masks:
            error_msg += f"  Images without masks: {missing_masks}\n"
        if missing_images:
            error_msg += f"  Masks without images: {missing_images}"
        raise AssertionError(error_msg)

    # =========================================================================
    # STEP 3: Shuffle data with fixed seed for reproducibility
    # =========================================================================
    # Setting a fixed seed ensures:
    # - Same split every time the script runs
    # - Experiment reproducibility
    # - Fair comparison between different model versions
    random.seed(seed)
    random.shuffle(image_files)

    # =========================================================================
    # STEP 4: Calculate split indices
    # =========================================================================
    total = len(image_files)
    train_end = int(train_ratio * total)
    val_end = train_end + int(val_ratio * total)

    # Create dictionary mapping split names to their file lists
    splits = {
        "train": image_files[:train_end],
        "val": image_files[train_end:val_end],
        "test": image_files[val_end:]
    }

    # =========================================================================
    # STEP 5: Copy files to their respective split directories
    # =========================================================================
    # We COPY instead of MOVE to preserve original data
    # This allows re-running the split with different ratios if needed
    
    for split_name, files in splits.items():
        # Create output directories for this split
        img_out = os.path.join(output_dir, split_name, "images")
        mask_out = os.path.join(output_dir, split_name, "masks")

        create_dir(img_out)
        create_dir(mask_out)

        # Copy each image-mask pair
        for filename in files:
            # Copy image
            src_image = os.path.join(image_dir, filename)
            dst_image = os.path.join(img_out, filename)
            shutil.copy(src_image, dst_image)
            
            # Copy corresponding mask (same filename ensures pairing)
            src_mask = os.path.join(mask_dir, filename)
            dst_mask = os.path.join(mask_out, filename)
            shutil.copy(src_mask, dst_mask)

    # =========================================================================
    # STEP 6: Print summary
    # =========================================================================
    print("=" * 50)
    print("Dataset split completed successfully!")
    print("=" * 50)
    print(f"Total samples: {total}")
    print(f"   Train:      {len(splits['train']):4d} ({len(splits['train'])/total*100:.1f}%)")
    print(f"   Validation: {len(splits['val']):4d} ({len(splits['val'])/total*100:.1f}%)")
    print(f"   Test:       {len(splits['test']):4d} ({len(splits['test'])/total*100:.1f}%)")
    print(f"\nOutput directory: {output_dir}")
    print(f"Random seed: {seed}")
    
    return splits


def verify_split_integrity(output_dir):
    """
    Verify that the split was performed correctly.
    
    Checks:
    1. All split directories exist
    2. Image-mask counts match in each split
    3. No duplicate files across splits (no data leakage)
    
    Args:
        output_dir (str): Path to the splits directory
    
    Returns:
        bool: True if verification passes
    """
    
    print("\nVerifying split integrity...")
    
    all_files = []
    
    for split_name in ["train", "val", "test"]:
        img_dir = os.path.join(output_dir, split_name, "images")
        mask_dir = os.path.join(output_dir, split_name, "masks")
        
        # Check directories exist
        if not os.path.exists(img_dir):
            print(f"[ERROR] Missing directory: {img_dir}")
            return False
            
        if not os.path.exists(mask_dir):
            print(f"[ERROR] Missing directory: {mask_dir}")
            return False
        
        # Check image-mask count match
        images = set(os.listdir(img_dir))
        masks = set(os.listdir(mask_dir))
        
        if images != masks:
            print(f"[ERROR] Image-mask mismatch in {split_name} split")
            return False
        
        print(f"   [OK] {split_name}: {len(images)} pairs verified")
        
        # Collect all files for overlap check
        all_files.extend(images)
    
    # Check for duplicates (data leakage)
    if len(all_files) != len(set(all_files)):
        print("[ERROR] Data leakage detected: duplicate files found across splits!")
        return False
    
    print("[OK] Split integrity verified - no data leakage detected!")
    return True


def get_split_summary(output_dir):
    """
    Get a summary of the dataset split.
    
    Args:
        output_dir (str): Path to the splits directory
    
    Returns:
        dict: Summary statistics for each split
    """
    
    summary = {}
    
    for split_name in ["train", "val", "test"]:
        img_dir = os.path.join(output_dir, split_name, "images")
        
        if os.path.exists(img_dir):
            files = os.listdir(img_dir)
            summary[split_name] = {
                "count": len(files),
                "files": files
            }
        else:
            summary[split_name] = {"count": 0, "files": []}
    
    return summary


# =============================================================================
# MAIN EXECUTION - For running the split script directly
# =============================================================================

if __name__ == "__main__":
    """
    Run the dataset splitting process.
    
    Before running, ensure you have:
    1. Dental X-ray images in data/raw/images/
    2. Corresponding masks in data/raw/masks/
    """
    
    # Define input and output paths
    image_dir = "data/raw/images"
    mask_dir = "data/raw/masks"
    output_dir = "data/splits"
    
    # Check if source directories exist and contain files
    if not os.path.exists(image_dir):
        print(f"[ERROR] Image directory not found: {image_dir}")
        print("   Please add dental X-ray images first.")
        exit(1)
        
    if not os.path.exists(mask_dir):
        print(f"[ERROR] Mask directory not found: {mask_dir}")
        print("   Please add segmentation masks first.")
        exit(1)
    
    # Check if there are actually files to split
    image_files = [f for f in os.listdir(image_dir) if not f.startswith('.')]
    
    if len(image_files) == 0:
        print(f"[ERROR] No images found in: {image_dir}")
        print("   Please add dental X-ray images before splitting.")
        exit(1)
    
    # Perform the split
    try:
        split_dataset(
            image_dir=image_dir,
            mask_dir=mask_dir,
            output_dir=output_dir,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=42  # Fixed seed for reproducibility
        )
        
        # Verify the split was successful
        verify_split_integrity(output_dir)
        
    except AssertionError as e:
        print(f"[ERROR] Validation Error: {e}")
    except Exception as e:
        print(f"[ERROR] Unexpected Error: {e}")
