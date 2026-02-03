"""
Segmentation Dataset Module for Dental Caries Detection
========================================================
This module provides a PyTorch Dataset class for loading dental X-ray
images and their corresponding segmentation masks.

The dataset loader:
- Loads preprocessed images from data/processed/
- Loads masks from data/splits/
- Ensures perfect image-mask alignment by filename
- Returns tensors ready for deep learning

Compatible with both VS Code (local) and Google Colab environments.

Author: Hack4Health Team
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class DentalSegmentationDataset(Dataset):
    """
    PyTorch Dataset class for dental X-ray image segmentation.
    
    This class loads preprocessed dental X-ray images and their
    corresponding binary segmentation masks for training, validation,
    or testing a segmentation model.
    
    Attributes:
        image_dir (str): Path to directory containing preprocessed images
        mask_dir (str): Path to directory containing segmentation masks
        image_size (tuple): Target size for resizing (height, width)
        filenames (list): Sorted list of image filenames
        transform (callable, optional): Optional transform for augmentation
    
    Example:
        >>> dataset = DentalSegmentationDataset(
        ...     image_dir="data/processed/train/images",
        ...     mask_dir="data/splits/train/masks",
        ...     image_size=(256, 256)
        ... )
        >>> image, mask = dataset[0]
        >>> print(image.shape, mask.shape)
        torch.Size([1, 256, 256]) torch.Size([1, 256, 256])
    """

    def __init__(self, image_dir, mask_dir, image_size=(256, 256), transform=None):
        """
        Initialize the dataset.
        
        Args:
            image_dir (str): Path to preprocessed images directory
            mask_dir (str): Path to segmentation masks directory
            image_size (tuple): Target size (height, width) for resizing
                               Default: (256, 256) - good balance of detail and memory
            transform (callable, optional): Optional transform to apply to both
                                           image and mask (for augmentation)
        
        Raises:
            ValueError: If directories don't exist or have mismatched files
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.transform = transform
        
        # Validate directories exist
        if not os.path.exists(image_dir):
            raise ValueError(f"Image directory not found: {image_dir}")
        if not os.path.exists(mask_dir):
            raise ValueError(f"Mask directory not found: {mask_dir}")
        
        # Get sorted list of filenames (excluding hidden files)
        self.filenames = sorted([
            f for f in os.listdir(image_dir) 
            if not f.startswith('.') and os.path.isfile(os.path.join(image_dir, f))
        ])
        
        # Validate that masks exist for all images
        mask_files = set(os.listdir(mask_dir))
        missing_masks = [f for f in self.filenames if f not in mask_files]
        
        if missing_masks:
            raise ValueError(f"Missing masks for images: {missing_masks[:5]}...")
        
        if len(self.filenames) == 0:
            raise ValueError(f"No images found in: {image_dir}")

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        
        Returns:
            int: Number of image-mask pairs
        """
        return len(self.filenames)

    def __getitem__(self, idx):
        """
        Get a single image-mask pair by index.
        
        This method:
        1. Loads the image and mask from disk
        2. Resizes both to the target size
        3. Normalizes the image to [0, 1]
        4. Binarizes the mask (0 or 1)
        5. Converts both to PyTorch tensors
        
        Args:
            idx (int): Index of the sample to retrieve
        
        Returns:
            tuple: (image_tensor, mask_tensor)
                - image_tensor: Shape (1, H, W), dtype float32, range [0, 1]
                - mask_tensor: Shape (1, H, W), dtype float32, values {0, 1}
        
        Raises:
            ValueError: If image or mask cannot be loaded
        """
        # Get filename for this index
        filename = self.filenames[idx]
        
        # Construct full paths
        image_path = os.path.join(self.image_dir, filename)
        mask_path = os.path.join(self.mask_dir, filename)
        
        # Load image in grayscale
        # Dental X-rays are naturally grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Validate successful loading
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        if mask is None:
            raise ValueError(f"Failed to load mask: {mask_path}")
        
        # Resize to target size
        # Using INTER_LINEAR for images (smooth interpolation)
        # Using INTER_NEAREST for masks (preserve binary values)
        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)
        
        # Normalize image to [0, 1] range
        image = image.astype(np.float32) / 255.0
        
        # Binarize mask: any non-zero value becomes 1
        # This handles masks saved as 0-255 or 0-1
        mask = (mask > 0).astype(np.float32)
        
        # Apply optional transforms (for data augmentation)
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        
        # Add channel dimension: (H, W) -> (1, H, W)
        # PyTorch expects (C, H, W) format for images
        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)
        
        # Convert to PyTorch tensors
        image_tensor = torch.from_numpy(image)
        mask_tensor = torch.from_numpy(mask)
        
        return image_tensor, mask_tensor
    
    def get_filename(self, idx):
        """
        Get the filename for a given index.
        
        Useful for saving predictions with matching filenames.
        
        Args:
            idx (int): Index of the sample
        
        Returns:
            str: Filename of the sample
        """
        return self.filenames[idx]


def get_dataloaders(processed_dir, splits_dir, batch_size=8, 
                    image_size=(256, 256), num_workers=0):
    """
    Create DataLoaders for train, validation, and test sets.
    
    This is a convenience function that creates all three DataLoaders
    with appropriate settings for training and evaluation.
    
    Args:
        processed_dir (str): Base directory for preprocessed images
                            (e.g., "data/processed")
        splits_dir (str): Base directory for masks
                         (e.g., "data/splits")
        batch_size (int): Number of samples per batch (default: 8)
        image_size (tuple): Target image size (default: (256, 256))
        num_workers (int): Number of worker processes for data loading
                          Use 0 for Colab compatibility (default: 0)
    
    Returns:
        dict: Dictionary containing 'train', 'val', 'test' DataLoaders
    
    Example:
        >>> loaders = get_dataloaders(
        ...     processed_dir="data/processed",
        ...     splits_dir="data/splits",
        ...     batch_size=8
        ... )
        >>> for images, masks in loaders['train']:
        ...     # Training loop
        ...     pass
    """
    dataloaders = {}
    
    for split in ['train', 'val', 'test']:
        # Paths for this split
        image_dir = os.path.join(processed_dir, split, 'images')
        mask_dir = os.path.join(splits_dir, split, 'masks')
        
        # Check if directories exist
        if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
            print(f"[WARNING] Skipping {split}: directories not found")
            continue
        
        # Create dataset
        dataset = DentalSegmentationDataset(
            image_dir=image_dir,
            mask_dir=mask_dir,
            image_size=image_size
        )
        
        # Create DataLoader
        # shuffle=True for training, False for validation/test
        shuffle = (split == 'train')
        
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()  # Faster GPU transfer
        )
        
        print(f"[OK] {split}: {len(dataset)} samples, {len(dataloaders[split])} batches")
    
    return dataloaders


def verify_dataset(dataset, num_samples=3):
    """
    Verify dataset loading by checking a few samples.
    
    Prints shape and value range information for validation.
    
    Args:
        dataset (DentalSegmentationDataset): Dataset to verify
        num_samples (int): Number of samples to check
    """
    print("\nDataset Verification")
    print("-" * 40)
    print(f"Total samples: {len(dataset)}")
    
    for i in range(min(num_samples, len(dataset))):
        image, mask = dataset[i]
        filename = dataset.get_filename(i)
        
        print(f"\nSample {i + 1}: {filename}")
        print(f"  Image shape: {image.shape}")
        print(f"  Image range: [{image.min():.3f}, {image.max():.3f}]")
        print(f"  Mask shape:  {mask.shape}")
        print(f"  Mask unique: {torch.unique(mask).tolist()}")
        print(f"  Mask coverage: {mask.mean().item() * 100:.2f}%")
    
    print("-" * 40)
    print("Verification complete!")


def visualize_sample(dataset, idx=0, save_path=None):
    """
    Visualize a sample from the dataset.
    
    Args:
        dataset (DentalSegmentationDataset): Dataset to visualize from
        idx (int): Index of sample to visualize
        save_path (str, optional): Path to save the visualization
    """
    import matplotlib.pyplot as plt
    
    image, mask = dataset[idx]
    filename = dataset.get_filename(idx)
    
    # Convert tensors to numpy for visualization
    # Remove channel dimension: (1, H, W) -> (H, W)
    image_np = image.squeeze().numpy()
    mask_np = mask.squeeze().numpy()
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Original image
    axes[0].imshow(image_np, cmap='gray')
    axes[0].set_title(f'Image: {filename}')
    axes[0].axis('off')
    
    # Mask
    axes[1].imshow(mask_np, cmap='gray')
    axes[1].set_title('Mask')
    axes[1].axis('off')
    
    # Overlay
    overlay = np.zeros((*image_np.shape, 3))
    overlay[:, :, 0] = image_np  # Red channel
    overlay[:, :, 1] = image_np  # Green channel
    overlay[:, :, 2] = image_np  # Blue channel
    overlay[:, :, 0] = np.where(mask_np > 0, 1.0, overlay[:, :, 0])  # Red for caries
    
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# =============================================================================
# MAIN EXECUTION - For testing the dataset loader
# =============================================================================

if __name__ == "__main__":
    """
    Test the dataset loader.
    
    Before running, ensure you have:
    1. Preprocessed images in data/processed/
    2. Masks in data/splits/
    """
    
    # Define paths
    processed_dir = "data/processed"
    splits_dir = "data/splits"
    
    print("=" * 50)
    print("Testing Dental Segmentation Dataset Loader")
    print("=" * 50)
    
    # Test with training data
    train_image_dir = os.path.join(processed_dir, "train", "images")
    train_mask_dir = os.path.join(splits_dir, "train", "masks")
    
    if not os.path.exists(train_image_dir):
        print(f"[ERROR] Preprocessed images not found: {train_image_dir}")
        print("   Please run preprocessing (Step 3) first.")
        exit(1)
    
    if not os.path.exists(train_mask_dir):
        print(f"[ERROR] Masks not found: {train_mask_dir}")
        print("   Please run dataset splitting (Step 2) first.")
        exit(1)
    
    try:
        # Create dataset
        dataset = DentalSegmentationDataset(
            image_dir=train_image_dir,
            mask_dir=train_mask_dir,
            image_size=(256, 256)
        )
        
        # Verify dataset
        verify_dataset(dataset)
        
        # Create DataLoaders
        print("\nCreating DataLoaders...")
        loaders = get_dataloaders(
            processed_dir=processed_dir,
            splits_dir=splits_dir,
            batch_size=8
        )
        
        # Test batch loading
        if 'train' in loaders:
            print("\nTesting batch loading...")
            images, masks = next(iter(loaders['train']))
            print(f"Batch images shape: {images.shape}")
            print(f"Batch masks shape:  {masks.shape}")
        
        print("\n[OK] Dataset loader working correctly!")
        
    except Exception as e:
        print(f"[ERROR] {e}")
