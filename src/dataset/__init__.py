"""
Dataset Module for Dental Caries Segmentation
==============================================
Contains data loading utilities, splitting functions, 
and PyTorch Dataset classes.
"""

# Data loading utilities
from .data_loader import (
    load_image_mask_pairs,
    load_image,
    load_mask,
    visualize_pair,
    visualize_overlay,
    get_dataset_statistics
)

# Dataset splitting utilities
from .split_dataset import (
    split_dataset,
    verify_split_integrity,
    get_split_summary
)

# PyTorch Dataset class
from .segmentation_dataset import (
    DentalSegmentationDataset,
    get_dataloaders,
    verify_dataset,
    visualize_sample
)

__all__ = [
    # Data loading
    'load_image_mask_pairs',
    'load_image',
    'load_mask',
    'visualize_pair',
    'visualize_overlay',
    'get_dataset_statistics',
    # Dataset splitting
    'split_dataset',
    'verify_split_integrity',
    'get_split_summary',
    # PyTorch Dataset
    'DentalSegmentationDataset',
    'get_dataloaders',
    'verify_dataset',
    'visualize_sample'
]
