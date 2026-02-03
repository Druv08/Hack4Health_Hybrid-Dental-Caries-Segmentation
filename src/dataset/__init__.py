"""
Dataset Module for Dental Caries Segmentation
==============================================
Contains data loading utilities and PyTorch Dataset classes.
"""

from .data_loader import (
    load_image_mask_pairs,
    load_image,
    load_mask,
    visualize_pair,
    visualize_overlay,
    get_dataset_statistics
)

__all__ = [
    'load_image_mask_pairs',
    'load_image',
    'load_mask',
    'visualize_pair',
    'visualize_overlay',
    'get_dataset_statistics'
]
