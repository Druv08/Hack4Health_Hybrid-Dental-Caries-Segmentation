"""
Preprocessing Module for Dental Caries Segmentation
====================================================
Contains classical image processing functions for enhancing
dental X-ray images before deep learning segmentation.

Pipeline:
1. CLAHE (contrast enhancement)
2. Filtering (noise reduction)
3. Normalization (intensity scaling)
"""

from .preprocess_images import (
    apply_clahe,
    apply_filter,
    normalize_image,
    preprocess_single_image,
    preprocess_and_save,
    run_preprocessing,
    visualize_preprocessing
)

__all__ = [
    'apply_clahe',
    'apply_filter',
    'normalize_image',
    'preprocess_single_image',
    'preprocess_and_save',
    'run_preprocessing',
    'visualize_preprocessing'
]
