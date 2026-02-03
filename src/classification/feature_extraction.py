"""
Feature Extraction from Segmentation Masks
===========================================
Extracts numerical features from segmentation masks for classification.

Features extracted:
- Lesion area (pixel count)
- Lesion ratio (area / total pixels)
- Number of connected components
- Mean/max component size
- Centroid locations
- Bounding box statistics

Author: Hack4Health Team
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple


def extract_features(mask: np.ndarray, probability_map: np.ndarray = None) -> Dict[str, float]:
    """
    Extract classification features from a segmentation mask.
    
    Args:
        mask: Binary segmentation mask (0 or 255)
        probability_map: Optional raw probability map from model
    
    Returns:
        Dictionary of extracted features
    """
    # Ensure binary mask
    binary_mask = (mask > 127).astype(np.uint8)
    
    # Basic area statistics
    total_pixels = mask.size
    lesion_pixels = np.sum(binary_mask)
    lesion_ratio = lesion_pixels / total_pixels
    
    # Connected component analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )
    
    # Exclude background (label 0)
    num_components = num_labels - 1
    
    # Component size statistics
    if num_components > 0:
        component_sizes = stats[1:, cv2.CC_STAT_AREA]
        mean_component_size = np.mean(component_sizes)
        max_component_size = np.max(component_sizes)
        min_component_size = np.min(component_sizes)
        std_component_size = np.std(component_sizes) if num_components > 1 else 0
    else:
        mean_component_size = 0
        max_component_size = 0
        min_component_size = 0
        std_component_size = 0
    
    # Bounding box coverage
    if num_components > 0:
        all_x = stats[1:, cv2.CC_STAT_LEFT]
        all_y = stats[1:, cv2.CC_STAT_TOP]
        all_w = stats[1:, cv2.CC_STAT_WIDTH]
        all_h = stats[1:, cv2.CC_STAT_HEIGHT]
        
        min_x = np.min(all_x)
        min_y = np.min(all_y)
        max_x = np.max(all_x + all_w)
        max_y = np.max(all_y + all_h)
        
        bbox_area = (max_x - min_x) * (max_y - min_y)
        bbox_ratio = bbox_area / total_pixels
    else:
        bbox_area = 0
        bbox_ratio = 0
    
    # Probability statistics (if available)
    if probability_map is not None:
        mean_prob = np.mean(probability_map)
        max_prob = np.max(probability_map)
        std_prob = np.std(probability_map)
        
        # Mean probability in lesion regions
        if lesion_pixels > 0:
            lesion_mean_prob = np.mean(probability_map[binary_mask > 0])
        else:
            lesion_mean_prob = 0
    else:
        mean_prob = 0
        max_prob = 0
        std_prob = 0
        lesion_mean_prob = 0
    
    features = {
        # Area features
        'lesion_pixels': float(lesion_pixels),
        'lesion_ratio': float(lesion_ratio),
        
        # Component features
        'num_components': float(num_components),
        'mean_component_size': float(mean_component_size),
        'max_component_size': float(max_component_size),
        'min_component_size': float(min_component_size),
        'std_component_size': float(std_component_size),
        
        # Bounding box features
        'bbox_area': float(bbox_area),
        'bbox_ratio': float(bbox_ratio),
        
        # Probability features
        'mean_probability': float(mean_prob),
        'max_probability': float(max_prob),
        'std_probability': float(std_prob),
        'lesion_mean_probability': float(lesion_mean_prob),
    }
    
    return features


def features_to_vector(features: Dict[str, float]) -> np.ndarray:
    """
    Convert feature dictionary to numpy array for classification.
    """
    feature_order = [
        'lesion_ratio',
        'num_components',
        'mean_component_size',
        'max_component_size',
        'bbox_ratio',
        'mean_probability',
        'max_probability',
        'lesion_mean_probability',
    ]
    
    return np.array([features.get(k, 0) for k in feature_order])


def print_features(features: Dict[str, float]) -> None:
    """
    Print extracted features in a readable format.
    """
    print("\n" + "=" * 50)
    print("EXTRACTED FEATURES")
    print("=" * 50)
    print(f"  Lesion Pixels:        {features['lesion_pixels']:.0f}")
    print(f"  Lesion Ratio:         {features['lesion_ratio']:.6f}")
    print(f"  Num Components:       {features['num_components']:.0f}")
    print(f"  Mean Component Size:  {features['mean_component_size']:.2f}")
    print(f"  Max Component Size:   {features['max_component_size']:.0f}")
    print(f"  Bounding Box Ratio:   {features['bbox_ratio']:.6f}")
    print(f"  Mean Probability:     {features['mean_probability']:.4f}")
    print(f"  Max Probability:      {features['max_probability']:.4f}")
    print(f"  Lesion Mean Prob:     {features['lesion_mean_probability']:.4f}")
    print("=" * 50)
