import cv2
import numpy as np


def remove_small_components(mask, min_area=150):
    """
    Remove small isolated regions from a binary mask.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    cleaned = np.zeros_like(mask)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == i] = 255

    return cleaned


def smooth_mask(mask):
    """
    Smooth mask boundaries using morphological operations.
    """
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def postprocess_mask(mask):
    """
    Full post-processing pipeline.
    """
    mask = remove_small_components(mask)
    mask = smooth_mask(mask)
    return mask
