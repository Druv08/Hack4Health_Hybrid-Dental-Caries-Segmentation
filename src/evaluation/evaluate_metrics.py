"""
Evaluation Metrics for Dental Caries Segmentation
==================================================
This module computes standard medical image segmentation metrics:
- Dice Coefficient (F1 Score for segmentation)
- Intersection over Union (IoU / Jaccard Index)

These metrics are computed by comparing model predictions against
ground truth binary masks.

IMPORTANT:
- Predictions must be thresholded before metric computation
- Ground truth masks must be binary (0 = background, 1 = caries)
- Both masks must have the same spatial dimensions

Author: Hack4Health Team
"""

import os
import numpy as np
import cv2
from typing import Tuple, List, Dict


# =============================================================================
# CONFIGURABLE PARAMETERS
# =============================================================================
DEFAULT_THRESHOLD = 0.25  # Probability threshold for binary conversion


# =============================================================================
# CORE METRIC FUNCTIONS
# =============================================================================

def dice_score(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> float:
    """
    Compute Dice Coefficient between prediction and ground truth.
    
    Formula: Dice = 2 * |intersection| / (|pred| + |gt|)
    
    Args:
        pred: Binary prediction mask (0 or 1)
        gt: Binary ground truth mask (0 or 1)
        eps: Small epsilon to avoid division by zero
    
    Returns:
        Dice coefficient in range [0.0, 1.0]
        - 0.0 = no overlap
        - 1.0 = perfect overlap
    
    Medical Interpretation:
        - Dice > 0.7 is generally considered good for medical segmentation
        - Dice > 0.8 is considered excellent
    """
    # Ensure binary and float for computation
    pred = pred.astype(np.float32)
    gt = gt.astype(np.float32)
    
    # Flatten to 1D for robust computation
    pred_flat = pred.flatten()
    gt_flat = gt.flatten()
    
    # Compute intersection and sums
    intersection = np.sum(pred_flat * gt_flat)
    sum_pred = np.sum(pred_flat)
    sum_gt = np.sum(gt_flat)
    
    # Dice formula with epsilon for numerical stability
    dice = (2.0 * intersection + eps) / (sum_pred + sum_gt + eps)
    
    return float(dice)


def iou_score(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> float:
    """
    Compute Intersection over Union (IoU / Jaccard Index).
    
    Formula: IoU = |intersection| / |union|
           = |intersection| / (|pred| + |gt| - |intersection|)
    
    Args:
        pred: Binary prediction mask (0 or 1)
        gt: Binary ground truth mask (0 or 1)
        eps: Small epsilon to avoid division by zero
    
    Returns:
        IoU score in range [0.0, 1.0]
        - 0.0 = no overlap
        - 1.0 = perfect overlap
    
    Medical Interpretation:
        - IoU > 0.5 is generally acceptable
        - IoU > 0.7 is considered good
    """
    # Ensure binary and float for computation
    pred = pred.astype(np.float32)
    gt = gt.astype(np.float32)
    
    # Flatten to 1D
    pred_flat = pred.flatten()
    gt_flat = gt.flatten()
    
    # Compute intersection and union
    intersection = np.sum(pred_flat * gt_flat)
    union = np.sum(pred_flat) + np.sum(gt_flat) - intersection
    
    # IoU formula with epsilon
    iou = (intersection + eps) / (union + eps)
    
    return float(iou)


def precision_score(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> float:
    """
    Compute Precision: Of all predicted positives, how many are correct?
    
    Formula: Precision = TP / (TP + FP)
    """
    pred = pred.astype(np.float32).flatten()
    gt = gt.astype(np.float32).flatten()
    
    true_positives = np.sum(pred * gt)
    predicted_positives = np.sum(pred)
    
    return float((true_positives + eps) / (predicted_positives + eps))


def recall_score(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> float:
    """
    Compute Recall (Sensitivity): Of all actual positives, how many did we find?
    
    Formula: Recall = TP / (TP + FN)
    
    High recall is critical in medical screening - we don't want to miss caries.
    """
    pred = pred.astype(np.float32).flatten()
    gt = gt.astype(np.float32).flatten()
    
    true_positives = np.sum(pred * gt)
    actual_positives = np.sum(gt)
    
    return float((true_positives + eps) / (actual_positives + eps))


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def probability_to_binary(prob_map: np.ndarray, threshold: float = 0.25) -> np.ndarray:
    """
    Convert probability map to binary mask using threshold.
    
    Args:
        prob_map: Probability values in range [0.0, 1.0]
        threshold: Cutoff value (pixels >= threshold become 1)
    
    Returns:
        Binary mask with values 0 or 1
    """
    return (prob_map >= threshold).astype(np.uint8)


def load_mask(mask_path: str, target_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """
    Load and preprocess a mask image.
    
    Args:
        mask_path: Path to mask image file
        target_size: (width, height) to resize to
    
    Returns:
        Binary mask with values 0 or 1
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to load mask: {mask_path}")
    
    mask = cv2.resize(mask, target_size)
    
    # Normalize to binary (0 or 1)
    mask = (mask > 127).astype(np.uint8)
    
    return mask


def validate_masks(pred: np.ndarray, gt: np.ndarray) -> None:
    """
    Validate that prediction and ground truth masks are compatible.
    
    Raises:
        ValueError: If masks have incompatible shapes or values
    """
    if pred.shape != gt.shape:
        raise ValueError(
            f"Shape mismatch: prediction {pred.shape} vs ground truth {gt.shape}"
        )
    
    # Check for binary values
    pred_unique = np.unique(pred)
    gt_unique = np.unique(gt)
    
    if not np.all(np.isin(pred_unique, [0, 1])):
        print(f"[WARNING] Prediction contains non-binary values: {pred_unique}")
    
    if not np.all(np.isin(gt_unique, [0, 1])):
        print(f"[WARNING] Ground truth contains non-binary values: {gt_unique}")


# =============================================================================
# BATCH EVALUATION
# =============================================================================

def evaluate_single(
    pred: np.ndarray,
    gt: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD
) -> Dict[str, float]:
    """
    Evaluate a single prediction against ground truth.
    
    Args:
        pred: Probability map or binary prediction
        gt: Binary ground truth mask
        threshold: Threshold for converting probabilities to binary
    
    Returns:
        Dictionary with Dice, IoU, Precision, Recall scores
    """
    # Convert probability to binary if needed
    if pred.max() <= 1.0 and pred.dtype in [np.float32, np.float64]:
        pred_binary = probability_to_binary(pred, threshold)
    else:
        pred_binary = (pred > 0).astype(np.uint8)
    
    # Ensure ground truth is binary
    gt_binary = (gt > 0).astype(np.uint8)
    
    # Validate shapes
    validate_masks(pred_binary, gt_binary)
    
    # Compute all metrics
    return {
        'dice': dice_score(pred_binary, gt_binary),
        'iou': iou_score(pred_binary, gt_binary),
        'precision': precision_score(pred_binary, gt_binary),
        'recall': recall_score(pred_binary, gt_binary)
    }


def evaluate_dataset(
    predictions: List[np.ndarray],
    ground_truths: List[np.ndarray],
    threshold: float = DEFAULT_THRESHOLD,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Evaluate model performance over an entire dataset.
    
    Args:
        predictions: List of probability maps or binary predictions
        ground_truths: List of binary ground truth masks
        threshold: Threshold for converting probabilities to binary
        verbose: Print per-image scores
    
    Returns:
        Dictionary with mean Dice, IoU, Precision, Recall
    """
    if len(predictions) != len(ground_truths):
        raise ValueError(
            f"Mismatch: {len(predictions)} predictions vs {len(ground_truths)} ground truths"
        )
    
    all_dice = []
    all_iou = []
    all_precision = []
    all_recall = []
    
    for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
        scores = evaluate_single(pred, gt, threshold)
        
        all_dice.append(scores['dice'])
        all_iou.append(scores['iou'])
        all_precision.append(scores['precision'])
        all_recall.append(scores['recall'])
        
        if verbose:
            print(f"  Image {i+1}: Dice={scores['dice']:.4f}, IoU={scores['iou']:.4f}")
    
    # Compute means
    results = {
        'mean_dice': float(np.mean(all_dice)),
        'mean_iou': float(np.mean(all_iou)),
        'mean_precision': float(np.mean(all_precision)),
        'mean_recall': float(np.mean(all_recall)),
        'std_dice': float(np.std(all_dice)),
        'std_iou': float(np.std(all_iou)),
        'per_image_dice': all_dice,
        'per_image_iou': all_iou
    }
    
    return results


def print_evaluation_report(results: Dict[str, float]) -> None:
    """
    Print a formatted evaluation report.
    """
    print("\n" + "=" * 50)
    print("SEGMENTATION EVALUATION REPORT")
    print("=" * 50)
    print(f"  Mean Dice Coefficient:  {results['mean_dice']:.4f} (+/- {results['std_dice']:.4f})")
    print(f"  Mean IoU (Jaccard):     {results['mean_iou']:.4f} (+/- {results['std_iou']:.4f})")
    print(f"  Mean Precision:         {results['mean_precision']:.4f}")
    print(f"  Mean Recall:            {results['mean_recall']:.4f}")
    print("=" * 50)
    
    # Interpretation
    dice = results['mean_dice']
    if dice >= 0.8:
        quality = "EXCELLENT"
    elif dice >= 0.7:
        quality = "GOOD"
    elif dice >= 0.5:
        quality = "MODERATE"
    else:
        quality = "NEEDS IMPROVEMENT"
    
    print(f"  Overall Quality: {quality}")
    print("=" * 50 + "\n")


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    """
    Example: Evaluate predictions against ground truth masks.
    """
    print("Dental Caries Segmentation - Evaluation Demo")
    print("-" * 45)
    
    # Create synthetic example data for demonstration
    # In real usage, load actual predictions and ground truth masks
    
    # Example 1: Good prediction (high overlap)
    gt1 = np.zeros((256, 256), dtype=np.uint8)
    gt1[100:150, 100:150] = 1  # Ground truth caries region
    
    pred1 = np.zeros((256, 256), dtype=np.float32)
    pred1[105:148, 102:148] = 0.8  # Model prediction (slightly shifted)
    
    # Example 2: Moderate prediction
    gt2 = np.zeros((256, 256), dtype=np.uint8)
    gt2[50:100, 50:100] = 1
    
    pred2 = np.zeros((256, 256), dtype=np.float32)
    pred2[60:110, 55:95] = 0.6
    
    # Evaluate
    predictions = [pred1, pred2]
    ground_truths = [gt1, gt2]
    
    print("\nEvaluating predictions with threshold = 0.25...")
    results = evaluate_dataset(predictions, ground_truths, threshold=0.25)
    print_evaluation_report(results)
    
    # Single image evaluation example
    print("\nSingle image evaluation:")
    single_scores = evaluate_single(pred1, gt1, threshold=0.25)
    print(f"  Dice: {single_scores['dice']:.4f}")
    print(f"  IoU:  {single_scores['iou']:.4f}")
    print(f"  Precision: {single_scores['precision']:.4f}")
    print(f"  Recall: {single_scores['recall']:.4f}")
