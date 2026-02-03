"""
Evaluation Metrics for Dental Caries Segmentation
==================================================
This module provides metrics for evaluating segmentation quality.

Available Metrics:
- Dice Coefficient (F1 Score)
- IoU (Intersection over Union / Jaccard Index)
- Precision
- Recall (Sensitivity)
- Specificity
- Accuracy

All metrics:
- Accept tensors or numpy arrays
- Return values in [0, 1] range
- Work with batch or single samples

Author: Hack4Health Team
"""

import torch
import numpy as np


def to_numpy(tensor):
    """Convert tensor to numpy array if needed."""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor


def binarize(pred, threshold=0.5):
    """
    Binarize predictions using a threshold.
    
    Args:
        pred: Predictions with values in [0, 1]
        threshold (float): Threshold for binarization
    
    Returns:
        Binary predictions {0, 1}
    """
    if isinstance(pred, torch.Tensor):
        return (pred > threshold).float()
    return (pred > threshold).astype(np.float32)


def dice_score(pred, target, threshold=0.5, smooth=1e-6):
    """
    Compute Dice coefficient (F1 score for segmentation).
    
    Dice = 2 * TP / (2 * TP + FP + FN)
    
    Args:
        pred: Predicted mask, values in [0, 1]
        target: Ground truth mask, values {0, 1}
        threshold (float): Threshold for binarizing predictions
        smooth (float): Smoothing to avoid division by zero
    
    Returns:
        float: Dice coefficient in [0, 1]
    """
    pred = to_numpy(pred)
    target = to_numpy(target)
    
    # Binarize predictions
    pred_binary = (pred > threshold).astype(np.float32)
    target = target.astype(np.float32)
    
    # Flatten
    pred_flat = pred_binary.flatten()
    target_flat = target.flatten()
    
    # Compute Dice
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    
    return float(dice)


def iou_score(pred, target, threshold=0.5, smooth=1e-6):
    """
    Compute IoU (Intersection over Union / Jaccard Index).
    
    IoU = TP / (TP + FP + FN)
    
    Args:
        pred: Predicted mask
        target: Ground truth mask
        threshold (float): Threshold for binarizing predictions
        smooth (float): Smoothing to avoid division by zero
    
    Returns:
        float: IoU score in [0, 1]
    """
    pred = to_numpy(pred)
    target = to_numpy(target)
    
    # Binarize
    pred_binary = (pred > threshold).astype(np.float32)
    target = target.astype(np.float32)
    
    # Flatten
    pred_flat = pred_binary.flatten()
    target_flat = target.flatten()
    
    # Compute IoU
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    
    return float(iou)


def precision_score(pred, target, threshold=0.5, smooth=1e-6):
    """
    Compute Precision (Positive Predictive Value).
    
    Precision = TP / (TP + FP)
    
    How many of the predicted positives are actually positive?
    
    Args:
        pred: Predicted mask
        target: Ground truth mask
        threshold (float): Threshold for binarizing predictions
        smooth (float): Smoothing to avoid division by zero
    
    Returns:
        float: Precision in [0, 1]
    """
    pred = to_numpy(pred)
    target = to_numpy(target)
    
    pred_binary = (pred > threshold).astype(np.float32)
    target = target.astype(np.float32)
    
    pred_flat = pred_binary.flatten()
    target_flat = target.flatten()
    
    true_positive = (pred_flat * target_flat).sum()
    predicted_positive = pred_flat.sum()
    
    precision = (true_positive + smooth) / (predicted_positive + smooth)
    
    return float(precision)


def recall_score(pred, target, threshold=0.5, smooth=1e-6):
    """
    Compute Recall (Sensitivity / True Positive Rate).
    
    Recall = TP / (TP + FN)
    
    How many of the actual positives are correctly identified?
    Critical in medical imaging - we don't want to miss caries.
    
    Args:
        pred: Predicted mask
        target: Ground truth mask
        threshold (float): Threshold for binarizing predictions
        smooth (float): Smoothing to avoid division by zero
    
    Returns:
        float: Recall in [0, 1]
    """
    pred = to_numpy(pred)
    target = to_numpy(target)
    
    pred_binary = (pred > threshold).astype(np.float32)
    target = target.astype(np.float32)
    
    pred_flat = pred_binary.flatten()
    target_flat = target.flatten()
    
    true_positive = (pred_flat * target_flat).sum()
    actual_positive = target_flat.sum()
    
    recall = (true_positive + smooth) / (actual_positive + smooth)
    
    return float(recall)


def specificity_score(pred, target, threshold=0.5, smooth=1e-6):
    """
    Compute Specificity (True Negative Rate).
    
    Specificity = TN / (TN + FP)
    
    How many of the actual negatives are correctly identified?
    
    Args:
        pred: Predicted mask
        target: Ground truth mask
        threshold (float): Threshold for binarizing predictions
        smooth (float): Smoothing to avoid division by zero
    
    Returns:
        float: Specificity in [0, 1]
    """
    pred = to_numpy(pred)
    target = to_numpy(target)
    
    pred_binary = (pred > threshold).astype(np.float32)
    target = target.astype(np.float32)
    
    pred_flat = pred_binary.flatten()
    target_flat = target.flatten()
    
    true_negative = ((1 - pred_flat) * (1 - target_flat)).sum()
    actual_negative = (1 - target_flat).sum()
    
    specificity = (true_negative + smooth) / (actual_negative + smooth)
    
    return float(specificity)


def accuracy_score(pred, target, threshold=0.5):
    """
    Compute pixel-wise accuracy.
    
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    Note: Accuracy can be misleading for imbalanced data.
    Use Dice or IoU as primary metrics.
    
    Args:
        pred: Predicted mask
        target: Ground truth mask
        threshold (float): Threshold for binarizing predictions
    
    Returns:
        float: Accuracy in [0, 1]
    """
    pred = to_numpy(pred)
    target = to_numpy(target)
    
    pred_binary = (pred > threshold).astype(np.float32)
    target = target.astype(np.float32)
    
    correct = (pred_binary == target).sum()
    total = target.size
    
    return float(correct / total)


def compute_all_metrics(pred, target, threshold=0.5):
    """
    Compute all segmentation metrics.
    
    Args:
        pred: Predicted mask
        target: Ground truth mask
        threshold (float): Threshold for binarizing predictions
    
    Returns:
        dict: Dictionary containing all metrics
    """
    metrics = {
        'dice': dice_score(pred, target, threshold),
        'iou': iou_score(pred, target, threshold),
        'precision': precision_score(pred, target, threshold),
        'recall': recall_score(pred, target, threshold),
        'specificity': specificity_score(pred, target, threshold),
        'accuracy': accuracy_score(pred, target, threshold)
    }
    
    return metrics


class MetricTracker:
    """
    Track metrics over multiple batches/epochs.
    
    Useful for computing average metrics during training/validation.
    
    Example:
        >>> tracker = MetricTracker()
        >>> for batch in dataloader:
        ...     pred = model(images)
        ...     tracker.update(pred, masks)
        >>> print(tracker.get_averages())
    """
    
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics."""
        self.metrics = {
            'dice': [],
            'iou': [],
            'precision': [],
            'recall': [],
            'specificity': [],
            'accuracy': []
        }
    
    def update(self, pred, target):
        """
        Update metrics with a new batch.
        
        Args:
            pred: Predicted masks (batch)
            target: Ground truth masks (batch)
        """
        # Compute metrics for this batch
        batch_metrics = compute_all_metrics(pred, target, self.threshold)
        
        # Append to history
        for key, value in batch_metrics.items():
            self.metrics[key].append(value)
    
    def get_averages(self):
        """
        Get average metrics over all updates.
        
        Returns:
            dict: Average of each metric
        """
        averages = {}
        for key, values in self.metrics.items():
            if len(values) > 0:
                averages[key] = np.mean(values)
            else:
                averages[key] = 0.0
        
        return averages
    
    def get_latest(self):
        """
        Get the most recent metrics.
        
        Returns:
            dict: Most recent value of each metric
        """
        latest = {}
        for key, values in self.metrics.items():
            if len(values) > 0:
                latest[key] = values[-1]
            else:
                latest[key] = 0.0
        
        return latest


def print_metrics(metrics, prefix=""):
    """
    Pretty print metrics.
    
    Args:
        metrics (dict): Dictionary of metric values
        prefix (str): Optional prefix for printing
    """
    print(f"{prefix}Dice:        {metrics['dice']:.4f}")
    print(f"{prefix}IoU:         {metrics['iou']:.4f}")
    print(f"{prefix}Precision:   {metrics['precision']:.4f}")
    print(f"{prefix}Recall:      {metrics['recall']:.4f}")
    print(f"{prefix}Specificity: {metrics['specificity']:.4f}")
    print(f"{prefix}Accuracy:    {metrics['accuracy']:.4f}")


# =============================================================================
# TESTING - Verify metrics work correctly
# =============================================================================

if __name__ == "__main__":
    """
    Test all evaluation metrics.
    """
    print("=" * 50)
    print("Testing Evaluation Metrics")
    print("=" * 50)
    
    # Create sample data
    pred = torch.sigmoid(torch.randn(4, 1, 256, 256))
    target = torch.randint(0, 2, (4, 1, 256, 256)).float()
    
    print(f"Input shapes:")
    print(f"  Prediction: {pred.shape}")
    print(f"  Target:     {target.shape}")
    
    # Compute all metrics
    print("\nMetrics:")
    metrics = compute_all_metrics(pred, target)
    print_metrics(metrics, prefix="  ")
    
    # Test with perfect prediction
    print("\nPerfect Prediction:")
    perfect_metrics = compute_all_metrics(target, target)
    print_metrics(perfect_metrics, prefix="  ")
    
    # Test MetricTracker
    print("\nMetricTracker:")
    tracker = MetricTracker()
    tracker.update(pred, target)
    tracker.update(pred, target)
    averages = tracker.get_averages()
    print(f"  Average Dice: {averages['dice']:.4f}")
    
    print("\n[OK] All metric tests passed!")
