"""
Evaluation Module for Dental Caries Segmentation
=================================================
Contains metrics and evaluation utilities for assessing model performance.

Available Metrics:
- dice_score: Overlap measure (F1 for segmentation)
- iou_score: Intersection over Union (Jaccard Index)
- precision_score: Positive predictive value
- recall_score: Sensitivity (important for medical imaging)
- specificity_score: True negative rate
- accuracy_score: Pixel-wise accuracy
"""

from .metrics import (
    dice_score,
    iou_score,
    precision_score,
    recall_score,
    specificity_score,
    accuracy_score,
    compute_all_metrics,
    MetricTracker,
    print_metrics,
    binarize
)

__all__ = [
    'dice_score',
    'iou_score',
    'precision_score',
    'recall_score',
    'specificity_score',
    'accuracy_score',
    'compute_all_metrics',
    'MetricTracker',
    'print_metrics',
    'binarize'
]
