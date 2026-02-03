"""
Classification Metrics for Dental Caries Detection
===================================================
Computes classification metrics for binary (caries / no caries) task.

Metrics:
- Accuracy
- Precision
- Recall (Sensitivity)
- F1 Score
- Specificity

Author: Hack4Health Team
"""

import numpy as np
from typing import Dict, List, Tuple


def compute_confusion_matrix(
    y_true: List[int],
    y_pred: List[int]
) -> Tuple[int, int, int, int]:
    """
    Compute confusion matrix values.
    
    Returns:
        (TP, TN, FP, FN)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    
    return int(TP), int(TN), int(FP), int(FN)


def accuracy_score(y_true: List[int], y_pred: List[int]) -> float:
    """
    Compute accuracy: (TP + TN) / Total
    """
    TP, TN, FP, FN = compute_confusion_matrix(y_true, y_pred)
    total = TP + TN + FP + FN
    return (TP + TN) / total if total > 0 else 0.0


def precision_score(y_true: List[int], y_pred: List[int], eps: float = 1e-7) -> float:
    """
    Compute precision: TP / (TP + FP)
    Of all predicted caries, how many are actually caries?
    """
    TP, TN, FP, FN = compute_confusion_matrix(y_true, y_pred)
    return (TP + eps) / (TP + FP + eps)


def recall_score(y_true: List[int], y_pred: List[int], eps: float = 1e-7) -> float:
    """
    Compute recall (sensitivity): TP / (TP + FN)
    Of all actual caries, how many did we detect?
    Critical for medical screening - we don't want to miss caries!
    """
    TP, TN, FP, FN = compute_confusion_matrix(y_true, y_pred)
    return (TP + eps) / (TP + FN + eps)


def specificity_score(y_true: List[int], y_pred: List[int], eps: float = 1e-7) -> float:
    """
    Compute specificity: TN / (TN + FP)
    Of all healthy teeth, how many did we correctly identify as healthy?
    """
    TP, TN, FP, FN = compute_confusion_matrix(y_true, y_pred)
    return (TN + eps) / (TN + FP + eps)


def f1_score(y_true: List[int], y_pred: List[int], eps: float = 1e-7) -> float:
    """
    Compute F1 score: 2 * (precision * recall) / (precision + recall)
    Harmonic mean of precision and recall.
    """
    prec = precision_score(y_true, y_pred, eps)
    rec = recall_score(y_true, y_pred, eps)
    return 2 * (prec * rec) / (prec + rec + eps)


def compute_all_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """
    Compute all classification metrics.
    
    Args:
        y_true: Ground truth labels (0 = no caries, 1 = caries)
        y_pred: Predicted labels
    
    Returns:
        Dictionary with all metrics
    """
    TP, TN, FP, FN = compute_confusion_matrix(y_true, y_pred)
    
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'specificity': specificity_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'true_positives': TP,
        'true_negatives': TN,
        'false_positives': FP,
        'false_negatives': FN,
        'total_samples': TP + TN + FP + FN
    }


def print_classification_metrics(metrics: Dict[str, float]) -> None:
    """
    Print classification metrics in a formatted report.
    """
    print("\n" + "=" * 50)
    print("CLASSIFICATION METRICS REPORT")
    print("=" * 50)
    print(f"  Accuracy:     {metrics['accuracy']:.4f} ({metrics['accuracy']:.2%})")
    print(f"  Precision:    {metrics['precision']:.4f}")
    print(f"  Recall:       {metrics['recall']:.4f}")
    print(f"  Specificity:  {metrics['specificity']:.4f}")
    print(f"  F1 Score:     {metrics['f1_score']:.4f}")
    print("-" * 50)
    print("  Confusion Matrix:")
    print(f"    TP={metrics['true_positives']}  FN={metrics['false_negatives']}")
    print(f"    FP={metrics['false_positives']}  TN={metrics['true_negatives']}")
    print("-" * 50)
    print(f"  Total Samples: {metrics['total_samples']}")
    print("=" * 50)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    """
    Demo: Compute classification metrics on sample data.
    """
    print("Classification Metrics Demo")
    print("-" * 30)
    
    # Sample predictions (for demonstration)
    y_true = [1, 1, 0, 0, 1, 0, 1, 0, 1, 1]
    y_pred = [1, 0, 0, 0, 1, 1, 1, 0, 1, 1]
    
    metrics = compute_all_metrics(y_true, y_pred)
    print_classification_metrics(metrics)
