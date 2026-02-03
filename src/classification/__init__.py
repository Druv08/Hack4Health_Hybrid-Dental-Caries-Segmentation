"""
Classification Module for Dental Caries Detection
==================================================
Round 2: Binary classification (caries / no caries)

Components:
- feature_extraction: Extract features from segmentation masks
- classifier: Rule-based binary classifier
- classification_metrics: Accuracy, Precision, Recall, F1

Author: Hack4Health Team
"""

from .feature_extraction import (
    extract_features,
    features_to_vector,
    print_features
)

from .classifier import (
    CariesClassifier,
    get_label_name,
    print_classification_result
)

from .classification_metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    specificity_score,
    f1_score,
    compute_all_metrics,
    print_classification_metrics
)

__all__ = [
    # Feature extraction
    'extract_features',
    'features_to_vector',
    'print_features',
    
    # Classifier
    'CariesClassifier',
    'get_label_name',
    'print_classification_result',
    
    # Metrics
    'accuracy_score',
    'precision_score',
    'recall_score',
    'specificity_score',
    'f1_score',
    'compute_all_metrics',
    'print_classification_metrics',
]
